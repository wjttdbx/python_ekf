"""
NERM + EKF + SDRE 闭环仿真引擎

追踪星通过导航器（BaseNavigator 接口）估计相对状态，再由 SDRE 控制器计算推力。
支持 SDC-EKF / Jacobian-EKF / Square-Root UKF 可注入切换。
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

from aerospace.dynamics.nerm import OrbitalDynamics
from aerospace.control.sdre import SDREGameController
from aerospace.estimation.ekf import RelativeStateEKF
from aerospace.estimation.navigators import (
    BaseNavigator, SDCEKFNavigator, wrap_legacy_ekf,
)


@dataclass
class EKFSDRESimResult:
    """EKF+SDRE 仿真结果数据容器。

    Attributes
    ----------
    t              : (N,)      时间序列 (s)
    states         : (13, N)   真实状态历史 [X_p, X_e, nu]
    x_est_history  : (6, N)    导航器估计相对状态历史
    u_p_history    : (3, N)    追踪星推力历史
    u_e_history    : (3, N)    逃逸星推力历史
    dist_history   : (N,)      相对距离历史 (km)
    ekf_err_history: (6, N)    估计误差历史 (km, km/s)
    innov_history  : (m, N)    新息历史
    P_diag_history : (6, N)    协方差对角元素历史
    nis_history    : (N,) | None      NIS 历史
    nees_history   : (N,) | None      NEES 历史
    captured       : bool      是否在 early_stop 模式下捕获
    """
    t: np.ndarray
    states: np.ndarray
    x_est_history: np.ndarray
    u_p_history: np.ndarray
    u_e_history: np.ndarray
    dist_history: np.ndarray
    ekf_err_history: np.ndarray
    innov_history: np.ndarray
    P_diag_history: np.ndarray
    nis_history: np.ndarray
    nees_history: np.ndarray
    captured: bool = False


_VALID_OVERRIDE_KEYS = {"pos", "vel", "A"}
_VALID_OVERRIDE_VALS = {"estimated", "true"}
_VALID_PREDICTION_MAPS = {"euler", "flow"}


class EKFSDRESimulation:
    """NERM + 导航器 + SDRE 闭环仿真引擎。

    Parameters
    ----------
    dynamics      : OrbitalDynamics
    controller    : SDREGameController
    navigator     : BaseNavigator | None  可注入导航器（优先于 ekf）
    ekf           : RelativeStateEKF | None  旧版 EKF（自动包装为 SDCEKFNavigator）
    X_p0 / X_e0   : (6,) ndarray  追踪星/逃逸星初始绝对状态
    nu0           : float  初始真近点角 (rad)
    dt            : float  控制步长 (s)
    are_interval  : int    ARE 求解间隔步数
    capture_dist  : float  捕获距离阈值 (km)
    rng           : np.random.Generator | None  随机数生成器
    passive_evader : bool  逃逸星推力恒为零
    early_stop    : bool  首次进入 capture_dist 时终止仿真
    prediction_map : str  导航预测离散映射，'euler' 或 'flow'
    ctrl_state_override : dict | None  反事实通道配置 {pos/vel/A: estimated/true}
    """

    def __init__(self, dynamics: OrbitalDynamics, controller: SDREGameController,
                 navigator: BaseNavigator | None = None,
                 ekf: RelativeStateEKF | None = None,
                 X_p0: np.ndarray | None = None,
                 X_e0: np.ndarray | None = None,
                 nu0: float = 0.0, dt: float = 10.0,
                 are_interval: int = 1,
                 capture_dist: float = 0.1,
                 rng: np.random.Generator | None = None,
                 passive_evader: bool = True,
                 early_stop: bool = False,
                 prediction_map: str = "euler",
                 ctrl_state_override: dict | None = None):

        # ── 导航器：navigator 优先，ekf 作为后备 ──────────────────────
        if navigator is not None:
            self.navigator = navigator
        elif ekf is not None:
            self.navigator = wrap_legacy_ekf(ekf)
        else:
            raise ValueError("必须提供 navigator (BaseNavigator) 或 ekf (RelativeStateEKF)。")

        self.dynamics = dynamics
        self.controller = controller
        self.dt = dt
        self.are_interval = are_interval
        self.capture_dist = capture_dist
        self.rng = rng
        self.passive_evader = passive_evader
        self.early_stop = early_stop
        if prediction_map not in _VALID_PREDICTION_MAPS:
            raise ValueError(
                f"prediction_map 必须为 {_VALID_PREDICTION_MAPS}，收到: {prediction_map}"
            )
        self.prediction_map = prediction_map
        self.ctrl_override = ctrl_state_override
        self._validate_override()

        # 组装初始绝对状态 (13D)
        if X_p0 is None or X_e0 is None:
            raise ValueError("X_p0 和 X_e0 必须提供。")
        self.state0 = np.zeros(13)
        self.state0[0:6] = X_p0
        self.state0[6:12] = X_e0
        self.state0[12] = nu0

        self._B_ctrl = np.zeros((6, 3))
        self._B_ctrl[3:, :] = np.eye(3)

    # ── 控制状态覆盖校验 ─────────────────────────────────────────────

    def _validate_override(self) -> None:
        if self.ctrl_override is None:
            return
        for k, v in self.ctrl_override.items():
            if k not in _VALID_OVERRIDE_KEYS:
                raise ValueError(f"ctrl_state_override key 必须为 {_VALID_OVERRIDE_KEYS}，收到: {k}")
            if v not in _VALID_OVERRIDE_VALS:
                raise ValueError(f"ctrl_state_override value 必须为 {_VALID_OVERRIDE_VALS}，收到: {v}")

    # ── 五个纯步骤 ───────────────────────────────────────────────────

    def _get_orbital_context(self, nu: float) -> tuple[float, float, float]:
        """从真近点角提取轨道参数。"""
        return self.dynamics.get_orbital_params(nu)

    def _build_ctrl_state_and_A(self, state: np.ndarray,
                                 r_c: float, nu_dot: float, nu_ddot: float
                                 ) -> tuple[np.ndarray, np.ndarray]:
        """根据 ctrl_state_override 组合控制用相对状态和 A_SDC。

        Returns
        -------
        x_ctrl : (6,) ndarray  控制器使用的相对状态
        A_SDC_ctrl : (6,6) ndarray  控制器使用的 SDC 矩阵
        """
        X_p_true = state[0:6]
        x_true_rel = state[0:6] - state[6:12]
        x_est = self.navigator.x

        ov = self.ctrl_override or {}
        use_pos = ov.get("pos", "estimated")
        use_vel = ov.get("vel", "estimated")
        use_A = ov.get("A", "estimated")

        # 组装控制用相对状态
        pos_part = x_true_rel[:3] if use_pos == "true" else x_est[:3]
        vel_part = x_true_rel[3:] if use_vel == "true" else x_est[3:]
        x_ctrl = np.concatenate([pos_part, vel_part])

        # 组装控制用 A_SDC
        if use_A == "true":
            X_e_for_A = state[6:12]
        else:
            X_e_for_A = X_p_true - x_est
        A_SDC_ctrl = self.dynamics.get_SDC_matrix(X_p_true, X_e_for_A, r_c, nu_dot, nu_ddot)

        return x_ctrl, A_SDC_ctrl

    def _compute_control(self, state: np.ndarray, t: float, k: int,
                          r_c: float, nu_dot: float, nu_ddot: float
                          ) -> tuple[np.ndarray, np.ndarray]:
        """计算追踪星/逃逸星推力。"""
        x_ctrl, A_SDC_ctrl = self._build_ctrl_state_and_A(state, r_c, nu_dot, nu_ddot)
        solve_now = (k % self.are_interval == 0)
        x_true_rel = state[0:6] - state[6:12]
        u_p, u_e = self.controller.compute_control(
            A_SDC_ctrl, x_ctrl, t=t, solve_are=solve_now, x_rel_e=x_true_rel,
        )
        if self.passive_evader:
            u_e = np.zeros(3)
        return u_p, u_e

    def _propagate_true_state(self, state: np.ndarray,
                               u_p: np.ndarray, u_e: np.ndarray,
                               t: float, dt: float) -> np.ndarray:
        """通过 RK45 传播 13D 真实状态一个步长。"""
        sol = solve_ivp(
            self.dynamics.dynamics_13d, [t, t + dt], state,
            args=(u_p, u_e), method="RK45",
            rtol=1e-8, atol=1e-10,
        )
        return sol.y[:, -1]

    def _build_navigator_ctx(self, state: np.ndarray,
                              r_c: float, nu_dot: float, nu_ddot: float,
                              u_p: np.ndarray, u_e: np.ndarray
                              ) -> dict:
        """为导航器 predict() 构建上下文（A_SDC_fn 用于 Jacobian/UKF）。"""
        X_p_true = state[0:6]
        nu = float(state[12])
        du = u_p - u_e
        dt = self.dt
        dynamics = self.dynamics

        def A_SDC_fn(x_rel: np.ndarray) -> np.ndarray:
            X_e_est = X_p_true - x_rel
            return dynamics.get_SDC_matrix(X_p_true, X_e_est, r_c, nu_dot, nu_ddot)

        def euler_discrete(x_rel: np.ndarray) -> np.ndarray:
            A = A_SDC_fn(x_rel)
            return x_rel + dt * (A @ x_rel + self._B_ctrl @ du)

        def flow_discrete(x_rel: np.ndarray) -> np.ndarray:
            estimated_state = np.concatenate([X_p_true, X_p_true - x_rel, [nu]])
            derivative = dynamics.dynamics_13d
            k1 = derivative(0.0, estimated_state, u_p, u_e)
            k2 = derivative(0.5 * dt, estimated_state + 0.5 * dt * k1, u_p, u_e)
            k3 = derivative(0.5 * dt, estimated_state + 0.5 * dt * k2, u_p, u_e)
            k4 = derivative(dt, estimated_state + dt * k3, u_p, u_e)
            propagated = estimated_state + (dt / 6.0) * (
                k1 + 2.0 * k2 + 2.0 * k3 + k4
            )
            return propagated[0:6] - propagated[6:12]

        f_discrete = flow_discrete if self.prediction_map == "flow" else euler_discrete
        return {"A_SDC_fn": A_SDC_fn, "f_discrete": f_discrete}

    def _navigator_predict(self, state: np.ndarray,
                            A_SDC_nominal: np.ndarray,
                            u_p: np.ndarray, u_e: np.ndarray,
                            r_c: float, nu_dot: float, nu_ddot: float
                            ) -> tuple[np.ndarray, np.ndarray]:
        """调用导航器 predict，传入非线性传播上下文。"""
        ctx = self._build_navigator_ctx(state, r_c, nu_dot, nu_ddot, u_p, u_e)
        return self.navigator.predict(A_SDC_nominal, self._B_ctrl,
                                       u_p, u_e, self.dt, **ctx)

    def _generate_measurement(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """生成带噪声的测量值。返回 (z_true, z_meas)。"""
        z_true = RelativeStateEKF.measure(
            state[0:6], state[6:12],
            angle_only=self.navigator.angles_only,
        )
        if self.rng is not None:
            z_meas = z_true + self.rng.multivariate_normal(
                np.zeros(self.navigator.R.shape[0]), self.navigator.R)
        else:
            z_meas = z_true.copy()
        return z_true, z_meas

    # ── 主仿真循环 ───────────────────────────────────────────────────

    def run(self, t_end: float | None = None) -> EKFSDRESimResult:
        """执行闭环仿真。

        Parameters
        ----------
        t_end : float | None  仿真时长 (s)，默认 2 个轨道周期

        Returns
        -------
        EKFSDRESimResult
        """
        if t_end is None:
            t_end = 2.0 * self.dynamics.T_orbit

        N = int(t_end / self.dt)
        state = self.state0.copy()
        t = 0.0

        # 预分配
        t_hist        = np.zeros(N + 1)
        state_hist    = np.zeros((13, N + 1))
        x_est_hist    = np.zeros((6, N + 1))
        u_p_hist      = np.zeros((3, N + 1))
        u_e_hist      = np.zeros((3, N + 1))
        dist_hist     = np.zeros(N + 1)
        ekf_err_hist  = np.zeros((6, N + 1))
        innov_dim     = self.navigator.R.shape[0]
        innov_hist    = np.zeros((innov_dim, N + 1))
        P_diag_hist   = np.zeros((6, N + 1))
        nis_hist      = np.full(N + 1, np.nan)
        nees_hist     = np.full(N + 1, np.nan)

        def _record(k: int, st: np.ndarray, x_est: np.ndarray,
                    up: np.ndarray, ue: np.ndarray,
                    innov: np.ndarray | None = None,
                    nis: float = np.nan, nees: float = np.nan) -> None:
            t_hist[k]         = t
            state_hist[:, k]  = st
            x_est_hist[:, k]  = x_est
            u_p_hist[:, k]    = up
            u_e_hist[:, k]    = ue
            dist_hist[k]      = np.linalg.norm(st[0:3] - st[6:9])
            x_true_rel        = st[0:6] - st[6:12]
            ekf_err_hist[:, k] = x_est - x_true_rel
            P_diag_hist[:, k] = np.diag(self.navigator.P)
            if innov is not None:
                innov_hist[:, k] = innov
            nis_hist[k] = nis
            nees_hist[k] = nees

        # ── 第 0 步 ──────────────────────────────────────────────────
        u_p = np.zeros(3)
        u_e = np.zeros(3)
        _record(0, state, self.navigator.x, u_p, u_e)

        captured = False
        N_actual = N

        # ── 主循环 ───────────────────────────────────────────────────
        for k in range(N):
            nu = state[12]
            r_c, nu_dot, nu_ddot = self._get_orbital_context(nu)

            # 1. 控制计算（使用 t_k 状态）
            u_p, u_e = self._compute_control(state, t, k, r_c, nu_dot, nu_ddot)

            # 2. 导航器预测（使用 t_k 状态 + t_k 轨道参数）
            innov = None
            if self.rng is not None:
                X_p_true_tk = state[0:6]
                X_e_est = X_p_true_tk - self.navigator.x  # 均为 t_k
                A_SDC_nominal = self.dynamics.get_SDC_matrix(
                    X_p_true_tk, X_e_est, r_c, nu_dot, nu_ddot)
                x_priori, P_priori = self._navigator_predict(
                    state, A_SDC_nominal, u_p, u_e, r_c, nu_dot, nu_ddot)

            # 3. 真实状态传播到 t_{k+1}
            state = self._propagate_true_state(state, u_p, u_e, t, self.dt)
            t += self.dt

            # 4. 导航器更新（使用 t_{k+1} 测量）
            nis = np.nan
            nees = np.nan
            if self.rng is not None:
                _, z_meas = self._generate_measurement(state)
                innov = self.navigator.update(x_priori, P_priori, z_meas)

                # NIS = ν' @ S^{-1} @ ν
                S = getattr(self.navigator, '_last_S', None)
                if S is not None:
                    try:
                        nis = float(innov @ np.linalg.solve(S, innov))
                    except np.linalg.LinAlgError:
                        nis = np.nan

                # NEES = (x̂ - x)' @ P^{-1} @ (x̂ - x)
                x_true = state[0:6] - state[6:12]
                err_n = self.navigator.x - x_true
                try:
                    nees = float(err_n @ np.linalg.solve(self.navigator.P, err_n))
                except np.linalg.LinAlgError:
                    nees = np.nan
            else:
                self.navigator.x = state[0:6] - state[6:12]

            # 5. 记录
            _record(k + 1, state, self.navigator.x, u_p, u_e, innov, nis, nees)

            # 5. 早停判断（仅 early_stop=True 时启用）
            if self.early_stop and dist_hist[k + 1] < self.capture_dist:
                print(f"捕获！t = {t:.1f} s，相对距离 = {dist_hist[k+1]*1000:.1f} m")
                N_actual = k + 1
                captured = True
                break
        else:
            if self.early_stop:
                print(f"仿真结束，最终相对距离 = {dist_hist[N_actual]:.3f} km")

        sl = slice(0, N_actual + 1)
        return EKFSDRESimResult(
            t=t_hist[sl],
            states=state_hist[:, sl],
            x_est_history=x_est_hist[:, sl],
            u_p_history=u_p_hist[:, sl],
            u_e_history=u_e_hist[:, sl],
            dist_history=dist_hist[sl],
            ekf_err_history=ekf_err_hist[:, sl],
            innov_history=innov_hist[:, sl],
            P_diag_history=P_diag_hist[:, sl],
            nis_history=nis_hist[sl],
            nees_history=nees_hist[sl],
            captured=captured,
        )
