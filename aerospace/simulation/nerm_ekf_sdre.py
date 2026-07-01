"""
NERM + EKF + SDRE 闭环仿真引擎

追踪星通过 EKF 估计相对状态，再由 SDRE 控制器计算推力。
"""


from dataclasses import dataclass, field
import numpy as np
from scipy.integrate import solve_ivp

from aerospace.dynamics.nerm import OrbitalDynamics
from aerospace.control.sdre import SDREGameController
from aerospace.estimation.ekf import RelativeStateEKF


@dataclass
class EKFSDRESimResult:
    """EKF+SDRE 仿真结果数据容器。

    Attributes
    ----------
    t              : (N,)      时间序列 (s)
    states         : (13, N)   真实状态历史 [X_p, X_e, nu]
    x_est_history  : (6, N)    EKF 估计相对状态历史
    u_p_history    : (3, N)    追踪星推力历史
    u_e_history    : (3, N)    逃逸星推力历史
    dist_history   : (N,)      相对距离历史 (km)
    ekf_err_history: (6, N)    EKF 各分量估计误差历史 (km, km/s)
    innov_history  : (3, N)    EKF 新息历史 [δρ, δaz, δel]
    P_diag_history : (6, N)    EKF 协方差对角元素历史
    captured       : bool      是否成功追捕
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
    captured: bool = False


class EKFSDRESimulation:
    """NERM + EKF + SDRE 闭环仿真引擎。

    Parameters
    ----------
    dynamics   : OrbitalDynamics
    controller : SDREGameController
    ekf        : RelativeStateEKF
    X_p0       : (6,) ndarray  追踪星初始状态
    X_e0       : (6,) ndarray  逃逸星初始状态
    nu0        : float         初始真近点角 (rad)
    dt         : float         控制步长 (s)
    are_interval : int         ARE 求解间隔步数（1 = 每步求解）
    capture_dist : float       追捕成功距离阈值 (km)
    rng        : np.random.Generator | None  随机数生成器（None = 无噪声）
    """

    def __init__(self, dynamics: OrbitalDynamics, controller: SDREGameController,
                 ekf: RelativeStateEKF,
                 X_p0: np.ndarray, X_e0: np.ndarray,
                 nu0: float = 0.0, dt: float = 10.0,
                 are_interval: int = 1,
                 capture_dist: float = 0.1,
                 rng: np.random.Generator | None = None,
                 A_fixed: np.ndarray | None = None):
        self.dynamics = dynamics
        self.controller = controller
        self.ekf = ekf
        self.dt = dt
        self.are_interval = are_interval
        self.capture_dist = capture_dist
        self.rng = rng
        self.A_fixed = A_fixed

        self.state0 = np.zeros(13)
        self.state0[0:6] = X_p0
        self.state0[6:12] = X_e0
        self.state0[12] = nu0

        self._B_ctrl = np.zeros((6, 3))
        self._B_ctrl[3:, :] = np.eye(3)

    def run(self, t_end: float | None = None) -> EKFSDRESimResult:
        """执行闭环仿真。

        Parameters
        ----------
        t_end : float, optional  仿真时长 (s)，默认 2 个轨道周期

        Returns
        -------
        EKFSDRESimResult
        """
        if t_end is None:
            t_end = 2.0 * self.dynamics.T_orbit

        N = int(t_end / self.dt)
        state = self.state0.copy()
        t = 0.0

        t_hist        = np.zeros(N + 1)
        state_hist    = np.zeros((13, N + 1))
        x_est_hist    = np.zeros((6, N + 1))
        u_p_hist      = np.zeros((3, N + 1))
        u_e_hist      = np.zeros((3, N + 1))
        dist_hist     = np.zeros(N + 1)
        ekf_err_hist  = np.zeros((6, N + 1))
        innov_dim     = self.ekf.R.shape[0]
        innov_hist    = np.zeros((innov_dim, N + 1))
        P_diag_hist   = np.zeros((6, N + 1))

        def _record(k: int, st: np.ndarray, x_est: np.ndarray,
                    up: np.ndarray, ue: np.ndarray,
                    innov: np.ndarray | None = None) -> None:
            t_hist[k]         = t
            state_hist[:, k]  = st
            x_est_hist[:, k]  = x_est
            u_p_hist[:, k]    = up
            u_e_hist[:, k]    = ue
            dist_hist[k]      = np.linalg.norm(st[0:3] - st[6:9])
            x_true_rel        = st[0:6] - st[6:12]
            ekf_err_hist[:, k] = x_est - x_true_rel
            P_diag_hist[:, k] = np.diag(self.ekf.P)
            if innov is not None:
                innov_hist[:, k] = innov

        u_p = np.zeros(3)
        u_e = np.zeros(3)
        _record(0, state, self.ekf.x, u_p, u_e)

        captured = False
        N_actual = N

        for k in range(N):
            nu = state[12]
            r_c, nu_dot, nu_ddot = self.dynamics.get_orbital_params(nu)

            # ── SDRE 控制 ────────────────────────────────────────────────────
            # 追方知道自己的真实位置和EKF估计的相对状态
            X_p_true = state[0:6]
            if self.rng is not None:
                x_ctrl = self.ekf.x
            else:
                x_ctrl = state[0:6] - state[6:12]
            if self.A_fixed is not None:
                A_SDC = self.A_fixed
            else:
                # 从追方真实位置和相对估计推算逃方估计位置
                X_e_est = X_p_true - x_ctrl
                A_SDC = self.dynamics.get_SDC_matrix(X_p_true, X_e_est, r_c, nu_dot, nu_ddot)

            solve_now = (k % self.are_interval == 0)
            x_true_rel = state[0:6] - state[6:12]
            u_p, u_e = self.controller.compute_control(
                A_SDC, x_ctrl, t=t, solve_are=solve_now,
                x_rel_e=x_true_rel,
            )

            # ── 真实状态传播 ─────────────────────────────────────────────────
            sol = solve_ivp(
                self.dynamics.dynamics_13d, [t, t + self.dt], state,
                args=(u_p, u_e), method="RK45",
                rtol=1e-8, atol=1e-10,
            )
            state = sol.y[:, -1]
            t += self.dt

            # ── EKF 预测 & 更新（仅有噪声时运行）────────────────────────────
            innov = None
            if self.rng is not None:
                # EKF 预测复用控制步骤的 A_SDC(t_k)——SDC 框架下
                # 滤波与控制共享同一线性化点 x̂_{k|k}，无需重算。
                x_priori, P_priori = self.ekf.predict(A_SDC, self._B_ctrl, u_p, u_e, self.dt)

                _angle_only = self.ekf.angles_only
                _use_doppler = self.ekf.use_doppler
                z_true = RelativeStateEKF.measure(state[0:6], state[6:12],
                                                  angle_only=_angle_only,
                                                  use_doppler=_use_doppler)
                z_meas = z_true + self.rng.multivariate_normal(
                    np.zeros(self.ekf.R.shape[0]), self.ekf.R)
                innov = self.ekf.update(x_priori, P_priori, z_meas)
            else:
                self.ekf.x = state[0:6] - state[6:12]

            _record(k + 1, state, self.ekf.x, u_p, u_e, innov)

            if dist_hist[k + 1] < self.capture_dist:
                print(f"捕获！t = {t:.1f} s，相对距离 = {dist_hist[k+1]*1000:.1f} m")
                N_actual = k + 1
                captured = True
                break
        else:
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
            captured=captured,
        )
