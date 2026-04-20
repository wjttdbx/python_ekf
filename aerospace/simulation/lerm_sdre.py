"""
LERM + SDRE 追逃博弈闭环仿真引擎

将时变线性化相对运动方程（LERM）嵌入 SDRE 框架运行：
- 状态向量：6D 相对状态 x_rel = [x, y, z, vx, vy, vz]（km, km/s）
- 真近点角 ν：外部辅助标量，由开普勒方程更新，不是系统自由度
- 每个 ZOH 步：冻结 ν̇, ν̈ → 重建 A_LERM(t) → 求解 ARE → 积分 6D ODE → 更新 ν
- A_LERM 随 ν 变化（椭圆轨道），因此 P 矩阵随轨道位置周期性变化
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from aerospace.dynamics.lerm import LERMDynamics
from aerospace.control.sdre import SDREGameController


def _log_step(step: int, t: float, T_total: float,
              dist: float, u_p: np.ndarray, P_cur: np.ndarray | None,
              are_updated: bool = True) -> None:
    """输出统一格式的仿真进度日志。

    格式：[Step N]  t=XXX.Xs  进度=XX.X%  相对距离=X.XXXkm  |u_p|=X.XXe-XX  λ(P)=[...]  [ARE↑/P缓存]
    """
    if P_cur is not None:
        eigs = np.linalg.eigvalsh(P_cur)
        lam_str = f"λ(P)=[{eigs.min():.3e}, {eigs.max():.3e}]"
    else:
        lam_str = "λ(P)=[N/A, N/A]"
    are_tag = "[ARE↑]" if are_updated else "[P缓存]"
    print(
        f"[Step {step:4d}]  t={t:8.1f}s  进度={t / T_total * 100:5.1f}%  "
        f"相对距离={dist:.3f}km  "
        f"|u_p|={np.linalg.norm(u_p):.3e}  "
        f"{lam_str}  {are_tag}"
    )


@dataclass
class LERMSDRESimResult:
    """LERM-SDRE 仿真结果数据容器。

    Attributes
    ----------
    t          : (N,)      时间序列 (s)
    states     : (6, N)    相对状态历史 x_rel = X_p - X_e  [x,y,z,vx,vy,vz] (km, km/s)
    nu_history : (N,)      真近点角历史 (rad)，辅助量，不是自由度
    u_p_history: (3, N)    追踪者推力加速度历史 (km/s²)
    u_e_history: (3, N)    逃逸者推力加速度历史 (km/s²)
    P_history  : (N, 6, 6) 每步 ARE 解出的 P 矩阵历史
    X_p_history: (3, N)    追踪者在 LVLH 系中的绝对位置历史 (km)
    X_e_history: (3, N)    逃逸者在 LVLH 系中的绝对位置历史 (km)
    """

    t: np.ndarray
    states: np.ndarray
    nu_history: np.ndarray
    u_p_history: np.ndarray
    u_e_history: np.ndarray
    P_history: np.ndarray
    X_p_history: np.ndarray
    X_e_history: np.ndarray


class LERMSDRESimulation:
    """LERM 动力学 + SDRE 控制器的零和微分博弈闭环仿真。

    与 NERM+SDRE 的区别：
    - LERM 已线性化（r_d ≈ r_c），A_LERM 不依赖绝对位置，只需标量 ν
    - 系统演化维度为 6D（相对状态），ν 作为辅助时间标记单独更新
    - 对于椭圆轨道（e≠0），A_LERM 随 ν 时变，P 矩阵随轨道周期变化

    Parameters
    ----------
    dynamics     : LERMDynamics         LERM 动力学模型
    controller   : SDREGameController   SDRE 控制器
    x0           : (6,) ndarray         初始相对状态 [x,y,z,vx,vy,vz] (km, km/s)
    nu0          : float                初始真近点角 (rad)
    t_end        : float                仿真终止时间 (s)
    dt           : float                ZOH 控制更新周期 (s)，默认 20.0
    log_interval : int                  每隔多少控制步输出一次日志，默认 50
    are_interval : float | None         ARE 重新求解的时间间隔 (s)。
                                        None = 每步求解（默认）；正数 = 间隔秒数，
                                        期间复用上一次的 P 矩阵。
    x_e0         : (6,) ndarray | None  逃逸者初始 LVLH 6D 状态 (km, km/s)，None = 零（与主星重合）
    """

    def __init__(
        self,
        dynamics: LERMDynamics,
        controller: SDREGameController,
        x0: np.ndarray,
        nu0: float,
        t_end: float,
        dt: float = 20.0,
        log_interval: int = 50,
        are_interval: float | None = None,
        x_e0: np.ndarray | None = None,
    ):
        self.dynamics = dynamics
        self.controller = controller
        self.x0 = np.asarray(x0, dtype=float)
        self.nu0 = float(nu0)
        self.t_end = t_end
        self.dt = dt
        self.log_interval = log_interval
        self.are_interval = are_interval  # None = 每步求解；正数 = 间隔秒数
        self.x_e0 = np.zeros(6) if x_e0 is None else np.asarray(x_e0, dtype=float)

    def run(self) -> LERMSDRESimResult:
        """执行 ZOH 闭环仿真循环。

        每步流程：
            ① 由当前 ν 计算 ν̇, ν̈（开普勒方程）
            ② 重建 A_LERM(ν̇, ν̈)
            ③ 调用 SDRE 控制器求解 ARE → u_p, u_e
            ④ RK45 积分 6D ODE（A_LERM 与控制量在步内冻结）
            ⑤ Euler 更新 ν：ν += ν̇ · Δt

        Returns
        -------
        LERMSDRESimResult
        """
        t = 0.0
        state = self.x0.copy()    # 6D x_rel
        X_e   = self.x_e0.copy() # 6D 逃逸者绝对 LVLH 状态
        nu = self.nu0

        t_history:     list[float]       = [t]
        state_history: list[np.ndarray]  = [state.copy()]
        nu_history:    list[float]       = [nu]
        X_e_hist:      list[np.ndarray]  = [X_e[:3].copy()]
        u_p_history:   list[np.ndarray]  = []
        u_e_history:   list[np.ndarray]  = []
        P_history:     list[np.ndarray]  = []
        step = 0
        last_are_t: float = -np.inf

        # 初始时刻控制量（首次必解 ARE）
        _, nu_dot, nu_ddot = self.dynamics.orbital.get_orbital_params(nu)
        A_lerm = self.dynamics.get_A_matrix(nu_dot, nu_ddot)
        u_p, u_e = self.controller.compute_control(A_lerm, state, t=t, solve_are=True)
        last_are_t = t
        u_p_history.append(u_p.copy())
        u_e_history.append(u_e.copy())
        P0 = self.controller.last_P
        P_history.append(P0.copy() if P0 is not None else np.zeros((6, 6)))

        if self.are_interval is not None:
            print(f"Starting LERM-SDRE simulation... Total duration: {self.t_end:.1f} s  "
                  f"[ARE更新间隔: {self.are_interval:.1f}s]")
        else:
            print(f"Starting LERM-SDRE simulation... Total duration: {self.t_end:.1f} s")

        while t < self.t_end:
            step_size = min(self.dt, self.t_end - t)
            step += 1

            # ① 由当前 ν 计算轨道参数（在该 ZOH 步内冻结）
            _, nu_dot, nu_ddot = self.dynamics.orbital.get_orbital_params(nu)

            # ② 重建时变 A_LERM
            A_lerm = self.dynamics.get_A_matrix(nu_dot, nu_ddot)

            # ③ 判断本步是否需要重新求解 ARE
            need_are = (self.are_interval is None or
                        (t - last_are_t) >= self.are_interval)
            u_p, u_e = self.controller.compute_control(
                A_lerm, state, t=t, solve_are=need_are
            )
            if need_are:
                last_are_t = t

            # ④ 12D 积分：[x_rel(6), X_e(6)]（A_LERM, u_p, u_e 在步内冻结）
            _u_p, _u_e = u_p.copy(), u_e.copy()
            _A = A_lerm.copy()
            _B = self.dynamics.get_B_matrix()
            y0_12 = np.concatenate([state, X_e])
            sol = solve_ivp(
                lambda _t, y: np.concatenate([
                    _A @ y[:6] + _B @ (_u_p - _u_e),  # ẋ_rel
                    _A @ y[6:] + _B @ _u_e,            # ẋ_e
                ]),
                (t, t + step_size),
                y0_12,
                method="RK45",
                rtol=1e-8,
                atol=1e-10,
            )

            if not sol.success:
                raise RuntimeError(
                    f"Numerical integration failed at t={t:.1f}s: {sol.message}"
                )

            state = sol.y[:6, -1].copy()
            X_e   = sol.y[6:, -1].copy()
            t += step_size

            # ⑤ Euler 更新 ν（ν̇ 在步内冻结，与 ZOH 控制假设一致）
            nu += nu_dot * step_size

            t_history.append(t)
            state_history.append(state.copy())
            nu_history.append(nu)
            X_e_hist.append(X_e[:3].copy())
            u_p_history.append(u_p.copy())
            u_e_history.append(u_e.copy())
            P_cur = self.controller.last_P
            P_history.append(P_cur.copy() if P_cur is not None else np.zeros((6, 6)))

            # 周期性日志
            if step % self.log_interval == 0:
                dist = np.linalg.norm(state[:3])
                _log_step(step, t, self.t_end, dist, u_p, P_cur, are_updated=need_are)

        print("Simulation completed.")

        X_e_arr = np.array(X_e_hist).T    # (3, N)
        X_p_arr = np.array(state_history).T[:3, :] + X_e_arr  # (3, N) 位置部分

        return LERMSDRESimResult(
            t=np.array(t_history),
            states=np.array(state_history).T,    # (6, N)
            nu_history=np.array(nu_history),      # (N,)
            u_p_history=np.array(u_p_history).T, # (3, N)
            u_e_history=np.array(u_e_history).T, # (3, N)
            P_history=np.array(P_history),        # (N, 6, 6)
            X_p_history=X_p_arr,                  # (3, N)
            X_e_history=X_e_arr,                  # (3, N)
        )
