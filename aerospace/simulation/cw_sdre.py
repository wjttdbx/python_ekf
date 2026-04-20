"""
CW + SDRE 追逃博弈闭环仿真引擎

将 Clohessy-Wiltshire 线性模型嵌入 SDRE 框架运行：
- 状态向量为 6 维相对状态 x_rel = [x, y, z, vx, vy, vz]
- 系统矩阵 A 为 CW 常数矩阵，即 A_SDC(x) ≡ A_cw（不随状态变化）
- SDRE 控制器每步仍调用 ARE 求解器（P 应收敛到同一常数矩阵，可与 LQDG 离线解对比）
- 动力学积分：dx/dt = A_cw x + B_cw (u_p - u_e)，使用 RK45
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from aerospace.dynamics.cw import CWDynamics
from aerospace.control.sdre import SDREGameController


def _log_step(step: int, t: float, T_total: float,
              dist: float, u_p: np.ndarray, P_cur: np.ndarray | None,
              dist_unit: str = "m", are_updated: bool = True) -> None:
    """输出统一格式的仿真进度日志。

    格式：[Step N]  t=XXX.Xs  进度=XX.X%  相对距离=X.XXXm  |u_p|=X.XXe-XX  λ(P)=[...]  [ARE↑/P缓存]
    """
    if P_cur is not None:
        eigs = np.linalg.eigvalsh(P_cur)
        lam_str = f"λ(P)=[{eigs.min():.3e}, {eigs.max():.3e}]"
    else:
        lam_str = "λ(P)=[N/A, N/A]"
    are_tag = "[ARE↑]" if are_updated else "[P缓存]"
    print(
        f"[Step {step:4d}]  t={t:8.1f}s  进度={t / T_total * 100:5.1f}%  "
        f"相对距离={dist:.3f}{dist_unit}  "
        f"|u_p|={np.linalg.norm(u_p):.3e}  "
        f"{lam_str}  {are_tag}"
    )


@dataclass
class CWSDRESimResult:
    """CW-SDRE 仿真结果数据容器。

    Attributes
    ----------
    t          : (N,)      时间序列 (s)
    states     : (6, N)    相对状态历史 x_rel = X_p - X_e  [x,y,z,vx,vy,vz]
    u_p_history: (3, N)    追踪者推力加速度历史
    u_e_history: (3, N)    逃逸者推力加速度历史
    P_history  : (N, 6, 6) 每步 ARE 解出的 P 矩阵历史
    X_p_history: (3, N)    追踪者在 LVLH 系中的绝对位置历史
    X_e_history: (3, N)    逃逸者在 LVLH 系中的绝对位置历史
    """

    t: np.ndarray
    states: np.ndarray
    u_p_history: np.ndarray
    u_e_history: np.ndarray
    P_history: np.ndarray
    X_p_history: np.ndarray
    X_e_history: np.ndarray


class CWSDRESimulation:
    """CW 动力学 + SDRE 控制器的零和微分博弈闭环仿真。

    CW 的 A 矩阵为常数，因此 SDRE 每步求解的 ARE 应给出同一个 P，
    等价于 LQDG 的离线全局解（可用于验证两种方法的一致性）。

    Parameters
    ----------
    dynamics     : CWDynamics          CW 动力学模型（提供 A, B 矩阵）
    controller   : SDREGameController  SDRE 控制器实例
    x0           : (6,) ndarray        初始相对状态 [x, y, z, vx, vy, vz]
    t_end        : float               仿真终止时间 (s)
    dt           : float               控制更新周期 (s)，默认 5.0
    are_interval : float | None        ARE 重新求解的时间间隔 (s)。
                                       None = 每步求解（默认）；正数 = 间隔秒数，
                                       期间复用上一次的 P 矩阵。
    x_e0         : (6,) ndarray | None 逃逸者初始 LVLH 6D 状态，None = 零（与主星重合）
    """

    def __init__(
        self,
        dynamics: CWDynamics,
        controller: SDREGameController,
        x0: np.ndarray,
        t_end: float,
        dt: float = 5.0,
        log_interval: int = 50,
        are_interval: float | None = None,
        x_e0: np.ndarray | None = None,
    ):
        self.dynamics = dynamics
        self.controller = controller
        self.x0 = np.asarray(x0, dtype=float)
        self.t_end = t_end
        self.dt = dt
        self.log_interval = log_interval
        self.are_interval = are_interval  # None = 每步求解；正数 = 间隔秒数
        self.x_e0 = np.zeros(6) if x_e0 is None else np.asarray(x_e0, dtype=float)

        # CW 常数矩阵，A_SDC 在整个仿真过程中不变
        self.A_cw, self.B_cw = dynamics.get_state_space()

    def run(self) -> CWSDRESimResult:
        """执行 ZOH 闭环仿真循环。

        Returns
        -------
        CWSDRESimResult
            包含时间、状态、推力历史和 P 矩阵历史的结果
        """
        t = 0.0
        state = self.x0.copy()   # 6D x_rel
        X_e   = self.x_e0.copy() # 6D 逃逸者绝对 LVLH 状态

        t_history:     list[float]       = [t]
        state_history: list[np.ndarray]  = [state.copy()]
        X_e_hist:      list[np.ndarray]  = [X_e[:3].copy()]
        u_p_history:   list[np.ndarray]  = []
        u_e_history:   list[np.ndarray]  = []
        P_history:     list[np.ndarray]  = []
        step = 0
        last_are_t: float = -np.inf

        # 初始时刻控制量（首次必解 ARE）
        u_p, u_e = self.controller.compute_control(self.A_cw, state, t=t, solve_are=True)
        last_are_t = t
        u_p_history.append(u_p.copy())
        u_e_history.append(u_e.copy())
        P0 = self.controller.last_P
        P_history.append(P0.copy() if P0 is not None else np.zeros((6, 6)))

        if self.are_interval is not None:
            print(f"Starting CW-SDRE simulation... Total duration: {self.t_end:.1f} s  "
                  f"[ARE更新间隔: {self.are_interval:.1f}s]")
        else:
            print(f"Starting CW-SDRE simulation... Total duration: {self.t_end:.1f} s")

        while t < self.t_end:
            step_size = min(self.dt, self.t_end - t)
            step += 1

            need_are = (self.are_interval is None or
                        (t - last_are_t) >= self.are_interval)
            u_p, u_e = self.controller.compute_control(
                self.A_cw, state, t=t, solve_are=need_are
            )
            if need_are:
                last_are_t = t

            # 12D 积分：[x_rel(6), X_e(6)]
            # ẋ_rel = A x_rel + B (u_p - u_e)
            # ẋ_e   = A X_e   + B u_e
            _u_p, _u_e = u_p.copy(), u_e.copy()
            y0_12 = np.concatenate([state, X_e])
            sol = solve_ivp(
                lambda _t, y: np.concatenate([
                    self.A_cw @ y[:6] + self.B_cw @ (_u_p - _u_e),
                    self.A_cw @ y[6:] + self.B_cw @ _u_e,
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

            t_history.append(t)
            state_history.append(state.copy())
            X_e_hist.append(X_e[:3].copy())
            u_p_history.append(u_p.copy())
            u_e_history.append(u_e.copy())
            P_cur = self.controller.last_P
            P_history.append(P_cur.copy() if P_cur is not None else np.zeros((6, 6)))

            if step % self.log_interval == 0:
                dist = np.linalg.norm(state[:3])
                _log_step(step, t, self.t_end, dist, u_p, P_cur,
                          dist_unit="m", are_updated=need_are)

        print("Simulation completed.")

        X_e_arr = np.array(X_e_hist).T   # (3, N)
        X_p_arr = np.array(state_history).T[:3, :] + X_e_arr  # (3, N) 位置部分

        return CWSDRESimResult(
            t=np.array(t_history),
            states=np.array(state_history).T,    # (6, N)
            u_p_history=np.array(u_p_history).T, # (3, N)
            u_e_history=np.array(u_e_history).T, # (3, N)
            P_history=np.array(P_history),        # (N, 6, 6)
            X_p_history=X_p_arr,                  # (3, N)
            X_e_history=X_e_arr,                  # (3, N)
        )
