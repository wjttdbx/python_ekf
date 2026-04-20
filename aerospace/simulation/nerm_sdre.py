"""
SDRE 追逃博弈闭环仿真引擎

采用零阶保持器 (ZOH) 控制周期，进行连续时间数值积分仿真。
"""

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

from aerospace.dynamics.nerm import OrbitalDynamics
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
class SDRESimResult:
    """SDRE 仿真结果数据容器。

    Attributes
    ----------
    t          : (N,)      时间序列 (s)
    states     : (13, N)   状态向量历史 [X_p, X_e, nu]
    u_p_history: (3, N)    追踪星实际推力加速度历史
    u_e_history: (3, N)    逃逸星实际推力加速度历史
    P_history  : (N, 6, 6) 每步 ARE 解出的 P 矩阵历史
    """
    t: np.ndarray
    states: np.ndarray
    u_p_history: np.ndarray
    u_e_history: np.ndarray
    P_history: np.ndarray
    captured: bool = False

class SDRESimulation:
    """SDRE 零和微分博弈闭环仿真引擎。

    Parameters
    ----------
    dynamics : OrbitalDynamics
        动力学模型实例
    controller : SDREGameController
        控制器实例
    X_p0 : (6,) ndarray
        追踪星初始状态 [x, y, z, vx, vy, vz]
    X_e0 : (6,) ndarray
        逃逸星初始状态 [x, y, z, vx, vy, vz]
    nu0 : float
        初始真近点角 (rad)
    dt : float
        控制更新周期 (s)，默认 20.0
    """

    def __init__(self, dynamics: OrbitalDynamics, controller: SDREGameController,
                 X_p0: np.ndarray, X_e0: np.ndarray, nu0: float = 0.0, dt: float = 20.0,
                 log_interval: int = 50, are_interval: float | None = None):
        self.dynamics = dynamics
        self.controller = controller

        self.dt = dt
        self.log_interval = log_interval
        self.are_interval = are_interval  # None = 每步求解 ARE；正数 = 间隔秒数
        self.t_span = (0.0, 10 * dynamics.T_orbit)  # 默认仿真 10 个周期
        self.capture_dist = 0.1  # 追捕成功距离阈值 (km)
        
        self.state0 = np.zeros(13)
        self.state0[0:6] = X_p0
        self.state0[6:12] = X_e0
        self.state0[12] = nu0

    def run(self) -> SDRESimResult:
        """执行 ZOH 闭环仿真循环。

        Returns
        -------
        SDRESimResult
            包含时间、状态和推力历史的结果
        """
        # 总时长
        T_total = self.t_span[1]
        
        t = 0.0
        state = self.state0.copy()

        t_history:     list[float]        = [t]
        state_history: list[np.ndarray]   = [state.copy()]
        u_p_history:   list[np.ndarray]   = []
        u_e_history:   list[np.ndarray]   = []
        P_history:     list[np.ndarray]   = []
        step = 0
        captured = False
        last_are_t: float = -np.inf  # 上次 ARE 求解的时刻（-inf 确保首次必解）

        # 为了对应长度，初始时刻需要一个控制输入（首次必解 ARE）
        X_p = state[0:6]
        X_e = state[6:12]
        nu = state[12]
        r_c, nu_dot, nu_ddot = self.dynamics.get_orbital_params(nu)
        A_SDC = self.dynamics.get_SDC_matrix(X_p, X_e, r_c, nu_dot, nu_ddot)
        if hasattr(self.controller, "set_environment"):
            self.controller.set_environment(r_c, nu_dot, nu_ddot)
        if hasattr(self.controller, "set_positions"):
            self.controller.set_positions(X_p, X_e)
        u_p, u_e = self.controller.compute_control(A_SDC, X_p - X_e, t=t, solve_are=True)
        last_are_t = t

        u_p_history.append(u_p.copy())
        u_e_history.append(u_e.copy())
        P0 = self.controller.last_P
        P_history.append(P0.copy() if P0 is not None else np.zeros((6, 6)))

        if self.are_interval is not None:
            print(f"Starting SDRE simulation... Total duration: {T_total:.1f} s  "
                  f"[ARE更新间隔: {self.are_interval:.1f}s]")
        else:
            print(f"Starting SDRE simulation... Total duration: {T_total:.1f} s")

        while t < T_total:
            step_size = min(self.dt, T_total - t)
            step += 1

            X_p = state[0:6]
            X_e = state[6:12]
            nu = state[12]

            r_c, nu_dot, nu_ddot = self.dynamics.get_orbital_params(nu)
            A_SDC = self.dynamics.get_SDC_matrix(X_p, X_e, r_c, nu_dot, nu_ddot)
            if hasattr(self.controller, "set_environment"):
                self.controller.set_environment(r_c, nu_dot, nu_ddot)
            if hasattr(self.controller, "set_positions"):
                self.controller.set_positions(X_p, X_e)

            # 判断本步是否需要重新求解 ARE
            need_are = (self.are_interval is None or
                        (t - last_are_t) >= self.are_interval)
            u_p, u_e = self.controller.compute_control(
                A_SDC, X_p - X_e, t=t, solve_are=need_are
            )
            if need_are:
                last_are_t = t

            # RK45 积分
            sol = solve_ivp(
                lambda t_eval, y: self.dynamics.dynamics_13d(t_eval, y, u_p, u_e),
                (t, t + step_size),
                state,
                method="RK45",
                rtol=1e-8,
                atol=1e-10
            )

            if not sol.success:
                raise RuntimeError(f"Numerical integration failed at t={t}: {sol.message}")

            state = sol.y[:, -1].copy()
            t += step_size

            t_history.append(t)
            state_history.append(state.copy())
            u_p_history.append(u_p.copy())
            u_e_history.append(u_e.copy())
            P_cur = self.controller.last_P
            P_history.append(P_cur.copy() if P_cur is not None else np.zeros((6, 6)))

            # 周期性状态日志
            if step % self.log_interval == 0:
                x_rel_pos = state[0:3] - state[6:9]
                dist = np.linalg.norm(x_rel_pos)
                _log_step(step, t, T_total, dist, u_p, P_cur, are_updated=need_are)

            # 追捕成功终止
            x_rel_pos = state[0:3] - state[6:9]
            if np.linalg.norm(x_rel_pos) < self.capture_dist:
                print(f"Capture achieved at t={t:.1f}s (dist={np.linalg.norm(x_rel_pos)*1000:.1f}m).")
                captured = True
                break

        print("Simulation completed.")

        return SDRESimResult(
            t=np.array(t_history),
            states=np.array(state_history).T,
            u_p_history=np.array(u_p_history).T,
            u_e_history=np.array(u_e_history).T,
            P_history=np.array(P_history),
            captured=captured,
        )
