"""
2D NERM+SDRE 追逃博弈闭环仿真引擎

9D 状态 = [X_p(4), X_e(4), nu(1)]，ZOH 控制 + RK45 积分。
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from aerospace.dynamics.nerm_2d import OrbitalDynamics2D
from aerospace.control.sdre_2d import SDREGameController2D


def _log_step(step, t, T_total, dist, u_p, P_cur, are_updated=True):
    if P_cur is not None:
        eigs = np.linalg.eigvalsh(P_cur)
        lam_str = f"λ(P)=[{eigs.min():.3e}, {eigs.max():.3e}]"
    else:
        lam_str = "λ(P)=[N/A, N/A]"
    tag = "[ARE↑]" if are_updated else "[P缓存]"
    print(
        f"[Step {step:4d}]  t={t:8.1f}s  进度={t / T_total * 100:5.1f}%  "
        f"相对距离={dist:.3f}km  |u_p|={np.linalg.norm(u_p):.3e}  {lam_str}  {tag}"
    )


@dataclass
class SDRESimResult2D:
    """2D 仿真结果。

    states     : (9, N)    [X_p(4), X_e(4), nu]
    u_p/e_hist : (2, N)
    P_history  : (N, 4, 4)
    """

    t: np.ndarray
    states: np.ndarray
    u_p_history: np.ndarray
    u_e_history: np.ndarray
    P_history: np.ndarray
    captured: bool = False


class SDRESimulation2D:
    """2D 面内 SDRE 闭环仿真引擎。"""

    def __init__(
        self,
        dynamics: OrbitalDynamics2D,
        controller: SDREGameController2D,
        X_p0: np.ndarray,
        X_e0: np.ndarray,
        nu0: float = 0.0,
        dt: float = 20.0,
        log_interval: int = 50,
        are_interval: float | None = None,
    ):
        self.dynamics = dynamics
        self.controller = controller
        self.dt = dt
        self.log_interval = log_interval
        self.are_interval = are_interval
        self.t_span = (0.0, 10 * dynamics.T_orbit)
        self.capture_dist = 0.1  # 追捕成功距离阈值 (km)

        self.state0 = np.zeros(9)
        self.state0[0:4] = X_p0
        self.state0[4:8] = X_e0
        self.state0[8] = nu0

    def run(self) -> SDRESimResult2D:
        T_total = self.t_span[1]
        t = 0.0
        state = self.state0.copy()

        t_history: list[float] = [t]
        state_history: list[np.ndarray] = [state.copy()]
        u_p_history: list[np.ndarray] = []
        u_e_history: list[np.ndarray] = []
        P_history: list[np.ndarray] = []
        step = 0
        captured = False
        last_are_t: float = -np.inf

        X_p = state[0:4]
        X_e = state[4:8]
        nu = state[8]
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
        P_history.append(P0.copy() if P0 is not None else np.zeros((4, 4)))

        interval_msg = f"  [ARE间隔: {self.are_interval:.1f}s]" if self.are_interval else ""
        print(f"Starting 2D SDRE simulation... Total: {T_total:.1f} s{interval_msg}")

        while t < T_total:
            step_size = min(self.dt, T_total - t)
            step += 1

            X_p = state[0:4]
            X_e = state[4:8]
            nu = state[8]

            r_c, nu_dot, nu_ddot = self.dynamics.get_orbital_params(nu)
            A_SDC = self.dynamics.get_SDC_matrix(X_p, X_e, r_c, nu_dot, nu_ddot)
            if hasattr(self.controller, "set_environment"):
                self.controller.set_environment(r_c, nu_dot, nu_ddot)
            if hasattr(self.controller, "set_positions"):
                self.controller.set_positions(X_p, X_e)

            need_are = self.are_interval is None or (t - last_are_t) >= self.are_interval
            u_p, u_e = self.controller.compute_control(
                A_SDC, X_p - X_e, t=t, solve_are=need_are
            )
            if need_are:
                last_are_t = t

            sol = solve_ivp(
                lambda tt, y: self.dynamics.dynamics_9d(tt, y, u_p, u_e),
                (t, t + step_size),
                state,
                method="RK45",
                rtol=1e-8,
                atol=1e-10,
            )

            if not sol.success:
                raise RuntimeError(f"Integration failed at t={t}: {sol.message}")

            state = sol.y[:, -1].copy()
            t += step_size

            t_history.append(t)
            state_history.append(state.copy())
            u_p_history.append(u_p.copy())
            u_e_history.append(u_e.copy())
            P_cur = self.controller.last_P
            P_history.append(P_cur.copy() if P_cur is not None else np.zeros((4, 4)))

            if step % self.log_interval == 0:
                x_rel_pos = state[0:2] - state[4:6]
                dist = np.linalg.norm(x_rel_pos)
                _log_step(step, t, T_total, dist, u_p, P_cur, are_updated=need_are)

            # 追捕成功终止
            x_rel_pos = state[0:2] - state[4:6]
            if np.linalg.norm(x_rel_pos) < self.capture_dist:
                print(f"Capture achieved at t={t:.1f}s (dist={np.linalg.norm(x_rel_pos)*1000:.1f}m).")
                captured = True
                break

        print("Simulation completed.")

        return SDRESimResult2D(
            t=np.array(t_history),
            states=np.array(state_history).T,
            u_p_history=np.array(u_p_history).T,
            u_e_history=np.array(u_e_history).T,
            P_history=np.array(P_history),
            captured=captured,
        )
