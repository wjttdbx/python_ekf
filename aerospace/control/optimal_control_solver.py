"""
直接配点法求解 NERM 2D 时间‑能量最优控制问题 (scipy.optimize 版本)

min_{T, u(·)}  T + gamma * ∫_0^T ||u(t)||^2 dt
s.t. x(0)=x0, r(T)=0, NERM 2D 动力学
"""

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from pathlib import Path

MU = 3.986e5
A_C = 15000.0
E_C = 0.5


def orbital_params(nu):
    r_c = A_C * (1 - E_C**2) / (1 + E_C * np.cos(nu))
    nu_dot = np.sqrt(MU * A_C * (1 - E_C**2)) / r_c**2
    r_c_dot = np.sqrt(MU / (A_C * (1 - E_C**2))) * E_C * np.sin(nu)
    nu_ddot = -2 * r_c_dot * nu_dot / r_c
    return r_c, nu_dot, nu_ddot


def dynamics_5d(t, state, u_func, N_ctrl, T):
    """ODE RHS: state=[dx,dy,dvx,dvy,nu], control 由分段常数插值"""
    dx, dy, dvx, dvy, nu = state

    # 从控制参数化中插值当前控制
    idx = min(int(t / T * N_ctrl), N_ctrl - 1)
    u = u_func(idx)

    r_c, nu_dot, nu_ddot = orbital_params(nu)
    r_p = np.sqrt((r_c + dx)**2 + dy**2)

    grav_x = -MU * (r_c + dx) / r_p**3 + MU / r_c**2
    grav_y = -MU * dy / r_p**3

    ddx = 2 * nu_dot * dvy + nu_ddot * dy + nu_dot**2 * dx + grav_x + u[0]
    ddy = -2 * nu_dot * dvx - nu_ddot * dx + nu_dot**2 * dy + grav_y + u[1]

    return [dvx, dvy, ddx, ddy, nu_dot]


def make_nlp(x0, gamma, N_ctrl):
    """构建 NLP: 决策变量 = [u_x[0], u_y[0], ..., u_x[N-1], u_y[N-1], T]"""

    def objective_and_constraints(vars_vec):
        u_flat = vars_vec[:-1]
        T = vars_vec[-1]
        u_grid = u_flat.reshape(N_ctrl, 2)
        dt_ctrl = T / N_ctrl
        u_max = 0.01

        # 推力限制惩罚
        u_norm = np.linalg.norm(u_grid, axis=1)
        penalty = 0
        for i in range(N_ctrl):
            if u_norm[i] > u_max:
                penalty += 1e4 * (u_norm[i] - u_max)**2
        if T < 100:
            penalty += 1e6 * (100 - T)**2

        # 仿真
        def u_func(idx):
            return u_grid[idx]

        sol = solve_ivp(
            dynamics_5d, (0, T), x0,
            args=(u_func, N_ctrl, T),
            method="RK45", rtol=1e-8, atol=1e-10,
            max_step=dt_ctrl / 2,
        )

        x_final = sol.y[:, -1]
        pos_err = np.sqrt(x_final[0]**2 + x_final[1]**2)

        # 目标: T + gamma * Σ ||u_k||² * dt
        J = T + gamma * np.sum(u_norm**2) * dt_ctrl + penalty + 1e3 * pos_err

        return J

    # 初始猜测
    r0 = np.sqrt(x0[0]**2 + x0[1]**2)
    T_guess = max(r0 * 50, 5000)
    u0 = np.zeros(N_ctrl * 2)
    x_init = np.concatenate([u0, [T_guess]])

    # 边界
    bounds = [(-0.01, 0.01)] * (N_ctrl * 2) + [(100, 100000)]

    return objective_and_constraints, x_init, bounds


def solve_optimal_control(x0, gamma=1e7, N_ctrl=30, max_iter=2000):
    """求解最优控制问题

    Parameters
    ----------
    x0 : (5,) ndarray
    gamma : float  时间-能量权衡系数
    N_ctrl : int   控制分段数
    max_iter : int

    Returns
    -------
    dict with t_grid, x_traj, u_traj, T_opt, converged
    """
    obj_fn, x_init, bounds = make_nlp(x0, gamma, N_ctrl)

    result = minimize(
        obj_fn, x_init,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter, "ftol": 1e-10, "gtol": 1e-8, "disp": True},
    )

    converged = result.success
    if not converged:
        print(f"  Warning: optimizer returned {result.message}")

    u_opt = result.x[:-1].reshape(N_ctrl, 2)
    T_opt = result.x[-1]
    dt_ctrl = T_opt / N_ctrl

    # 细化仿真获取轨迹
    def u_fine(t):
        idx = min(int(t / T_opt * N_ctrl), N_ctrl - 1)
        return u_opt[idx]

    def dyn_5d(t, y):
        dx, dy, dvx, dvy, nu = y
        u = u_fine(t)
        r_c, nu_dot, nu_ddot = orbital_params(nu)
        r_p = np.sqrt((r_c + dx)**2 + dy**2)
        grav_x = -MU * (r_c + dx) / r_p**3 + MU / r_c**2
        grav_y = -MU * dy / r_p**3
        ddx = 2 * nu_dot * dvy + nu_ddot * dy + nu_dot**2 * dx + grav_x + u[0]
        ddy = -2 * nu_dot * dvx - nu_ddot * dx + nu_dot**2 * dy + grav_y + u[1]
        return [dvx, dvy, ddx, ddy, nu_dot]

    sol = solve_ivp(dyn_5d, (0, T_opt), x0, method="RK45", rtol=1e-8, atol=1e-10)

    # 提取控制时间序列
    t_fine = sol.t
    x_fine = sol.y
    u_traj = np.column_stack([u_fine(t) for t in t_fine])

    return {
        "t_grid": t_fine,
        "x_traj": x_fine,
        "u_traj": u_traj,
        "T_opt": T_opt,
        "converged": converged,
        "gamma": gamma,
        "result": result,
    }


if __name__ == "__main__":
    X_p0 = np.array([100.0, 500.0, 0.01, 0.01])
    X_e0 = np.array([0.0, 0.0, 0.0, 0.0])
    nu0 = 0.0
    x_rel = X_p0 - X_e0
    x0 = np.array([x_rel[0], x_rel[1], x_rel[2], x_rel[3], nu0])

    for gamma in [1e5, 3e5, 1e6, 3e6, 1e7]:
        print(f"\n{'='*60}")
        print(f"gamma = {gamma:.1e}")
        print(f"{'='*60}")
        r = solve_optimal_control(x0, gamma=gamma, N_ctrl=40, max_iter=500)
        print(f"\n  converged={r['converged']}, T={r['T_opt']:.1f}s={r['T_opt']/3600:.2f}h")
        if r["converged"]:
            E = float(np.trapezoid(np.sum(r["u_traj"]**2, axis=0), r["t_grid"]))
            peak_u = float(np.max(np.linalg.norm(r["u_traj"], axis=0)))
            print(f"  E={E:.4e}, peak_u={peak_u:.4e}")
            print(f"  terminal pos: [{r['x_traj'][0,-1]:.2e}, {r['x_traj'][1,-1]:.2e}]")

            out_dir = Path("outputs/optimal_control")
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez(out_dir / f"sol_gamma_{gamma:.0e}.npz",
                     t=r["t_grid"], x=r["x_traj"], u=r["u_traj"],
                     T=r["T_opt"], gamma=gamma, x0=x0)
