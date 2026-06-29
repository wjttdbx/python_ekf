"""
CasADi + IPOPT 直接配点法求解 NERM 3D 时间-能量最优控制问题

    min_{T, u(·)}  T + gamma * ∫_0^T ||u(t)||² dt
    s.t. x(0)=x0, ||r(T)|| <= tol, NERM 3D 相对运动力学

使用归一化时间 τ = t/T ∈ [0,1]，梯形配点法。
输出: (t_k, x_k, u_k, λ_k) — 最优轨迹 + 协态 (Pontryagin 乘子)

对比 2D 版: 状态从 5D→7D (增加 dz, dvz)，控制从 2D→3D (增加 uz)
"""

import numpy as np
import casadi as ca
from pathlib import Path

MU = 3.986e5
A_C = 15000.0
E_C = 0.5


def _orbital_params(nu):
    """Keplerian 参数 (CasADi 符号表达式)"""
    r_c = A_C * (1 - E_C**2) / (1 + E_C * ca.cos(nu))
    nu_dot = ca.sqrt(MU * A_C * (1 - E_C**2)) / r_c**2
    r_c_dot = ca.sqrt(MU / (A_C * (1 - E_C**2))) * E_C * ca.sin(nu)
    nu_ddot = -2 * r_c_dot * nu_dot / r_c
    return r_c, nu_dot, nu_ddot


def _dynamics_rhs(x, u):
    """NERM 3D 相对运动 RHS (CasADi 符号)

    x = [dx, dy, dz, dvx, dvy, dvz, nu]  (7D)
    u = [ux, uy, uz]                       (3D)
    """
    dx, dy, dz, dvx, dvy, dvz, nu = (
        x[0], x[1], x[2], x[3], x[4], x[5], x[6],
    )

    r_c, nu_dot, nu_ddot = _orbital_params(nu)
    r_p = ca.sqrt((r_c + dx)**2 + dy**2 + dz**2)

    grav_x = -MU * (r_c + dx) / r_p**3 + MU / r_c**2
    grav_y = -MU * dy / r_p**3
    grav_z = -MU * dz / r_p**3

    ddx = (2 * nu_dot * dvy + nu_ddot * dy + nu_dot**2 * dx
           + grav_x + u[0])
    ddy = (-2 * nu_dot * dvx - nu_ddot * dx + nu_dot**2 * dy
           + grav_y + u[1])
    ddz = grav_z + u[2]

    return ca.vertcat(dvx, dvy, dvz, ddx, ddy, ddz, nu_dot)


def solve_optimal_control(
    x0: np.ndarray,
    gamma: float = 1e7,
    N: int = 40,
    u_max: float = 0.01,
    T_guess: float = None,
    opts: dict = None,
) -> dict:
    """CasADi + IPOPT 梯形直接配点法求解 3D 最优控制。

    使用 τ = t/T 归一化时间，消除自由终端时间带来的非线性。

    Parameters
    ----------
    x0 : (7,)     [dx, dy, dz, dvx, dvy, dvz, nu]
    gamma : float 时间-能量权衡系数
    N : int       配点区间数
    u_max : float 推力上限 (km/s²)
    T_guess : float | None
    opts : dict   额外 IPOPT 选项

    Returns
    -------
    dict: t_grid(N+1,), x_traj(7,N+1), u_traj(3,N+1), lambda_traj(6,N+1),
          T_opt, converged, gamma, stats
    """
    opti = ca.Opti()

    # ── 决策变量: 归一化时间网格 ──
    X = opti.variable(7, N + 1)
    U = opti.variable(3, N)
    T_var = opti.variable()
    dtau = 1.0 / N

    # ── 初始猜测 ──
    r0 = float(np.sqrt(x0[0]**2 + x0[1]**2 + x0[2]**2))
    if T_guess is None:
        T_guess = max(r0 * 80, 10000.0)

    opti.set_initial(T_var, T_guess)
    for i in range(7):
        yf = 0.0 if i < 6 else (x0[6] + T_guess * 1e-4)
        opti.set_initial(X[i, :], np.linspace(x0[i], yf, N + 1))
    opti.set_initial(U, np.zeros((3, N)))

    # ── 目标: T + gamma * T * Σ ||u_k||² * dτ ──
    J = T_var
    for k in range(N):
        J += gamma * T_var * ca.sumsqr(U[:, k]) * dtau
    opti.minimize(J)

    # ── 梯形配点约束 ──
    for k in range(N):
        xk = X[:, k]
        xk1 = X[:, k + 1]
        uk = U[:, k]
        fk = _dynamics_rhs(xk, uk)
        fk1 = _dynamics_rhs(xk1, uk)
        dt_half = 0.5 * T_var * dtau
        opti.subject_to(xk1 == xk + dt_half * (fk + fk1))

    # ── 边界约束 ──
    opti.subject_to(X[:, 0] == x0)
    opti.subject_to(X[0, -1]**2 + X[1, -1]**2 + X[2, -1]**2 <= 1e-6)

    # ── 变量边界 ──
    opti.subject_to(T_var >= 100)
    opti.subject_to(T_var <= 500000)
    # u_max 约束已移除，允许 CasADi 自由选择推力
    for k in range(N + 1):
        opti.subject_to(opti.bounded(-2000, X[0, k], 2000))
        opti.subject_to(opti.bounded(-2000, X[1, k], 2000))
        opti.subject_to(opti.bounded(-2000, X[2, k], 2000))

    # ── IPOPT 选项 ──
    ipopt_opts = {
        "print_level": 1,
        "tol": 1e-6,
        "acceptable_tol": 1e-4,
        "acceptable_iter": 10,
        "max_iter": 1000,
        "linear_solver": "mumps",
        "hessian_approximation": "limited-memory",
        "nlp_scaling_method": "gradient-based",
        "obj_scaling_factor": -1,
    }
    if opts:
        ipopt_opts.update(opts)
    opti.solver("ipopt", {}, ipopt_opts)

    # ── 求解 ──
    try:
        sol = opti.solve()
    except RuntimeError as e:
        print(f"  IPOPT failed: {e}")
        try:
            sol = opti.debug
        except Exception:
            return dict(converged=False, message=str(e))

    # ── 提取 ──
    X_opt = sol.value(X)
    U_opt = sol.value(U)
    T_opt = float(sol.value(T_var))

    t_grid = np.linspace(0, T_opt, N + 1)
    U_traj = np.column_stack([
        U_opt[:, k] if k < N else U_opt[:, -1] for k in range(N + 1)
    ])

    # ── 协态恢复 (PMP: λ_vel = -2γ u*) ──
    lambda_traj = np.zeros((6, N + 1))
    for k in range(N + 1):
        idx = min(k, N - 1)
        lambda_traj[3:6, k] = -2.0 * gamma * U_opt[:, idx]

    # 后向积分位置协态
    def _costate_jac(lam, x_np):
        dx, dy, dz, dvx, dvy, dvz, nu = x_np
        r_c = A_C * (1 - E_C**2) / (1 + E_C * np.cos(nu))
        nu_dot = np.sqrt(MU * A_C * (1 - E_C**2)) / r_c**2
        r_c_dot = np.sqrt(MU / (A_C * (1 - E_C**2))) * E_C * np.sin(nu)
        nu_ddot = -2 * r_c_dot * nu_dot / r_c
        r_p = np.sqrt((r_c + dx)**2 + dy**2 + dz**2)
        r3, r5 = r_p**3, r_p**5
        dgx_dx = -MU / r3 + 3 * MU * (r_c + dx)**2 / r5
        dgx_dy = 3 * MU * (r_c + dx) * dy / r5
        dgx_dz = 3 * MU * (r_c + dx) * dz / r5
        dgy_dy = -MU / r3 + 3 * MU * dy**2 / r5
        dgy_dz = 3 * MU * dy * dz / r5
        dgz_dz = -MU / r3 + 3 * MU * dz**2 / r5
        dfdx = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [nu_dot**2 + dgx_dx, nu_ddot + dgx_dy, dgx_dz, 0, 2 * nu_dot, 0],
            [-nu_ddot + dgx_dy, nu_dot**2 + dgy_dy, dgy_dz, -2 * nu_dot, 0, 0],
            [dgx_dz, dgy_dz, dgz_dz, 0, 0, 0],
        ])
        return -dfdx.T @ lam

    lam = np.zeros(6)
    lam[3:6] = lambda_traj[3:6, N]
    for k in range(N, 0, -1):
        dt_back = t_grid[k] - t_grid[k - 1]
        lam = lam - _costate_jac(lam, X_opt[:, k]) * dt_back
        lambda_traj[:, k - 1] = lam

    stats = sol.stats()
    converged = stats.get("success", False)

    term_err = float(np.sqrt(
        X_opt[0, -1]**2 + X_opt[1, -1]**2 + X_opt[2, -1]**2
    ))
    print(f"  converged={converged}, T={T_opt:.0f}s={T_opt/3600:.2f}h, "
          f"term_err={term_err:.2e}")

    if converged:
        E_ctrl = float(np.trapezoid(np.sum(U_traj**2, axis=0), t_grid))
        peak_u = float(np.max(np.linalg.norm(U_traj, axis=0)))
        print(f"  E_ctrl={E_ctrl:.4e}, peak_u={peak_u:.4e}")
        print(f"  iter={stats.get('iter_count', '?')}")

    return dict(
        t_grid=t_grid, x_traj=X_opt, u_traj=U_traj, lambda_traj=lambda_traj,
        T_opt=T_opt, converged=converged, gamma=gamma, stats=stats,
    )


if __name__ == "__main__":
    # 初始条件: 3D 相对状态 (LVLH坐标系)
    X_p0 = np.array([100.0, 500.0, 50.0, 0.01, 0.01, 0.005])
    X_e0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    nu0 = 0.0
    x_rel = X_p0 - X_e0
    x0 = np.array([x_rel[0], x_rel[1], x_rel[2],
                   x_rel[3], x_rel[4], x_rel[5], nu0])

    out_dir = Path("outputs/optimal_control_3d")
    out_dir.mkdir(parents=True, exist_ok=True)

    for gamma in [1e5, 3e5, 1e6, 3e6, 1e7]:
        print(f"\n{'='*60}")
        print(f"gamma = {gamma:.1e}")
        print(f"{'='*60}")
        r = solve_optimal_control(x0, gamma=gamma, N=40)
        if r["converged"]:
            E = float(np.trapezoid(np.sum(r["u_traj"]**2, axis=0), r["t_grid"]))
            print(f"  E={E:.4e}")
            print(f"  terminal: [{r['x_traj'][0,-1]:.2e}, "
                  f"{r['x_traj'][1,-1]:.2e}, {r['x_traj'][2,-1]:.2e}]")
            np.savez(
                out_dir / f"sol_gamma_{gamma:.0e}.npz",
                t=r["t_grid"], x=r["x_traj"], u=r["u_traj"],
                lam=r["lambda_traj"], T=r["T_opt"], gamma=gamma, x0=x0,
            )
