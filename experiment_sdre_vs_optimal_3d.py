"""
3D SDRE vs CasADi 公平对比: 相同代价函数

    J = ∫ (xᵀQ x + uᵀ R u) dt    (纯追捕, γ → ∞)

Q/R 使用 SDRE 默认参数. CasADi 固定 T = T_sdre, 比较真最小 J* vs SDRE 的 J.
"""

import numpy as np
import casadi as ca
from pathlib import Path
from scipy.integrate import solve_ivp

MU = 3.986e5; A_C = 15000.0; E_C = 0.5
OUT_DIR = Path("outputs/sdre_vs_optimal_3d")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _orbital_params(nu):
    r_c = A_C * (1 - E_C ** 2) / (1 + E_C * ca.cos(nu))
    nu_dot = ca.sqrt(MU * A_C * (1 - E_C ** 2)) / r_c ** 2
    r_c_dot = ca.sqrt(MU / (A_C * (1 - E_C ** 2))) * E_C * ca.sin(nu)
    nu_ddot = -2 * r_c_dot * nu_dot / r_c
    return r_c, nu_dot, nu_ddot


def _dynamics_rhs(x, u):
    dx, dy, dz, dvx, dvy, dvz, nu = (
        x[0], x[1], x[2], x[3], x[4], x[5], x[6],
    )
    r_c, nu_dot, nu_ddot = _orbital_params(nu)
    r_p = ca.sqrt((r_c + dx) ** 2 + dy ** 2 + dz ** 2)
    grav_x = -MU * (r_c + dx) / r_p ** 3 + MU / r_c ** 2
    grav_y = -MU * dy / r_p ** 3
    grav_z = -MU * dz / r_p ** 3
    ddx = (2 * nu_dot * dvy + nu_ddot * dy + nu_dot ** 2 * dx + grav_x + u[0])
    ddy = (-2 * nu_dot * dvx - nu_ddot * dx + nu_dot ** 2 * dy + grav_y + u[1])
    ddz = grav_z + u[2]
    return ca.vertcat(dvx, dvy, dvz, ddx, ddy, ddz, nu_dot)


def solve_optimal_with_sdre_cost(
    x0: np.ndarray, T_fixed: float, Q_diag: np.ndarray, R_diag: np.ndarray,
    N: int = 60, u_max: float = 0.01, warm_x: np.ndarray = None, warm_u: np.ndarray = None,
) -> dict:
    """CasADi: min ∫(xᵀQ x + uᵀ R u) dt, fixed T=T_fixed"""
    opti = ca.Opti()
    X = opti.variable(7, N + 1)
    U = opti.variable(3, N)
    dtau = 1.0 / N

    # 初始猜测: SDRE 轨迹 warm-start, 否则线性插值
    if warm_x is not None and warm_u is not None:
        # 将 SDRE 轨迹映射到 N+1 网格
        t_warm = np.linspace(0, T_fixed, warm_x.shape[1])
        t_new = np.linspace(0, T_fixed, N + 1)
        X_guess = np.zeros((7, N + 1))
        for i in range(7):
            X_guess[i, :] = np.interp(t_new, t_warm, warm_x[i])
        U_guess = np.zeros((3, N))
        t_unew = np.linspace(0, T_fixed, N)
        for i in range(3):
            U_guess[i, :] = np.interp(t_unew, t_warm, warm_u[i])
    else:
        X_guess = np.zeros((7, N + 1))
        for i in range(7):
            yf = 0.0 if i < 6 else (x0[6] + T_fixed * 1e-4)
            X_guess[i, :] = np.linspace(x0[i], yf, N + 1)
        U_guess = np.zeros((3, N))
    opti.set_initial(X, X_guess)
    opti.set_initial(U, U_guess)

    Q_mat = np.diag(Q_diag)
    R_mat = np.diag(R_diag)
    J = 0
    for k in range(N):
        xk = X[:6, k]; uk = U[:, k]
        J += (xk.T @ Q_mat @ xk + uk.T @ R_mat @ uk) * T_fixed * dtau
    opti.minimize(J)

    for k in range(N):
        xk = X[:, k]; xk1 = X[:, k + 1]; uk = U[:, k]
        fk = _dynamics_rhs(xk, uk); fk1 = _dynamics_rhs(xk1, uk)
        dt_half = 0.5 * T_fixed * dtau
        opti.subject_to(xk1 == xk + dt_half * (fk + fk1))

    opti.subject_to(X[:, 0] == x0)
    opti.subject_to(X[0, -1] ** 2 + X[1, -1] ** 2 + X[2, -1] ** 2 <= 1e-4)  # relaxed
    # u_max 约束已移除，允许 CasADi 自由选择推力
    for k in range(N + 1):
        opti.subject_to(opti.bounded(-2000, X[0, k], 2000))
        opti.subject_to(opti.bounded(-2000, X[1, k], 2000))
        opti.subject_to(opti.bounded(-2000, X[2, k], 2000))

    opti.solver("ipopt", {}, {
        "print_level": 0, "tol": 1e-6, "max_iter": 5000,
        "linear_solver": "mumps", "hessian_approximation": "limited-memory",
        "acceptable_tol": 1e-3, "acceptable_iter": 15,
    })
    try:
        sol = opti.solve()
        converged = True
    except RuntimeError as e:
        print(f"  IPOPT: {str(e)[:100]}")
        try:
            sol = opti.debug
            converged = False
        except Exception:
            return dict(converged=False, message=str(e))

    X_opt = sol.value(X); U_opt = sol.value(U)
    t_grid = np.linspace(0, T_fixed, N + 1)
    U_traj = np.column_stack([U_opt[:, k] if k < N else U_opt[:, -1] for k in range(N + 1)])
    term_err = float(np.sqrt(X_opt[0, -1]**2 + X_opt[1, -1]**2 + X_opt[2, -1]**2))
    stats = sol.stats()
    print(f"  iter={stats.get('iter_count','?')} term_err={term_err:.2e} "
          f"success={stats.get('success',False)}")
    return dict(converged=converged, t=t_grid, x=X_opt, u=U_traj, T=T_fixed,
                term_err=term_err)


def run_sdre_3d(x_rel0: np.ndarray, nu0: float, Q: np.ndarray, R: np.ndarray,
                dt: float = 20.0) -> dict:
    """3D SDRE 闭环仿真 (纯追捕, γ → ∞)."""
    from aerospace.dynamics.nerm import OrbitalDynamics
    from scipy.linalg import solve_continuous_are

    orb = OrbitalDynamics(mu=MU, a_c=A_C, e_c=E_C)
    T_end = orb.T_orbit * 5.0
    N_steps = int(T_end / dt)
    X_p = np.array([*x_rel0[:3], *x_rel0[3:6]])
    X_e = np.zeros(6)
    nu = nu0; t_now = 0.0
    traj_rel, u_hist = [], []
    traj_rel.append(x_rel0[:6].copy())
    u_hist.append(np.zeros(3))
    B = np.zeros((6, 3)); B[3:, :] = np.eye(3)

    for k in range(N_steps):
        x_rel = X_p - X_e
        r_c, nu_dot, nu_ddot = orb.get_orbital_params(nu)
        A = orb.get_SDC_matrix(X_p, X_e, r_c, nu_dot, nu_ddot)
        try:
            P = solve_continuous_are(A, B, Q, R)
        except Exception:
            P = np.zeros((6, 6))
        u_p = -np.linalg.inv(R) @ B.T @ P @ x_rel
        state = np.concatenate([X_p, X_e, [nu]])
        sol = solve_ivp(orb.dynamics_13d, [t_now, t_now + dt], state,
                        args=(u_p, np.zeros(3)), method="RK45", rtol=1e-8, atol=1e-10)
        state = sol.y[:, -1]
        X_p = state[0:6]; X_e = state[6:12]; nu = state[12]; t_now += dt
        traj_rel.append((X_p - X_e).copy()); u_hist.append(u_p.copy())
        if np.linalg.norm(X_p[:3] - X_e[:3]) < 0.1:
            break

    t_arr = np.arange(len(traj_rel)) * dt
    x_arr = np.array(traj_rel).T; u_arr = np.array(u_hist).T
    d_arr = np.linalg.norm(x_arr[:3], axis=0)
    cap_idx = np.argmax(d_arr < 0.1) if np.any(d_arr < 0.1) else len(d_arr) - 1
    return dict(t=t_arr, x=x_arr, u=u_arr, T_cap=t_arr[cap_idx], cap_idx=cap_idx)


def compute_cost(t, x, u, Q, R):
    n = len(t); J = 0.0
    for i in range(1, n):
        J += (x[:6, i] @ Q @ x[:6, i] + u[:, i] @ R @ u[:, i]) * (t[i] - t[i-1])
    return float(J)


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    X_p0 = np.array([100.0, 500.0, 50.0, 0.01, 0.01, 0.005])
    X_e0 = np.zeros(6); nu0 = 0.0
    x_rel0 = X_p0 - X_e0
    x0 = np.array([*x_rel0, nu0])

    Q = np.diag([1.0, 1.0, 1.0, 10.0, 10.0, 10.0])
    R = np.eye(3) * 1e13

    # ── SDRE ──
    print("Running SDRE 3D ...")
    sdre = run_sdre_3d(x_rel0, nu0, Q, R, dt=20.0)
    T_sdre = sdre["T_cap"]
    print(f"  capture: T={T_sdre:.0f}s ({T_sdre/3600:.2f}h)")

    # ── CasADi (same Q,R, same T, SDRE warm-start) ──
    print(f"\nRunning CasADi (T={T_sdre:.0f}s, same Q,R, SDRE warm-start) ...")
    # 构建 7D warm-start: [rel_pos(3), rel_vel(3), nu]
    nu_hist = np.linspace(nu0, nu0 + T_sdre * 1e-4, sdre["x"].shape[1])
    warm_x = np.vstack([sdre["x"][:6], nu_hist])
    opt = solve_optimal_with_sdre_cost(
        x0, T_fixed=T_sdre, Q_diag=np.diag(Q), R_diag=np.diag(R), N=60, u_max=0.01,
        warm_x=warm_x, warm_u=sdre["u"],
    )
    print(f"  converged={opt['converged']}")

    # ── 代价对比 ──
    J_s = compute_cost(sdre["t"], sdre["x"], sdre["u"], Q, R)
    has_opt = opt["converged"] or ('u' in opt)
    if has_opt:
        J_o = compute_cost(opt["t"], opt["x"], opt["u"], Q, R)
        u_opt_mean = np.mean(np.linalg.norm(opt['u'], axis=0))
        u_opt_peak = np.max(np.linalg.norm(opt['u'], axis=0))
    else:
        J_o = float("nan")

    print(f"\n{'='*60}")
    print("Cost: J = ∫(xᵀQx + uᵀRu) dt")
    print(f"{'='*60}")
    print(f"  SDRE:        J = {J_s:.6e}")
    if has_opt:
        print(f"  CasADi:      J = {J_o:.6e}")
        print(f"  SDRE extra:  {(J_s/J_o - 1)*100:.1f}%")
    else:
        print(f"  CasADi:      FAILED")
    u_s_mean = np.mean(np.linalg.norm(sdre['u'], axis=0))
    u_s_peak = np.max(np.linalg.norm(sdre['u'], axis=0))
    print(f"  ||u_sdre||   mean={u_s_mean:.2e}  peak={u_s_peak:.2e}")
    if has_opt:
        print(f"  ||u_opt||    mean={u_opt_mean:.2e}  peak={u_opt_peak:.2e}")

    # ── 绘图 ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"3D SDRE vs CasADi (Q=diag(1,1,1,10,10,10), R=1e13·I₃) | T={T_sdre/3600:.1f}h",
                 fontsize=13)

    labels = [("SDRE", sdre["t"], sdre["x"], sdre["u"], "b", "--")]
    if has_opt:
        labels.append(("CasADi", opt["t"], opt["x"], opt["u"], "r", "-"))

    # Distance
    ax = axes[0, 0]
    for name, t, x, u, c, ls in labels:
        d = np.linalg.norm(x[:3], axis=0)
        ax.semilogy(t / 3600, d, color=c, ls=ls, lw=1.5, label=name)
    ax.set_xlabel("Time (h)"); ax.set_ylabel("Distance (km)")
    ax.legend(); ax.grid(alpha=0.4); ax.set_title("Relative Distance")

    # xy
    ax = axes[0, 1]
    for name, t, x, u, c, ls in labels:
        ax.plot(x[0], x[1], color=c, ls=ls, lw=1, label=f"{name} (xy)")
    ax.scatter(x_rel0[0], x_rel0[1], c="g", s=50, zorder=5)
    ax.scatter(0, 0, c="r", s=50, marker="*", zorder=5)
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
    ax.legend(fontsize=8); ax.grid(alpha=0.4); ax.set_aspect("equal"); ax.set_title("xy")

    # xz
    ax = axes[0, 2]
    for name, t, x, u, c, ls in labels:
        ax.plot(x[0], x[2], color=c, ls=ls, lw=1, label=name)
    ax.scatter(x_rel0[0], x_rel0[2], c="g", s=50, zorder=5)
    ax.scatter(0, 0, c="r", s=50, marker="*", zorder=5)
    ax.set_xlabel("x (km)"); ax.set_ylabel("z (km)")
    ax.legend(fontsize=8); ax.grid(alpha=0.4); ax.set_aspect("equal"); ax.set_title("xz")

    # ||u||
    ax = axes[1, 0]
    for name, t, x, u, c, ls in labels:
        ax.plot(t / 3600, np.linalg.norm(u, axis=0) * 1e6, color=c, ls=ls, lw=1.5, label=name)
    ax.set_xlabel("Time (h)"); ax.set_ylabel("||u|| (μm/s²)")
    ax.legend(); ax.grid(alpha=0.4); ax.set_title("Control Magnitude")

    # u_x
    ax = axes[1, 1]
    for name, t, x, u, c, ls in labels:
        ax.plot(t / 3600, u[0] * 1e6, color=c, ls=ls, lw=1, label=f"{name} u_x")
    ax.set_xlabel("Time (h)"); ax.set_ylabel("u_x (μm/s²)")
    ax.legend(fontsize=7); ax.grid(alpha=0.4); ax.set_title("u_x")

    # u_z
    ax = axes[1, 2]
    for name, t, x, u, c, ls in labels:
        ax.plot(t / 3600, u[2] * 1e6, color=c, ls=ls, lw=1, label=f"{name} u_z")
    ax.set_xlabel("Time (h)"); ax.set_ylabel("u_z (μm/s²)")
    ax.legend(fontsize=7); ax.grid(alpha=0.4); ax.set_title("u_z")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "sdre_vs_optimal_3d.png", dpi=150)
    plt.close(fig)
    print(f"\nPlot: {OUT_DIR / 'sdre_vs_optimal_3d.png'}")
