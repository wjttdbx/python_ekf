"""
微分博弈 TPBVP: 用 Pontryagin 必要条件 + solve_bvp 求真正的最优解

    J = ∫ (x^T Q x + u_p^T R u_p - γ^2 u_e^T R u_e) dt

必要最优条件:
    u_p = -1/2 * R^{-1} B^T λ
    u_e = 1/(2γ^2) * R^{-1} B^T λ
    dx/dt = f(x) - 1/2*(1-γ^{-2}) * B R^{-1} B^T λ
    dλ/dt = -2Qx - (∂f/∂x)^T λ
    边界: x(0)=x0, λ(T)=0

状态: [dx, dy, dvx, dvy, nu] (5D), 协态: [λx,λy,λvx,λvy,λnu] (5D)
"""

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
from pathlib import Path

MU = 3.986e5
A_C = 15000.0
E_C = 0.5

OUT_DIR = Path("outputs/tpbvp")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def orbital_params(nu):
    r_c = A_C * (1 - E_C**2) / (1 + E_C * np.cos(nu))
    nu_dot = np.sqrt(MU * A_C * (1 - E_C**2)) / r_c**2
    r_c_dot = np.sqrt(MU / (A_C * (1 - E_C**2))) * E_C * np.sin(nu)
    nu_ddot = -2 * r_c_dot * nu_dot / r_c
    return r_c, nu_dot, nu_ddot


def dynamics_and_jacobian(x):
    """计算 5D NERM 动力学 f(x) (无控制) 和 Jacobian ∂f/∂x (5×5)"""
    dx, dy, dvx, dvy, nu = x
    r_c, nu_dot, nu_ddot = orbital_params(nu)

    dr_c_dnu = A_C * (1 - E_C**2) * E_C * np.sin(nu) / (1 + E_C * np.cos(nu))**2
    r_p = np.sqrt((r_c + dx)**2 + dy**2)
    r3, r5 = r_p**3, r_p**5

    # 引力
    grav_x = -MU * (r_c + dx) / r3 + MU / r_c**2
    grav_y = -MU * dy / r3

    # 动力学 f(x) (5D)
    f = np.array([
        dvx,
        dvy,
        2 * nu_dot * dvy + nu_ddot * dy + nu_dot**2 * dx + grav_x,
        -2 * nu_dot * dvx - nu_ddot * dx + nu_dot**2 * dy + grav_y,
        nu_dot,
    ])

    # Jacobian (5×5)
    dgx_dx = -MU / r3 + 3 * MU * (r_c + dx)**2 / r5
    dgx_dy = 3 * MU * (r_c + dx) * dy / r5
    dgy_dy = -MU / r3 + 3 * MU * dy**2 / r5

    # nu_dot, nu_ddot 对 nu 的导数 (有限差分)
    eps = 1e-8
    _, nd1, ndd1 = orbital_params(nu + eps)
    _, nd0, ndd0 = orbital_params(nu)
    dnu_dot_dnu = (nd1 - nd0) / eps
    dnu_ddot_dnu = (ndd1 - ndd0) / eps

    dgrav_x_dnu = (
        -MU * dr_c_dnu / r3
        + 3 * MU * (r_c + dx) * dr_c_dnu * (r_c + dx) / r5
        - 2 * MU * dr_c_dnu / r_c**3
    ) if abs(r_c) > 1e-9 else 0
    dgrav_y_dnu = 3 * MU * dy * dr_c_dnu / r5 if abs(r5) > 1e-9 else 0

    J = np.zeros((5, 5))
    J[0, 2] = 1.0
    J[1, 3] = 1.0
    J[2, 0] = nu_dot**2 + dgx_dx
    J[2, 1] = nu_ddot + dgx_dy
    J[2, 3] = 2 * nu_dot
    J[2, 4] = (2 * dnu_dot_dnu * dvy + dnu_ddot_dnu * dy
               + 2 * nu_dot * dnu_dot_dnu * dx + dgrav_x_dnu)
    J[3, 0] = -nu_ddot + dgx_dy
    J[3, 1] = nu_dot**2 + dgy_dy
    J[3, 2] = -2 * nu_dot
    J[3, 4] = (-2 * dnu_dot_dnu * dvx - dnu_ddot_dnu * dx
               + 2 * nu_dot * dnu_dot_dnu * dy + dgrav_y_dnu)
    J[4, 4] = dnu_dot_dnu

    return f, J


def tpbvp_ode(tau, y, T, r, gamma, Q):
    """TPBVP ODE: dy/dτ = T * RHS

    y = [x(5), λ(5)]  shape (10,) or (10, N)
    """
    y_arr = np.atleast_2d(y)
    if y_arr.shape[0] != 10:
        y_arr = y_arr.T

    N_pts = y_arr.shape[1]
    result = np.zeros_like(y_arr)

    for i in range(N_pts):
        yi = y_arr[:, i]
        x = yi[:5]
        lam = yi[5:]

        f, J = dynamics_and_jacobian(x)

        c = 0.5 * (1.0 - gamma**(-2)) / r
        u_eff = np.array([-c * lam[2], -c * lam[3]])

        dxdt = f.copy()
        dxdt[2] += u_eff[0]
        dxdt[3] += u_eff[1]

        Qx = np.zeros(5)
        Qx[:4] = Q @ x[:4]
        dlamdt = -2.0 * Qx - J.T @ lam

        result[:, i] = np.concatenate([dxdt, dlamdt])

    # 返回与输入相同形状
    out = T * result
    if y.ndim == 1:
        return out[:, 0]
    return out


def bc_func(ya, yb, x0):
    """边界条件: x(0)=x0, λ(T)=0"""
    return np.concatenate([ya[:5] - x0, yb[5:]])


def solve_tpbvp(
    x0: np.ndarray, T: float, r: float,
    gamma: float = np.sqrt(2), Q: np.ndarray = None, N: int = 200,
) -> dict:
    """求解微分博弈 TPBVP (用 continuation: 从短 T 逐步延长)"""
    if Q is None:
        Q = np.eye(4)

    # ── Continuation: 从 T/8 开始, 逐步翻倍延至 T ──
    T_list = [T / 8, T / 4, T / 2, T]
    prev_sol = None

    for Ti in T_list:
        print(f"    T={Ti:.0f}s ...", end=" ", flush=True)
        tau = np.linspace(0, 1, N)

        if prev_sol is not None and prev_sol.success:
            y_guess = prev_sol.sol(tau)
        else:
            y_guess = np.zeros((10, N))
            for i in range(4):
                y_guess[i] = np.linspace(x0[i], 0.0, N)
            y_guess[4] = np.linspace(x0[4], x0[4] + Ti * 1e-4, N)
            # 小协态初始猜测
            y_guess[5:] = 1e-8 * np.random.randn(5, N)

        sol = solve_bvp(
            lambda t, y: tpbvp_ode(t, y, Ti, r, gamma, Q),
            lambda ya, yb: bc_func(ya, yb, x0),
            tau, y_guess,
            tol=1e-4, max_nodes=50000,
        )

        if sol.success:
            print(f"ok ({len(sol.x)} nodes)")
            prev_sol = sol
        else:
            print(f"FAIL: {sol.message}")

    if prev_sol is None or not prev_sol.success:
        return dict(converged=False, message="All T attempts failed")

    sol = prev_sol
    tau_sol = sol.x
    t_sol = tau_sol * T
    y_sol = sol.y
    x_sol = y_sol[:5]
    lam_sol = y_sol[5:]

    u_p = np.row_stack([-0.5 / r * lam_sol[2], -0.5 / r * lam_sol[3]])
    u_e = np.row_stack([0.5 / (gamma**2 * r) * lam_sol[2],
                         0.5 / (gamma**2 * r) * lam_sol[3]])

    print(f"  TPBVP converged, {len(tau_sol)} nodes")
    return dict(converged=True, t=t_sol, x_traj=x_sol, lam_traj=lam_sol,
                u_p=u_p, u_e=u_e, T=T, r=r, gamma=gamma, sol=sol)


def run_sdre_trajectory(x0, orb, T, r, gamma, dt=20.0):
    """SDRE 闭环仿真, 返回与 TPBVP 兼容的结果"""
    from aerospace.control.sdre_2d import SDREGameController2D

    ctrl = SDREGameController2D(Q=np.eye(4), R=np.eye(2) * r, gamma=gamma)
    N = int(T / dt)
    state = np.zeros(9)
    state[0:4] = x0[:4]
    state[4:8] = np.zeros(4)
    state[8] = x0[4]

    t_now = 0.0
    traj_rel, nu_hist, u_p_hist, u_e_hist = [], [], [], []
    traj_rel.append(x0[:4]); nu_hist.append(x0[4])
    u_p_hist.append(np.zeros(2)); u_e_hist.append(np.zeros(2))

    for k in range(N):
        nu = state[8]; X_p = state[0:4]; x_rel = X_p - state[4:8]
        r_c, nu_dot, nu_ddot = orb.get_orbital_params(nu)
        A_SDC = orb.get_SDC_matrix(X_p, X_p - x_rel, r_c, nu_dot, nu_ddot)
        u_p, u_e = ctrl.compute_control(A_SDC, x_rel, t=t_now, solve_are=True,
                                        x_rel_e=x_rel)
        sol = solve_ivp(orb.dynamics_9d, [t_now, t_now + dt], state,
                        args=(u_p, u_e), method="RK45", rtol=1e-8, atol=1e-10)
        state = sol.y[:, -1]; t_now += dt
        traj_rel.append(state[0:4] - state[4:8])
        nu_hist.append(state[8])
        u_p_hist.append(u_p.copy()); u_e_hist.append(u_e.copy())

    t_arr = dt * np.arange(len(traj_rel))
    x_arr = np.row_stack([np.array(traj_rel).T, np.array(nu_hist)])
    return dict(t=t_arr, x_traj=x_arr,
                u_p=np.array(u_p_hist).T, u_e=np.array(u_e_hist).T)


if __name__ == "__main__":
    from aerospace.dynamics.nerm_2d import OrbitalDynamics2D
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 初始条件
    X_p0 = np.array([100.0, 500.0, 0.01, 0.01])
    X_e0 = np.zeros(4)
    x0 = np.array([*(X_p0 - X_e0), 0.0])
    orb = OrbitalDynamics2D(mu=MU, a_c=A_C, e_c=E_C)

    gamma = np.sqrt(2)
    Q = np.eye(4)
    r_val = 1e13

    # 用 SDRE 获取捕获时间
    print("=== SDRE reference ===")
    sdre_ref = run_sdre_trajectory(x0, orb, T=orb.T_orbit * 2.0, r=r_val, gamma=gamma)
    dist_ref = np.linalg.norm(sdre_ref["x_traj"][:2], axis=0)
    cap_idx = np.argmax(dist_ref < 0.1) if np.any(dist_ref < 0.1) else len(dist_ref) - 1
    T_cap = sdre_ref["t"][cap_idx]
    print(f"  SDRE capture at T={T_cap:.0f}s ({T_cap/3600:.2f}h)")

    # TPBVP: 用 SDRE 捕获时间
    T_tpbvp = T_cap * 1.05
    print(f"\n=== TPBVP: T={T_tpbvp:.0f}s ===")
    res = solve_tpbvp(x0, T=T_tpbvp, r=r_val, gamma=gamma, Q=Q, N=150)

    # ── 绘图 ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"TPBVP vs SDRE: R={r_val:.0e}, γ={gamma:.3f}", fontsize=13)

    t_s = sdre_ref["t"]
    for name, t_arr, x_arr, u_p, u_e, c, ls in [
        ("SDRE", t_s, sdre_ref["x_traj"], sdre_ref["u_p"], sdre_ref["u_e"],
         "tab:blue", "--"),
    ]:
        pass  # placeholder for labeling

    # Distance
    ax = axes[0, 0]
    d_s = np.linalg.norm(sdre_ref["x_traj"][:2], axis=0)
    ax.semilogy(t_s / 3600, d_s, "b--", lw=1.5, label="SDRE")
    if res["converged"]:
        d_t = np.linalg.norm(res["x_traj"][:2], axis=0)
        ax.semilogy(res["t"] / 3600, d_t, "r-", lw=1.5, label="TPBVP")
        ax.axvline(T_cap / 3600, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("Time (h)"); ax.set_ylabel("Distance (km)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)
    ax.set_title("Relative Distance")

    # 2D Trajectory
    ax = axes[0, 1]
    ax.plot(sdre_ref["x_traj"][0], sdre_ref["x_traj"][1], "b--", lw=1.5, label="SDRE")
    if res["converged"]:
        ax.plot(res["x_traj"][0], res["x_traj"][1], "r-", lw=1.5, label="TPBVP")
    ax.scatter(x0[0], x0[1], c="g", s=50, marker="o", zorder=5)
    ax.scatter(0, 0, c="r", s=50, marker="*", zorder=5)
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4); ax.set_aspect("equal")
    ax.set_title("2D Relative Trajectory")

    # u_p magnitude
    ax = axes[0, 2]
    ax.plot(t_s / 3600, np.linalg.norm(sdre_ref["u_p"], axis=0) * 1e6, "b--", lw=1.5, label="SDRE")
    if res["converged"]:
        ax.plot(res["t"] / 3600, np.linalg.norm(res["u_p"], axis=0) * 1e6, "r-", lw=1.5, label="TPBVP")
    ax.set_xlabel("Time (h)"); ax.set_ylabel("||u_p|| (um/s^2)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)
    ax.set_title("Pursuer Control Magnitude")

    # u_e magnitude
    ax = axes[1, 0]
    ax.plot(t_s / 3600, np.linalg.norm(sdre_ref["u_e"], axis=0) * 1e6, "b--", lw=1.5, label="SDRE")
    if res["converged"]:
        ax.plot(res["t"] / 3600, np.linalg.norm(res["u_e"], axis=0) * 1e6, "r-", lw=1.5, label="TPBVP")
    ax.set_xlabel("Time (h)"); ax.set_ylabel("||u_e|| (um/s^2)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)
    ax.set_title("Evader Control Magnitude")

    # u_p components
    ax = axes[1, 1]
    ax.plot(t_s / 3600, sdre_ref["u_p"][0] * 1e6, "b--", lw=1, alpha=0.7, label="SDRE u_x")
    ax.plot(t_s / 3600, sdre_ref["u_p"][1] * 1e6, "b:", lw=1, alpha=0.7, label="SDRE u_y")
    if res["converged"]:
        ax.plot(res["t"] / 3600, res["u_p"][0] * 1e6, "r-", lw=1, label="TPBVP u_x")
        ax.plot(res["t"] / 3600, res["u_p"][1] * 1e6, "r:", lw=1, label="TPBVP u_y")
    ax.set_xlabel("Time (h)"); ax.set_ylabel("u (um/s^2)")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.4)
    ax.set_title("Pursuer Control Components")

    # Costates
    ax = axes[1, 2]
    if res["converged"]:
        for i, (name, c) in enumerate(zip(
            ["λ_x", "λ_y", "λ_vx", "λ_vy", "λ_ν"],
            ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
        )):
            ax.plot(res["t"] / 3600, res["lam_traj"][i], color=c, lw=1, label=name)
    ax.set_xlabel("Time (h)"); ax.legend(fontsize=7); ax.grid(True, alpha=0.4)
    ax.set_title("Costates λ(t)")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "tpbvp_vs_sdre.png", dpi=150)
    plt.close(fig)
    print(f"\nPlot: {OUT_DIR / 'tpbvp_vs_sdre.png'}")

    # ── 汇总 ──
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    if res["converged"]:
        n = min(len(res["t"]), len(t_s))
        dt_ = 1.0
        J_t = float(np.trapezoid(
            np.sum(res["x_traj"][:4, :n] * (Q @ res["x_traj"][:4, :n]), axis=0)
            + r_val * (np.sum(res["u_p"][:, :n]**2, axis=0)
                       - gamma**2 * np.sum(res["u_e"][:, :n]**2, axis=0)),
            res["t"][:n]
        ))
        J_s = float(np.trapezoid(
            np.sum(sdre_ref["x_traj"][:4, :n] * (Q @ sdre_ref["x_traj"][:4, :n]), axis=0)
            + r_val * (np.sum(sdre_ref["u_p"][:, :n]**2, axis=0)
                       - gamma**2 * np.sum(sdre_ref["u_e"][:, :n]**2, axis=0)),
            t_s[:n]
        ))
        # Also compute ||u_p(t) - u_sdre(t)|| integrated error
        u_err = float(np.trapezoid(
            np.linalg.norm(res["u_p"][:, :n] - sdre_ref["u_p"][:, :n], axis=0),
            res["t"][:n]
        ))
        print(f"  TPBVP J* = {J_t:.6e}")
        print(f"  SDRE  J  = {J_s:.6e}")
        print(f"  ΔJ = {J_t - J_s:.3e}  ({(J_t - J_s)/abs(J_s)*100:.4f}%)")
        print(f"  ∫||u_tpbvp - u_sdre|| dt = {u_err:.3e}")
    else:
        print(f"  TPBVP failed: {res.get('message', 'unknown')}")
