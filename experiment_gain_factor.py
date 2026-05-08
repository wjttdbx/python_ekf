"""
时变增益因子 f(t_go): 修正 SDRE 有限时间水平次优性 (纯追捕版)

从 TPBVP 解中在每个状态点对比:
  f(t) = ||u_tpbvp(t)|| / ||u_SDRE(x(t))||

其中 u_SDRE(x) 是在同一状态下求解 ARE 得到的控制。
拟合 f(t_go) 模型用于缩放 SDRE 控制。
"""

import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from pathlib import Path

MU = 3.986e5; A_C = 15000.0; E_C = 0.5
OUT_DIR = Path("outputs/tpbvp")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_sdre_control_at_state(x_rel, nu, orb, r, gamma):
    """在给定状态下求解 ARE 并计算 SDRE 控制"""
    X_p = x_rel.copy()
    X_e = np.zeros(4)
    r_c, nu_dot, nu_ddot = orb.get_orbital_params(nu)
    A_SDC = orb.get_SDC_matrix(X_p, X_e, r_c, nu_dot, nu_ddot)

    Q = np.eye(4)
    B = np.zeros((4, 2)); B[2:, :] = np.eye(2)
    R = np.eye(2) * r
    R_eff = R / (1.0 - gamma**(-2))

    try:
        P = solve_continuous_are(A_SDC, B, Q, R_eff)
    except Exception:
        return np.zeros(2)

    return -np.linalg.inv(R) @ B.T @ P @ x_rel


def extract_gain_factor(t, x_traj, u_traj, T, orb, r, gamma):
    """在每个时间点计算 f(t) = ||u_tpbvp|| / ||u_sdre(x)|| 并关联 t_go"""
    t_go = T - t
    factors = []
    for i in range(0, len(t_go), 3):  # 每 3 个点取一个
        x = x_traj[:4, i]
        nu = x_traj[4, i]
        u_opt = u_traj[:, i]
        u_sdre = compute_sdre_control_at_state(x, nu, orb, r, gamma)
        norm_opt = np.linalg.norm(u_opt)
        norm_sdre = np.linalg.norm(u_sdre)
        if norm_sdre > 1e-15:
            factors.append((t_go[i], norm_opt / norm_sdre))
    return np.array(factors)


def fit_gain_factor(t_go, f_vals):
    """拟合 f(t_go) = 1 - exp(-α * t_go)"""
    t_go = np.asarray(t_go); f_vals = np.asarray(f_vals)
    def model(t, alpha):
        return 1.0 - np.exp(-alpha * t)
    try:
        popt, _ = curve_fit(model, t_go, f_vals, p0=[1e-4], maxfev=10000)
        alpha = popt[0]
    except Exception:
        alpha = 1e-4
    return dict(alpha=alpha)


def run_sdre_with_gain_factor(x0, orb, T, r, gamma, gain_fn, dt=20.0):
    """SDRE 仿真，每步用 gain_fn(t_go) 缩放控制"""
    from aerospace.control.sdre_2d import SDREGameController2D
    ctrl = SDREGameController2D(Q=np.eye(4), R=np.eye(2) * r, gamma=gamma)
    N = int(T / dt)
    state = np.zeros(9); state[0:4] = x0[:4]; state[4:8] = np.zeros(4); state[8] = x0[4]
    t_now = 0.0
    traj_rel, u_hist = [x0[:4].copy()], [np.zeros(2)]
    for k in range(N):
        nu = state[8]; X_p = state[0:4]; x_rel = X_p - state[4:8]
        r_c, nu_dot, nu_ddot = orb.get_orbital_params(nu)
        A_SDC = orb.get_SDC_matrix(X_p, X_p - x_rel, r_c, nu_dot, nu_ddot)
        u_p_s, u_e_s = ctrl.compute_control(
            A_SDC, x_rel, t=t_now, solve_are=True, x_rel_e=x_rel)
        f = gain_fn(T - t_now)
        u_p, u_e = f * u_p_s, f * u_e_s
        sol = solve_ivp(orb.dynamics_9d, [t_now, t_now + dt], state,
                        args=(u_p, u_e), method="RK45", rtol=1e-8, atol=1e-10)
        state = sol.y[:, -1]; t_now += dt
        traj_rel.append(state[0:4] - state[4:8]); u_hist.append(u_p.copy())
    t_arr = dt * np.arange(len(traj_rel))
    x_arr = np.vstack([np.array(traj_rel).T, np.full((1, len(traj_rel)), state[8])])
    return dict(t=t_arr, x_traj=x_arr, u_p=np.array(u_hist).T)


def compute_pursuit_J(x_traj, u_p, r):
    """纯追捕代价 J = ∫ (x^T Q x + u^T R u) dt"""
    Q = np.eye(4); n = x_traj.shape[1]; J = 0.0
    for i in range(n):
        x = x_traj[:4, i]; u = u_p[:, i]
        J += x @ Q @ x + r * np.sum(u**2)
    return float(J)


if __name__ == "__main__":
    from aerospace.dynamics.nerm_2d import OrbitalDynamics2D
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from experiment_tpbvp_compare import solve_tpbvp

    X_p0 = np.array([100.0, 500.0, 0.01, 0.01])
    x0 = np.array([*(X_p0 - np.zeros(4)), 0.0])
    orb = OrbitalDynamics2D(mu=MU, a_c=A_C, e_c=E_C)
    r_val = 1e13
    gamma = 1e6  # pure pursuit

    # SDRE 参考
    print("=== Reference SDRE ===")
    from experiment_tpbvp_compare import run_sdre_trajectory
    sdre_ref = run_sdre_trajectory(x0, orb, T=orb.T_orbit * 2.0, r=r_val, gamma=gamma)
    dist_ref = np.linalg.norm(sdre_ref["x_traj"][:2], axis=0)
    cap_idx = np.argmax(dist_ref < 0.1) if np.any(dist_ref < 0.1) else len(dist_ref)-1
    T_cap = sdre_ref["t"][cap_idx]
    print(f"SDRE capture: T={T_cap:.0f}s ({T_cap/3600:.2f}h)")

    # TPBVP
    T_tpbvp = T_cap * 1.05
    print(f"\n=== TPBVP pure pursuit (T={T_tpbvp:.0f}s) ===")
    res = solve_tpbvp(x0, T=T_tpbvp, r=r_val, gamma=gamma, Q=np.eye(4), N=100)
    if not res["converged"]:
        print("TPBVP failed"); exit(1)

    # 提取增益因子
    gains = extract_gain_factor(res["t"], res["x_traj"], res["u_p"],
                                T_tpbvp, orb, r_val, gamma)
    fit = fit_gain_factor(gains[:, 0], gains[:, 1])
    alpha = fit['alpha']
    print(f"\nGain fit: alpha={alpha:.6e}")
    for tgo_h in [0.5, 1, 2, 4, 8]:
        fv = 1.0 - np.exp(-alpha * tgo_h * 3600)
        print(f"  f({tgo_h}h) = {fv:.4f}")

    # SDRE + gain factor
    print("\n=== SDRE + f(t_go) ===")
    gf_res = run_sdre_with_gain_factor(
        x0, orb, T_tpbvp, r_val, gamma,
        gain_fn=lambda tgo: 1.0 - np.exp(-alpha * max(tgo, 0))
    )

    # 代价对比
    J_t = compute_pursuit_J(res["x_traj"], res["u_p"], r_val)
    J_s = compute_pursuit_J(sdre_ref["x_traj"], sdre_ref["u_p"], r_val)
    J_g = compute_pursuit_J(gf_res["x_traj"], gf_res["u_p"], r_val)

    print(f"\n{'='*60}")
    print("Pure Pursuit Cost J = ∫(x^T Q x + u^T R u) dt")
    print(f"{'='*60}")
    print(f"  TPBVP (true optimum):  J = {J_t:.4e}")
    print(f"  SDRE (steady gain):    J = {J_s:.4e}")
    print(f"  SDRE + f(t_go):        J = {J_g:.4e}")
    print(f"  Improv over SDRE:      {(1-J_g/J_s)*100:.1f}%")
    print(f"  Gap to true optimum:   {(J_g/J_t-1)*100:.1f}%")

    # 绘图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Gain Factor f(t_go) — Pure Pursuit  (alpha={alpha:.3e})",
                 fontsize=13)

    # f vs t_go
    ax = axes[0, 0]
    ax.scatter(gains[:, 0]/3600, gains[:, 1], s=5, alpha=0.4, label="||u_opt||/||u_sdre||")
    tgo_p = np.linspace(0, T_tpbvp, 500)
    ax.plot(tgo_p/3600, 1.0 - np.exp(-alpha * tgo_p), "r-", lw=2, label="fit")
    ax.set_xlabel("Time-to-go (h)"); ax.set_ylabel("f = ||u_opt||/||u_sdre||")
    ax.set_title("Gain Factor from TPBVP"); ax.legend(fontsize=8); ax.grid(alpha=0.4)

    # Distance
    ax = axes[0, 1]
    for label, t_a, x_a, c, ls in [
        ("TPBVP", res["t"], res["x_traj"], "k", "-"),
        ("SDRE", sdre_ref["t"], sdre_ref["x_traj"], "b", "--"),
        ("SDRE+f", gf_res["t"], gf_res["x_traj"], "r", "-."),
    ]:
        ax.semilogy(t_a/3600, np.linalg.norm(x_a[:2], axis=0),
                    color=c, ls=ls, lw=1.5, label=label)
    ax.set_xlabel("Time (h)"); ax.set_ylabel("Distance (km)")
    ax.set_title("Distance vs Time"); ax.legend(fontsize=8); ax.grid(alpha=0.4)

    # 2D
    ax = axes[0, 2]
    for label, x_a, c, ls in [("TPBVP", res["x_traj"], "k", "-"),
                                ("SDRE", sdre_ref["x_traj"], "b", "--"),
                                ("SDRE+f", gf_res["x_traj"], "r", "-.")]:
        ax.plot(x_a[0], x_a[1], color=c, ls=ls, lw=1.5, label=label)
    ax.scatter(x0[0], x0[1], c="g", s=50, zorder=5)
    ax.scatter(0, 0, c="r", s=50, marker="*", zorder=5)
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
    ax.set_title("2D Trajectory"); ax.legend(fontsize=8); ax.grid(alpha=0.4); ax.set_aspect("equal")

    # Control
    ax = axes[1, 0]
    for label, t_a, u_a, c, ls in [
        ("TPBVP", res["t"], res["u_p"], "k", "-"),
        ("SDRE", sdre_ref["t"], sdre_ref["u_p"], "b", "--"),
        ("SDRE+f", gf_res["t"], gf_res["u_p"], "r", "-."),
    ]:
        ax.plot(t_a/3600, np.linalg.norm(u_a, axis=0)*1e6,
                color=c, ls=ls, lw=1.5, label=label)
    ax.set_xlabel("Time (h)"); ax.set_ylabel("||u|| (um/s^2)")
    ax.set_title("Control Magnitude"); ax.legend(fontsize=8); ax.grid(alpha=0.4)

    # Control u_x
    ax = axes[1, 1]
    for label, t_a, u_a, c, ls in [
        ("TPBVP", res["t"], res["u_p"], "k", "-"),
        ("SDRE", sdre_ref["t"], sdre_ref["u_p"], "b", "--"),
        ("SDRE+f", gf_res["t"], gf_res["u_p"], "r", "-."),
    ]:
        ax.plot(t_a/3600, u_a[0]*1e6, color=c, ls=ls, lw=1, label=f"{label} u_x")
    ax.set_xlabel("Time (h)"); ax.set_ylabel("u_x (um/s^2)")
    ax.set_title("Control u_x"); ax.legend(fontsize=7); ax.grid(alpha=0.4)

    # f(t_go) during sim
    ax = axes[1, 2]
    f_sim = [1.0 - np.exp(-alpha * max(T_tpbvp - tt, 0)) for tt in gf_res["t"]]
    ax.plot(gf_res["t"]/3600, f_sim, "r-", lw=2)
    ax.set_xlabel("Time (h)"); ax.set_ylabel("f(t_go)"); ax.set_ylim(0, 1.05)
    ax.set_title("Gain Factor During Simulation"); ax.grid(alpha=0.4)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "gain_factor_pursuit.png", dpi=150)
    plt.close(fig)
    print(f"\nPlot: {OUT_DIR / 'gain_factor_pursuit.png'}")
