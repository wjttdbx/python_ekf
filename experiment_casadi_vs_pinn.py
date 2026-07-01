"""
CasADi 最优控制 vs SDRE vs 最好 PINN — 同代价函数公平对比

使用与 SDRE 相同的代价函数: J = ∫(xᵀQx + uᵀRu) dt  (纯追捕)
在同一初始条件下比较三种方法的真实代价和轨迹。
"""

import sys
import time
import numpy as np
from pathlib import Path
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are, LinAlgError
import casadi as ca

AERO = Path(__file__).parent.parent.parent / "python_aerospace"
sys.path.insert(0, str(AERO))
from aerospace.dynamics.nerm import OrbitalDynamics
from aerospace.control.neural import NeuralSDREController


MU = 3.986e5; A_C = 15000.0; E_C = 0.5


# ──── CasADi: 直接配点法 ────
def _orbital_params(nu):
    r_c = A_C * (1 - E_C**2) / (1 + E_C * ca.cos(nu))
    nu_dot = ca.sqrt(MU * A_C * (1 - E_C**2)) / r_c**2
    r_c_dot = ca.sqrt(MU / (A_C * (1 - E_C**2))) * E_C * ca.sin(nu)
    nu_ddot = -2 * r_c_dot * nu_dot / r_c
    return r_c, nu_dot, nu_ddot


def _dynamics_rhs(x, u):
    dx, dy, dz, dvx, dvy, dvz, nu = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
    r_c, nu_dot, nu_ddot = _orbital_params(nu)
    r_p = ca.sqrt((r_c + dx)**2 + dy**2 + dz**2)
    grav_x = -MU * (r_c + dx) / r_p**3 + MU / r_c**2
    grav_y = -MU * dy / r_p**3
    grav_z = -MU * dz / r_p**3
    ddx = 2*nu_dot*dvy + nu_ddot*dy + nu_dot**2*dx + grav_x + u[0]
    ddy = -2*nu_dot*dvx - nu_ddot*dx + nu_dot**2*dy + grav_y + u[1]
    ddz = grav_z + u[2]
    return ca.vertcat(dvx, dvy, dvz, ddx, ddy, ddz, nu_dot)


def solve_casadi_sdre_cost(x0, T_fixed, Q_diag, R_diag, N=60,
                           warm_x=None, warm_u=None):
    opti = ca.Opti()
    X = opti.variable(7, N+1); U = opti.variable(3, N); dtau = 1.0/N

    if warm_x is not None and warm_u is not None:
        tw = np.linspace(0, T_fixed, warm_x.shape[1])
        tn = np.linspace(0, T_fixed, N+1)
        Xg = np.zeros((7, N+1))
        for i in range(7): Xg[i, :] = np.interp(tn, tw, warm_x[i])
        Ug = np.zeros((3, N))
        tu = np.linspace(0, T_fixed, N)
        for i in range(3): Ug[i, :] = np.interp(tu, tw, warm_u[i])
    else:
        Xg = np.zeros((7, N+1))
        for i in range(7):
            yf = 0.0 if i < 6 else (x0[6] + T_fixed * 1e-4)
            Xg[i, :] = np.linspace(x0[i], yf, N+1)
        Ug = np.zeros((3, N))
    opti.set_initial(X, Xg); opti.set_initial(U, Ug)

    Qm = np.diag(Q_diag); Rm = np.diag(R_diag)
    J_expr = 0
    for k in range(N):
        xk = X[:6, k]; uk = U[:, k]
        J_expr += (xk.T @ Qm @ xk + uk.T @ Rm @ uk) * T_fixed * dtau
    opti.minimize(J_expr)

    for k in range(N):
        xk = X[:, k]; xk1 = X[:, k+1]; uk = U[:, k]
        fk = _dynamics_rhs(xk, uk); fk1 = _dynamics_rhs(xk1, uk)
        opti.subject_to(xk1 == xk + 0.5 * T_fixed * dtau * (fk + fk1))

    opti.subject_to(X[:, 0] == x0)
    opti.subject_to(X[0, -1]**2 + X[1, -1]**2 + X[2, -1]**2 <= 1e-4)
    for k in range(N+1):
        opti.subject_to(opti.bounded(-5000, X[0, k], 5000))
        opti.subject_to(opti.bounded(-5000, X[1, k], 5000))
        opti.subject_to(opti.bounded(-5000, X[2, k], 5000))

    opts = {"print_level": 0, "tol": 1e-6, "max_iter": 5000,
            "linear_solver": "mumps", "hessian_approximation": "limited-memory",
            "acceptable_tol": 1e-3, "acceptable_iter": 15}
    opti.solver("ipopt", {}, opts)

    try:
        sol = opti.solve()
        conv = True
    except RuntimeError as e:
        print(f"  IPOPT: {str(e)[:100]}")
        try:
            sol = opti.debug; conv = False
        except Exception:
            return dict(converged=False, message=str(e))

    Xo = sol.value(X); Uo = sol.value(U)
    tg = np.linspace(0, T_fixed, N+1)
    Ut = np.column_stack([Uo[:, k] if k < N else Uo[:, -1] for k in range(N+1)])
    te = float(np.sqrt(Xo[0, -1]**2 + Xo[1, -1]**2 + Xo[2, -1]**2))
    stats = sol.stats()
    n_iter = stats.get("iter_count", "?")
    print(f"  CasADi: iter={n_iter}, term_err={te:.2e}, converged={conv}")
    return dict(converged=conv or te < 1e-2, t=tg, x=Xo, u=Ut, T=T_fixed)


def compute_cost(t, x, u, Q, R):
    J = 0.0
    for i in range(1, len(t)):
        dt_i = t[i] - t[i-1]
        J += (x[:6, i] @ Q @ x[:6, i] + u[:, i] @ R @ u[:, i]) * dt_i
    return float(J)


def run_sdre(x_rel0, nu0, Q, R, dt=20.0, t_max_h=24):
    orb = OrbitalDynamics(mu=MU, a_c=A_C, e_c=E_C)
    T_end = t_max_h * 3600; N_steps = int(T_end / dt)
    X_p = np.array([*x_rel0[:3], *x_rel0[3:6]]); X_e = np.zeros(6)
    nu = nu0; t_now = 0.0
    R_inv = np.linalg.inv(R)
    B = np.zeros((6, 3)); B[3:, :] = np.eye(3)
    traj, u_hist = [x_rel0[:6].copy()], [np.zeros(3)]
    last_P = None
    t0 = time.perf_counter()
    for _ in range(N_steps):
        x_rel = X_p - X_e
        rc, nd, ndd = orb.get_orbital_params(nu)
        A = orb.get_SDC_matrix(X_p, X_e, rc, nd, ndd)
        try:
            P = solve_continuous_are(A, B, Q, R); last_P = P
        except Exception:
            P = last_P if last_P is not None else np.zeros((6, 6))
        u_p = -R_inv @ B.T @ P @ x_rel
        state = np.concatenate([X_p, X_e, [nu]])
        sol = solve_ivp(orb.dynamics_13d, [t_now, t_now+dt], state,
                        args=(u_p, np.zeros(3)), method="RK45", rtol=1e-8, atol=1e-10)
        state = sol.y[:, -1]
        X_p = state[0:6]; X_e = state[6:12]; nu = state[12]; t_now += dt
        traj.append((X_p - X_e).copy()); u_hist.append(u_p.copy())
        if np.linalg.norm(X_p[:3] - X_e[:3]) < 0.1:
            break
    elapsed = time.perf_counter() - t0
    t_arr = np.arange(len(traj)) * dt
    return dict(t=t_arr, x=np.array(traj).T, u=np.array(u_hist).T, elapsed_s=elapsed)


def run_pinn(x_rel0, nu0, Q, R, ckpt_path, dt=20.0, t_max_h=24):
    orb = OrbitalDynamics(mu=MU, a_c=A_C, e_c=E_C)
    T_end = t_max_h * 3600; N_steps = int(T_end / dt)
    X_p = np.array([*x_rel0[:3], *x_rel0[3:6]]); X_e = np.zeros(6)
    nu = nu0; t_now = 0.0
    traj, u_hist = [x_rel0[:6].copy()], [np.zeros(3)]
    ctrl = NeuralSDREController(checkpoint_path=str(ckpt_path))
    ctrl.r = R; ctrl.r_inv = np.linalg.inv(R)
    ctrl.gamma = 1e10; ctrl.q = Q
    t0 = time.perf_counter()
    for _ in range(N_steps):
        x_rel = X_p - X_e
        rc, nd, ndd = orb.get_orbital_params(nu)
        A = orb.get_SDC_matrix(X_p, X_e, rc, nd, ndd)
        u_p, _ = ctrl.compute_control(A, x_rel, t=t_now)
        state = np.concatenate([X_p, X_e, [nu]])
        sol = solve_ivp(orb.dynamics_13d, [t_now, t_now+dt], state,
                        args=(u_p, np.zeros(3)), method="RK45", rtol=1e-8, atol=1e-10)
        state = sol.y[:, -1]
        X_p = state[0:6]; X_e = state[6:12]; nu = state[12]; t_now += dt
        traj.append((X_p - X_e).copy()); u_hist.append(u_p.copy())
        if np.linalg.norm(X_p[:3] - X_e[:3]) < 0.1:
            break
    elapsed = time.perf_counter() - t0
    t_arr = np.arange(len(traj)) * dt
    fb = ctrl.fallback_calls / max(ctrl.total_calls, 1)
    return dict(t=t_arr, x=np.array(traj).T, u=np.array(u_hist).T,
                elapsed_s=elapsed, fallback_rate=fb)


if __name__ == "__main__":
    X_p0 = np.array([100.0, 500.0, 50.0, 0.01, 0.01, 0.005])
    X_e0 = np.zeros(6); nu0 = 0.0
    x_rel0 = X_p0 - X_e0
    x0_7d = np.array([*x_rel0, nu0])
    Q = np.diag([1.0, 1.0, 1.0, 10.0, 10.0, 10.0])
    R = np.eye(3) * 1e13

    print("=" * 75)
    print("SDRE vs CasADi vs PINN — 3D 纯追捕公平对比 (同 Q,R)")
    print("=" * 75)
    print(f"初始: Xp0={X_p0[:3]} km, Xe0=0")
    print(f"Q=diag(1,1,1,10,10,10), R=1e13·I₃, J=∫(xᵀQx + uᵀRu)dt\n")

    print("[1/3] SDRE exact ARE...")
    sdre = run_sdre(x_rel0, nu0, Q, R)
    T_sdre = sdre["t"][-1]
    J_sdre = compute_cost(sdre["t"], sdre["x"], sdre["u"], Q, R)
    E_sdre = float(np.trapezoid(np.sum(sdre["u"]**2, axis=0), sdre["t"]))
    peak_sdre = float(np.max(np.linalg.norm(sdre["u"], axis=0)))
    print(f"  T={T_sdre/3600:.2f}h, J={J_sdre:.4e}, E={E_sdre:.4e}, peak_u={peak_sdre:.4e} km/s², wall={sdre['elapsed_s']:.1f}s")

    print("\n[2/3] CasADi IPOPT (same Q,R, same T, SDRE warm-start)...")
    nh = np.linspace(nu0, nu0 + T_sdre * 1e-4, sdre["x"].shape[1])
    warm_x = np.vstack([sdre["x"][:6], nh])
    casadi = solve_casadi_sdre_cost(x0_7d, T_sdre, np.diag(Q), np.diag(R),
                                    N=60, warm_x=warm_x, warm_u=sdre["u"])
    if casadi["converged"]:
        J_casadi = compute_cost(casadi["t"], casadi["x"], casadi["u"], Q, R)
        E_casadi = float(np.trapezoid(np.sum(casadi["u"]**2, axis=0), casadi["t"]))
        peak_casadi = float(np.max(np.linalg.norm(casadi["u"], axis=0)))
        print(f"  J={J_casadi:.4e}, E={E_casadi:.4e}, peak_u={peak_casadi:.4e} km/s²")
    else:
        J_casadi, E_casadi, peak_casadi = float("nan"), 0.0, 0.0
        print("  FAILED")

    print("\n[3/3] PINN (old_sdre_pinn)...")
    ckpt = AERO / "checkpoints" / "old" / "sdre_pinn" / "best_model.pt"
    if ckpt.exists():
        pinn = run_pinn(x_rel0, nu0, Q, R, ckpt)
        J_pinn = compute_cost(pinn["t"], pinn["x"], pinn["u"], Q, R)
        E_pinn = float(np.trapezoid(np.sum(pinn["u"]**2, axis=0), pinn["t"]))
        peak_pinn = float(np.max(np.linalg.norm(pinn["u"], axis=0)))
        print(f"  T={pinn['t'][-1]/3600:.2f}h, J={J_pinn:.4e}, E={E_pinn:.4e}, "
              f"peak_u={peak_pinn:.4e} km/s², wall={pinn['elapsed_s']:.1f}s, "
              f"fb={pinn['fallback_rate']*100:.1f}%")
    else:
        pinn = None; J_pinn = float("nan")
        print(f"  Checkpoint not found: {ckpt}")

    print("\n" + "=" * 75)
    print("结果汇总 (同代价函数)")
    print("=" * 75)
    hdr = f"{'Method':<18} {'T(h)':>7} {'J_cost':>14} {'E_ctrl':>12} {'peak_u':>10} {'wall(s)':>8} {'vs CasADi':>10}"
    print(hdr)
    print("-" * 75)

    ts = T_sdre / 3600
    gap_s = (J_sdre / J_casadi - 1) * 100 if not np.isnan(J_casadi) else float("nan")
    gs_str = f"{gap_s:+9.1f}%" if not np.isnan(gap_s) else "       N/A"
    print(f"  {'SDRE exact':<18} {ts:>7.2f} {J_sdre:>14.4e} {E_sdre:>12.4e} {peak_sdre:>10.4e} {sdre['elapsed_s']:>8.1f} {gs_str:>10}")

    if not np.isnan(J_casadi):
        print(f"  {'CasADi Optimal':<18} {ts:>7.2f} {J_casadi:>14.4e} {E_casadi:>12.4e} {peak_casadi:>10.4e} {'(min)':>8} {'baseline':>10}")

    if pinn:
        tp_val = pinn["t"][-1] / 3600
        gap_p = (J_pinn / J_casadi - 1) * 100 if not np.isnan(J_casadi) else float("nan")
        gp_str = f"{gap_p:+9.1f}%" if not np.isnan(gap_p) else "       N/A"
        print(f"  {'PINN (old)':<18} {tp_val:>7.2f} {J_pinn:>14.4e} {E_pinn:>12.4e} {peak_pinn:>10.4e} {pinn['elapsed_s']:>8.1f} {gp_str:>10}")

    print()
    print(f"  SDRE  vs CasADi 代价差: {gs_str}")
    if pinn and not np.isnan(J_casadi):
        print(f"  PINN  vs CasADi 代价差: {gp_str}")
        print(f"  PINN  vs SDRE   代价差: {(J_pinn/J_sdre - 1)*100:+.2f}%")

    print()
    print("-" * 75)
    print("推力限制")
    print("-" * 75)
    print("  CasADi IPOPT: 无硬推力上限 (u_max 约束已移除)")
    print("  SDRE exact:   无硬推力上限 (R=1e13 软约束)")
    print("  PINN:         无硬推力上限 (继承 SDRE 反馈律)")
    pc = peak_casadi * 1e6 if not np.isnan(J_casadi) else 0
    print(f"\n  实际 peak ||u|| (μm/s²): CasADi={pc:.1f}, SDRE={peak_sdre*1e6:.1f}", end="")
    if pinn:
        print(f", PINN={peak_pinn*1e6:.1f}")
    else:
        print()

    # 可视化
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        out_dir = AERO / "outputs" / "casadi_vs_pinn"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("CasADi vs SDRE vs PINN — Same Cost J=int(x^T Q x + u^T R u) dt",
                     fontsize=13, fontweight="bold")
        methods = []
        if not np.isnan(J_casadi):
            methods.append(("CasADi", casadi["t"], casadi["x"], casadi["u"], "k", "-", 2))
        methods.append(("SDRE", sdre["t"], sdre["x"], sdre["u"], "b", "--", 1.5))
        if pinn:
            methods.append(("PINN", pinn["t"], pinn["x"], pinn["u"], "r", ":", 1.5))
        titles = ["Distance", "xy", "xz", "Control", "ux", "uy"]
        for idx, (ax, title) in enumerate(zip(axes.flat, titles)):
            for nm, t, x, u, c, ls, lw in methods:
                if idx == 0:
                    ax.semilogy(t/3600, np.linalg.norm(x[:3], axis=0), c=c, ls=ls, lw=lw, label=nm)
                    ax.set_ylabel("Distance (km)")
                elif idx == 1:
                    ax.plot(x[0], x[1], c=c, ls=ls, lw=lw, label=nm)
                    ax.scatter(x_rel0[0], x_rel0[1], c="g", s=50, zorder=5)
                    ax.scatter(0, 0, c="r", s=50, marker="*", zorder=5)
                    ax.set_ylabel("y (km)")
                elif idx == 2:
                    ax.plot(x[0], x[2], c=c, ls=ls, lw=lw, label=nm)
                    ax.scatter(x_rel0[0], x_rel0[2], c="g", s=50, zorder=5)
                    ax.scatter(0, 0, c="r", s=50, marker="*", zorder=5)
                    ax.set_ylabel("z (km)")
                elif idx == 3:
                    ax.plot(t/3600, np.linalg.norm(u, axis=0)*1e6, c=c, ls=ls, lw=lw, label=nm)
                    ax.set_ylabel("||u|| (um/s^2)")
                elif idx == 4:
                    ax.plot(t/3600, u[0]*1e6, c=c, ls=ls, lw=lw, label=nm)
                    ax.set_ylabel("ux (um/s^2)")
                elif idx == 5:
                    ax.plot(t/3600, u[1]*1e6, c=c, ls=ls, lw=lw, label=nm)
                    ax.set_ylabel("uy (um/s^2)")
            ax.set_xlabel("Time (h)"); ax.legend(fontsize=7); ax.grid(alpha=0.4); ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_dir / "casadi_vs_pinn_fair.png", dpi=150)
        plt.close(fig)
        print(f"\nFigure saved: {out_dir / 'casadi_vs_pinn_fair.png'}")
    except Exception as e:
        print(f"\n  Plot failed: {e}")
