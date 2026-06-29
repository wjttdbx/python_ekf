"""
实验：四种 EKF 观测模式对比

1. 仅测角 (AO)              [az, el]         — 速度不可观测
2. 距离+角度 (RA)           [ρ, az, el]      — 距离可观，速度不可直接观
3. 角度+Doppler (AD)        [az, el, ρ̇]      — 速度可观，距离不可观
4. 全观测 (FULL)            [ρ, az, el, ρ̇]    — 全可观

所有模式用 Hypergame γ_p=3.0 以排除控制策略差异。
"""
import zhplot
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp

from aerospace.dynamics.nerm import OrbitalDynamics
from aerospace.estimation.ekf import RelativeStateEKF

DEG2RAD = np.pi / 180.0
MU = 3.986e5
DT = 10.0


def run_obs_mode(orb, X_p0, X_e0, nu0, mode_name, seed=42):
    """运行指定观测模式的闭环仿真（Hypergame γ_p=3.0）。"""

    x_rel0 = X_p0 - X_e0
    initial_dist = float(np.linalg.norm(x_rel0[:3]))

    SIGMA_ANG = 0.008 * DEG2RAD
    SIGMA_RANGE = 0.01       # 10 m 测距
    SIGMA_DOPPLER = 0.01     # 10 m/s Doppler
    Q_proc = np.diag([5e-4, 5e-4, 5e-4, 5e-8, 5e-8, 5e-8])

    scale_d = initial_dist / 3000.0
    sigma_pos = 10.0 * scale_d
    sigma_vel = 1e-3 * scale_d

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(6) * np.array([sigma_pos]*3 + [sigma_vel]*3)
    x0_est = x_rel0 + noise
    P0 = np.diag([sigma_pos**2]*3 + [sigma_vel**2]*3)

    # ── 根据模式配置 EKF ───────────────────────────────────────────────
    if "AO" in mode_name:
        angles_only, use_doppler = True, False
        R_meas = np.diag([SIGMA_ANG**2, SIGMA_ANG**2])
    elif "RA" in mode_name:
        angles_only, use_doppler = False, False
        R_meas = np.diag([SIGMA_RANGE**2, SIGMA_ANG**2, SIGMA_ANG**2])
    elif "AD" in mode_name:
        angles_only, use_doppler = True, True
        R_meas = np.diag([SIGMA_ANG**2, SIGMA_ANG**2, SIGMA_DOPPLER**2])
    elif "FULL" in mode_name:
        angles_only, use_doppler = False, True
        R_meas = np.diag([SIGMA_RANGE**2, SIGMA_ANG**2, SIGMA_ANG**2, SIGMA_DOPPLER**2])
    else:
        raise ValueError(f"Unknown mode: {mode_name}")

    ekf = RelativeStateEKF(x0=x0_est.copy(), P0=P0.copy(),
                           Q=Q_proc, R=R_meas,
                           angles_only=angles_only, use_doppler=use_doppler)
    rng_sim = np.random.default_rng(seed + 1)

    # ── 控制器：Hypergame γ_p=3.0 ─────────────────────────────────────
    Q_ctrl = np.eye(6)
    R_base = np.eye(3) * 1e13
    gamma_p = 3.0
    gamma_e = np.sqrt(2)
    R_eff_p = R_base / (1.0 - gamma_p**(-2))
    R_eff_e = R_base / (1.0 - gamma_e**(-2))
    B_p = np.zeros((6, 3)); B_p[3:, :] = np.eye(3); B_e = -B_p

    state = np.zeros(13)
    state[0:6] = X_p0; state[6:12] = X_e0; state[12] = nu0
    t = 0.0

    N = int(5.0 * orb.T_orbit / DT)
    dist_hist = np.zeros(N + 1)
    err_hist = np.zeros((6, N + 1))      # 位置+速度估计误差
    P_trace_hist = np.zeros(N + 1)        # 协方差迹
    u_p_hist = np.zeros(N + 1)

    last_P_p = None
    captured = False

    for k in range(N):
        nu = state[12]
        r_c, nu_dot, nu_ddot = orb.get_orbital_params(nu)
        X_p_true = state[0:6]; X_e_true = state[6:12]
        x_true_rel = X_p_true - X_e_true
        x_est = ekf.x
        X_e_est = X_p_true - x_est

        A_est = orb.get_SDC_matrix(X_p_true, X_e_est, r_c, nu_dot, nu_ddot)

        # ── 追方用 γ_p 解 P_p ──
        S_p = B_p @ np.linalg.inv(R_eff_p) @ B_p.T
        norm_Q = np.linalg.norm(Q_ctrl, "fro")
        norm_S_p = np.linalg.norm(S_p, "fro")
        alpha_p = np.sqrt(norm_Q / max(norm_S_p, 1e-30))
        try:
            P_p_bar = solve_continuous_are(A_est, B_p, Q_ctrl/alpha_p, R_eff_p/alpha_p)
            last_P_p = alpha_p * P_p_bar
        except Exception:
            if last_P_p is None:
                last_P_p = np.zeros((6, 6))
        P_p = last_P_p

        # ── 逃方用 γ_e 解 P_e ──
        S_e = B_p @ np.linalg.inv(R_eff_e) @ B_p.T
        norm_S_e = np.linalg.norm(S_e, "fro")
        alpha_e = np.sqrt(norm_Q / max(norm_S_e, 1e-30))
        try:
            P_e_bar = solve_continuous_are(A_est, B_p, Q_ctrl/alpha_e, R_eff_e/alpha_e)
            P_e = alpha_e * P_e_bar
        except Exception:
            P_e = np.zeros((6, 6))

        u_p = -np.linalg.inv(R_eff_p) @ B_p.T @ P_p @ x_est
        u_e = gamma_e**(-2) * np.linalg.inv(R_eff_e) @ B_e.T @ P_e @ x_true_rel

        # 传播
        sol = solve_ivp(orb.dynamics_13d, [t, t + DT], state,
                        args=(u_p, u_e), method="RK45", rtol=1e-8, atol=1e-10)
        state = sol.y[:, -1]
        t += DT

        # EKF
        x_priori, P_priori = ekf.predict(A_est, B_p, u_p, u_e, DT)
        z_true = RelativeStateEKF.measure(state[0:6], state[6:12],
                                          angle_only=angles_only,
                                          use_doppler=use_doppler)
        z_meas = z_true + rng_sim.multivariate_normal(np.zeros(len(z_true)), R_meas)
        ekf.update(x_priori, P_priori, z_meas)

        dist_hist[k + 1] = np.linalg.norm(state[0:3] - state[6:9])
        err_hist[:, k + 1] = ekf.x - x_true_rel
        P_trace_hist[k + 1] = np.trace(ekf.P)
        u_p_hist[k + 1] = np.linalg.norm(u_p)

        if dist_hist[k + 1] < 0.1:
            captured = True; N_actual = k + 1; break
    else:
        N_actual = N; t = np.nan

    sl = slice(0, N_actual + 1)
    return {
        "mode": mode_name, "captured": captured,
        "t_capture": t if captured else np.nan,
        "dist": dist_hist[sl], "err": err_hist[:, sl],
        "P_trace": P_trace_hist[sl], "u_p_norm": u_p_hist[sl],
        "final_dist": dist_hist[N_actual],
    }


def main():
    orb = OrbitalDynamics(mu=MU, a_c=15000.0, e_c=0.5)
    X_p0 = np.array([500.0, 500.0, 500.0, 0.01, 0.01, 0.01])
    X_e0 = np.zeros(6); nu0 = 0.0
    x_rel0 = X_p0 - X_e0
    print(f"初始距离: {np.linalg.norm(x_rel0[:3]):.1f} km  周期: {orb.T_orbit/3600:.2f} h\n")

    MODES = {
        "AO":   "仅测角 [az, el]",
        "RA":   "距离+角度 [ρ, az, el]",
        "AD":   "角度+Doppler [az, el, ρ̇]",
        "FULL": "全观测 [ρ, az, el, ρ̇]",
    }

    results = {}
    for key, label in MODES.items():
        print(f"运行 {label}...")
        results[key] = run_obs_mode(orb, X_p0, X_e0, nu0, key, seed=42)
        r = results[key]
        s = f"捕获 {r['t_capture']/3600:.2f} h" if r["captured"] else f"未捕获 {r['final_dist']:.1f} km"
        print(f"  → {s}\n")

    # ─── 绘图 ───────────────────────────────────────────────────────────
    T_orb = orb.T_orbit
    colors = {"AO": "tab:orange", "RA": "tab:blue", "AD": "tab:red", "FULL": "tab:green"}
    labels = {"AO": "仅测角", "RA": "距离+角度", "AD": "角度+Doppler", "FULL": "全观测"}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: 位置误差（三轴）
    axis_names = ["x (径向)", "y (沿迹)", "z (法向)"]
    for j in range(3):
        ax = axes[0, j]
        for key in MODES:
            r = results[key]
            t_h = np.arange(r["err"].shape[1]) * DT / T_orb
            ax.plot(t_h, np.abs(r["err"][j, :]), color=colors[key],
                    linewidth=1.2, label=labels[key])
        ax.set_ylabel(f"|Δ{axis_names[j]}| (km)")
        ax.set_title(f"位置估计误差 — {axis_names[j]}")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.4)
        if j == 2:
            ax.legend(fontsize=7)

    # Row 2: 速度误差（三轴） + 协方差迹
    for j in range(3):
        ax = axes[1, j]
        if j < 2:
            # 速度误差
            for key in MODES:
                r = results[key]
                t_h = np.arange(r["err"].shape[1]) * DT / T_orb
                ax.plot(t_h, np.abs(r["err"][3 + j, :]), color=colors[key],
                        linewidth=1.2, label=labels[key])
            ax.set_ylabel(f"|Δv{axis_names[j][0]}| (km/s)")
            ax.set_title(f"速度估计误差 — {['vx (径向)', 'vy (沿迹)'][j]}")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.4)
            ax.set_xlabel("时间 (轨道周期)")
        else:
            # 协方差迹
            for key in MODES:
                r = results[key]
                t_h = np.arange(len(r["P_trace"])) * DT / T_orb
                ax.plot(t_h, r["P_trace"], color=colors[key],
                        linewidth=1.2, label=labels[key])
            ax.set_ylabel("tr(P)")
            ax.set_title("EKF 协方差迹 (不确定性)")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.4)
            ax.set_xlabel("时间 (轨道周期)")
            ax.legend(fontsize=7)

    plt.suptitle("四种 EKF 观测模型对比 (Hypergame γ_p=3.0, a=15000 km, e=0.5)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = Path("outputs/figures/obs_model_comparison.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"图已保存: {out_path}")

    # ─── 汇总数据 ───────────────────────────────────────────────────────
    import csv
    data_dir = Path("outputs/data/obs_model_comparison")
    data_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "captured", "capture_time_h", "final_dist_km",
                     "final_pos_err_km", "final_vel_err_kms", "final_P_trace"])
        for key in MODES:
            r = results[key]
            final_pos_err = np.linalg.norm(r["err"][:3, -1])
            final_vel_err = np.linalg.norm(r["err"][3:, -1])
            w.writerow([key, r["captured"],
                        f"{r['t_capture']/3600:.4f}" if r["captured"] else "N/A",
                        f"{r['final_dist']:.4f}", f"{final_pos_err:.4f}",
                        f"{final_vel_err:.6f}", f"{r['P_trace'][-1]:.6e}"])

    # Save time series for the "winning" mode comparison
    for key in MODES:
        r = results[key]
        n = len(r["dist"])
        stride = max(1, n // 2000)
        rows = []
        for i in range(0, n, stride):
            t_h = i * DT / T_orb
            rows.append([t_h, r["dist"][i], r["err"][0, i], r["err"][1, i],
                         r["err"][2, i], r["err"][3, i], r["err"][4, i],
                         r["err"][5, i], r["P_trace"][i], r["u_p_norm"][i]])
        np.savetxt(data_dir / f"{key}.csv", np.array(rows),
                   delimiter=",",
                   header="time_orbits,dist,err_x,err_y,err_z,err_vx,err_vy,err_vz,P_trace,u_p_norm",
                   comments="", fmt=["%.6f"] + ["%.8e"]*9)
    print(f"数据已保存: {data_dir}/")
    print("  summary.csv  +  " + ", ".join(f"{k}.csv" for k in MODES))


if __name__ == "__main__":
    main()
