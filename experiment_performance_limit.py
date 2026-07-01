"""
实验：追寻 SDRE 框架的性能极限
1. Hypergame γ_p 扫描 — 找到性能饱和点
2. 有限时域 MPC — 逼近真正的"最优"
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


def run_hypergame(orb, X_p0, X_e0, nu0, gamma_p, ekf_base, t_end, seed=43):
    """运行 Hypergame 模式，返回结果字典。"""
    # 复制 EKF
    ekf = RelativeStateEKF(
        x0=ekf_base.x.copy(), P0=ekf_base.P.copy(),
        Q=ekf_base.Q, R=ekf_base.R, angles_only=True,
    )
    rng = np.random.default_rng(seed)

    Q_ctrl = np.eye(6)
    R_base = np.eye(3) * 1e13
    gamma_e = np.sqrt(2)

    R_eff_e = R_base / (1.0 - gamma_e**(-2))
    R_eff_p = R_base / (1.0 - gamma_p**(-2)) if gamma_p > 1.0 else R_base * 1e6

    B_p = np.zeros((6, 3)); B_p[3:, :] = np.eye(3)
    B_e = -B_p
    _B_ctrl = B_p.copy()

    state = np.zeros(13)
    state[0:6] = X_p0; state[6:12] = X_e0; state[12] = nu0
    t = 0.0

    N = int(t_end / DT)
    dist_hist = np.zeros(N + 1)
    u_p_hist = np.zeros(N + 1)
    u_e_hist = np.zeros(N + 1)

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

        # 追方用 γ_p 解 P_p
        norm_Q = np.linalg.norm(Q_ctrl, "fro")
        S_p = B_p @ np.linalg.inv(R_eff_p) @ B_p.T
        norm_S_p = np.linalg.norm(S_p, "fro")
        if norm_S_p > 1e-30:
            alpha_p = np.sqrt(norm_Q / norm_S_p)
            try:
                P_bar = solve_continuous_are(A_est, B_p, Q_ctrl/alpha_p, R_eff_p/alpha_p)
                last_P_p = alpha_p * P_bar
            except Exception:
                if last_P_p is None:
                    last_P_p = np.zeros((6, 6))
        P_p = last_P_p

        # 逃方用 γ_e=√2 解 P_e (也用 A_est)
        S_e = B_p @ np.linalg.inv(R_eff_e) @ B_p.T
        norm_S_e = np.linalg.norm(S_e, "fro")
        alpha_e = np.sqrt(norm_Q / norm_S_e)
        try:
            P_e_bar = solve_continuous_are(A_est, B_p, Q_ctrl/alpha_e, R_eff_e/alpha_e)
            P_e = alpha_e * P_e_bar
        except Exception:
            P_e = np.zeros((6, 6))

        R_eff_inv_p = np.linalg.inv(R_eff_p)
        R_eff_inv_e = np.linalg.inv(R_eff_e)

        u_p = -R_eff_inv_p @ B_p.T @ P_p @ x_est
        u_e = gamma_e**(-2) * R_eff_inv_e @ B_e.T @ P_e @ x_true_rel

        # 传播
        sol = solve_ivp(
            orb.dynamics_13d, [t, t + DT], state,
            args=(u_p, u_e), method="RK45", rtol=1e-8, atol=1e-10,
        )
        state = sol.y[:, -1]
        t += DT

        # EKF
        x_priori, P_priori = ekf.predict(A_est, _B_ctrl, u_p, u_e, DT)
        z_true = RelativeStateEKF.measure(state[0:6], state[6:12], angle_only=True)
        z_meas = z_true + rng.multivariate_normal(np.zeros(2), ekf.R)
        ekf.update(x_priori, P_priori, z_meas)

        dist_hist[k + 1] = np.linalg.norm(state[0:3] - state[6:9])
        u_p_hist[k + 1] = np.linalg.norm(u_p)
        u_e_hist[k + 1] = np.linalg.norm(u_e)

        if dist_hist[k + 1] < 0.1:
            captured = True
            N_actual = k + 1
            break
    else:
        N_actual = N
        t = np.nan

    sl = slice(0, N_actual + 1)
    return {
        "captured": captured, "t_capture": t if captured else np.nan,
        "dist": dist_hist[sl], "u_p_norm": u_p_hist[sl],
        "u_e_norm": u_e_hist[sl], "final_dist": dist_hist[N_actual],
        "gamma_p": gamma_p,
    }


def run_mpc(orb, X_p0, X_e0, nu0, ekf_base, t_end, horizon_steps=10, seed=43):
    """
    有限时域 MPC：每个时间步求解一个 N_step 的有限时域最优控制问题。
    追方正视非线性动力学，用 CasADi 求解。

    简化：用离散 SDC 传播 + 有限时域 LQ 的 DRE 解来近似非线性 MPC。
    """
    ekf = RelativeStateEKF(
        x0=ekf_base.x.copy(), P0=ekf_base.P.copy(),
        Q=ekf_base.Q, R=ekf_base.R, angles_only=True,
    )
    rng = np.random.default_rng(seed)

    Q_ctrl = np.eye(6)
    R_base = np.eye(3) * 1e13
    gamma_e = np.sqrt(2)
    R_eff = R_base / (1.0 - gamma_e**(-2))

    B_p = np.zeros((6, 3)); B_p[3:, :] = np.eye(3)
    B_e = -B_p
    _B_ctrl = B_p.copy()

    state = np.zeros(13)
    state[0:6] = X_p0; state[6:12] = X_e0; state[12] = nu0
    t = 0.0

    N = int(t_end / DT)
    dist_hist = np.zeros(N + 1)
    u_p_hist = np.zeros(N + 1)
    u_e_hist = np.zeros(N + 1)

    S_eff = B_p @ np.linalg.inv(R_eff) @ B_p.T
    captured = False

    for k in range(N):
        nu = state[12]
        r_c, nu_dot, nu_ddot = orb.get_orbital_params(nu)
        X_p_true = state[0:6]; X_e_true = state[6:12]
        x_true_rel = X_p_true - X_e_true
        x_est = ekf.x
        X_e_est = X_p_true - x_est

        A_est = orb.get_SDC_matrix(X_p_true, X_e_est, r_c, nu_dot, nu_ddot)

        # ── 有限时域 DRE (差分 Riccati 方程) ──────────────────────────
        # 向后递推 horizon_steps 步
        P_dre = Q_ctrl.copy()  # 终端代价 = Q
        P_seq = [P_dre]
        # 冻结 A_est 做开环预测（标准 SDRE+MPC 做法）
        F = np.eye(6) + A_est * DT
        for _ in range(horizon_steps):
            # DRE 向后步: P_{k} = Q + FᵀP_{k+1}F - FᵀP_{k+1}B(R+BᵀP_{k+1}B)⁻¹BᵀP_{k+1}F
            BPB = B_p.T @ P_dre @ B_p
            K_gain = np.linalg.inv(R_eff + BPB) @ B_p.T @ P_dre @ F
            P_next = Q_ctrl + F.T @ P_dre @ F - F.T @ P_dre @ B_p @ K_gain
            P_dre = P_next
            P_seq.append(P_dre)
        P_seq.reverse()  # P_0, P_1, ..., P_H

        # 使用第一步的有限时域增益
        P_0 = P_seq[0]
        K_0 = np.linalg.inv(R_eff + B_p.T @ P_0 @ B_p) @ B_p.T @ P_0 @ F

        u_p = -np.linalg.inv(R_eff) @ B_p.T @ P_0 @ x_est

        # 逃方仍用无限时域 ARE（固定策略）—— 用辛平衡防崩溃
        try:
            P_are = solve_continuous_are(A_est, B_p, Q_ctrl, R_eff)
        except Exception:
            # 辛平衡回退
            norm_Q = np.linalg.norm(Q_ctrl, "fro")
            S_are = B_p @ np.linalg.inv(R_eff) @ B_p.T
            norm_S = np.linalg.norm(S_are, "fro")
            alpha = np.sqrt(norm_Q / max(norm_S, 1e-30))
            P_are = alpha * solve_continuous_are(A_est, B_p, Q_ctrl/alpha, R_eff/alpha)
        u_e = gamma_e**(-2) * np.linalg.inv(R_eff) @ B_e.T @ P_are @ x_true_rel

        # 传播
        sol = solve_ivp(
            orb.dynamics_13d, [t, t + DT], state,
            args=(u_p, u_e), method="RK45", rtol=1e-8, atol=1e-10,
        )
        state = sol.y[:, -1]
        t += DT

        # EKF
        x_priori, P_priori = ekf.predict(A_est, _B_ctrl, u_p, u_e, DT)
        z_true = RelativeStateEKF.measure(state[0:6], state[6:12], angle_only=True)
        z_meas = z_true + rng.multivariate_normal(np.zeros(2), ekf.R)
        ekf.update(x_priori, P_priori, z_meas)

        dist_hist[k + 1] = np.linalg.norm(state[0:3] - state[6:9])
        u_p_hist[k + 1] = np.linalg.norm(u_p)
        u_e_hist[k + 1] = np.linalg.norm(u_e)

        if dist_hist[k + 1] < 0.1:
            captured = True
            N_actual = k + 1
            break
    else:
        N_actual = N
        t = np.nan

    sl = slice(0, N_actual + 1)
    return {
        "captured": captured, "t_capture": t if captured else np.nan,
        "dist": dist_hist[sl], "u_p_norm": u_p_hist[sl],
        "u_e_norm": u_e_hist[sl], "final_dist": dist_hist[N_actual],
        "method": f"MPC(H={horizon_steps})",
    }


def main():
    orb = OrbitalDynamics(mu=MU, a_c=15000.0, e_c=0.5)
    X_p0 = np.array([500.0, 500.0, 500.0, 0.01, 0.01, 0.01])
    X_e0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    nu0 = 0.0
    x_rel0 = X_p0 - X_e0
    initial_dist = float(np.linalg.norm(x_rel0[:3]))

    SIGMA_ANG = 0.008 * DEG2RAD
    seed = 42
    rng = np.random.default_rng(seed)
    R_meas_ao = np.diag([SIGMA_ANG**2, SIGMA_ANG**2])
    Q_proc = np.diag([5e-4, 5e-4, 5e-4, 5e-8, 5e-8, 5e-8])

    scale_d = initial_dist / 3000.0
    sigma_pos = 10.0 * scale_d
    sigma_vel = 1e-3 * scale_d
    noise = rng.standard_normal(6) * np.array([sigma_pos]*3 + [sigma_vel]*3)
    x0_est = x_rel0 + noise
    P0 = np.diag([sigma_pos**2]*3 + [sigma_vel**2]*3)

    ekf_template = RelativeStateEKF(
        x0=x0_est.copy(), P0=P0.copy(),
        Q=Q_proc, R=R_meas_ao, angles_only=True,
    )

    t_end = 5.0 * orb.T_orbit
    print(f"初始距离: {initial_dist:.1f} km  轨道周期: {orb.T_orbit/3600:.2f} h\n")

    # ─── 1. γ_p 扫描 ───────────────────────────────────────────────────
    gamma_vals = [np.sqrt(2), 1.5, 2.0, 3.0, 5.0, 10.0, 50.0, 100.0]
    hypergame_results = []

    print("=" * 65)
    print("Hypergame γ_p 扫描")
    print("-" * 65)
    for gp in gamma_vals:
        label = f"γ_p={gp:.1f}" if gp > 1.5 else f"γ_p=√2≈{gp:.2f} (Nash)"
        r = run_hypergame(orb, X_p0, X_e0, nu0, gp, ekf_template, t_end, seed=43)
        hypergame_results.append(r)
        status = f"捕获 {r['t_capture']/3600:.2f} h" if r["captured"] else f"未捕获 最终{r['final_dist']:.1f} km"
        print(f"  {label:<22} → {status}")

    # ─── 2. MPC 对照 ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("有限时域 MPC")
    print("-" * 65)
    mpc_results = {}
    for H in [5, 10, 20]:
        r = run_mpc(orb, X_p0, X_e0, nu0, ekf_template, t_end, horizon_steps=H, seed=43)
        mpc_results[H] = r
        status = f"捕获 {r['t_capture']/3600:.2f} h" if r["captured"] else f"未捕获"
        print(f"  MPC(H={H:<3}) → {status}")

    # ─── 3. 汇总 ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("性能极限对比")
    print("-" * 65)

    baseline_t = hypergame_results[0]["t_capture"]  # γ=√2 is at index 0
    print(f"{'方法':<25} {'捕获时间':<12} {'vs Nash':<10}")
    print("-" * 47)
    print(f"{'Nash 基线 (γ=√2)':<25} {baseline_t/3600:.2f} h      —")

    for r in hypergame_results[1:]:
        if r["captured"]:
            delta = (r["t_capture"] - baseline_t) / baseline_t * 100
            print(f"{'Hypergame γ_p='+str(r['gamma_p']):<25} {r['t_capture']/3600:.2f} h      {delta:+.1f}%")

    for H, r in mpc_results.items():
        if r["captured"]:
            delta = (r["t_capture"] - baseline_t) / baseline_t * 100
            print(f"{'MPC H='+str(H):<25} {r['t_capture']/3600:.2f} h      {delta:+.1f}%")
    print("=" * 65)

    # ─── 4. 绘图 ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    T_orb = orb.T_orbit

    # 左: γ_p 扫描 — 捕获时间
    valid = [(r["gamma_p"], r["t_capture"]/3600) for r in hypergame_results if r["captured"]]
    gps, times = zip(*valid)
    axes[0].semilogx(gps, times, "o-", color="tab:blue", markersize=6)
    axes[0].axhline(baseline_t/3600, color="gray", linestyle="--", label=f"Nash 基线 ({baseline_t/3600:.2f} h)")
    # 标注最优点
    best_idx = np.argmin(times)
    axes[0].annotate(f"最优 γ_p={gps[best_idx]:.0f}\n{times[best_idx]:.2f} h",
                     (gps[best_idx], times[best_idx]),
                     xytext=(gps[best_idx]*1.5, times[best_idx]+0.5),
                     arrowprops=dict(arrowstyle="->", color="tab:red"),
                     fontsize=9, color="tab:red")
    axes[0].set_xlabel("γ_p")
    axes[0].set_ylabel("捕获时间 (h)")
    axes[0].set_title("Hypergame γ_p 扫描 → 性能饱和")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.4)

    # 中: 距离曲线 — 选最优 hypergame vs Nash vs MPC
    axes[1].set_yscale("log")
    # Nash baseline
    t_h = np.arange(len(hypergame_results[0]["dist"])) * DT / T_orb
    axes[1].plot(t_h, hypergame_results[0]["dist"], "k-", linewidth=1.5, alpha=0.8,
                 label=f"Nash ({hypergame_results[0]['t_capture']/3600:.2f} h)")
    # Best hypergame
    best_r = hypergame_results[np.argmin([r["t_capture"] if r["captured"] else np.inf for r in hypergame_results])]
    t_h = np.arange(len(best_r["dist"])) * DT / T_orb
    axes[1].plot(t_h, best_r["dist"], color="tab:red", linewidth=2,
                 label=f"Hypergame γ_p={best_r['gamma_p']:.0f} ({best_r['t_capture']/3600:.2f} h)")
    # MPC H=20
    if 20 in mpc_results and mpc_results[20]["captured"]:
        r20 = mpc_results[20]
        t_h = np.arange(len(r20["dist"])) * DT / T_orb
        axes[1].plot(t_h, r20["dist"], color="tab:green", linewidth=2, linestyle="--",
                     label=f"MPC H=20 ({r20['t_capture']/3600:.2f} h)")
    axes[1].axhline(0.1, color="gray", linestyle=":", alpha=0.5)
    axes[1].set_xlabel("时间 (轨道周期)")
    axes[1].set_ylabel("相对距离 (km)")
    axes[1].set_title("距离收敛轨迹对比")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.4)

    # 右: 推力幅值对比
    axes[2].set_yscale("log")
    for r, label, c, ls in [
        (hypergame_results[0], "Nash", "black", "-"),
        (best_r, f"Hypergame γ_p={best_r['gamma_p']:.0f}", "tab:red", "-"),
    ]:
        t_h = np.arange(len(r["u_p_norm"])) * DT / T_orb
        axes[2].plot(t_h, r["u_p_norm"], color=c, linestyle=ls, linewidth=1.5,
                     label=f"{label} u_p")
        axes[2].plot(t_h, r["u_e_norm"], color=c, linestyle=":", linewidth=1,
                     label=f"{label} u_e")
    axes[2].set_xlabel("时间 (轨道周期)")
    axes[2].set_ylabel("推力幅值 (km/s²)")
    axes[2].set_title("推力图谱对比 (实线=追方, 虚线=逃方)")
    axes[2].legend(fontsize=6)
    axes[2].grid(True, alpha=0.4)

    plt.suptitle("SDRE 追逃博弈性能极限探索", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = Path("outputs/figures/performance_limit.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n图已保存: {out_path}")

    # ─── 保存数据 ───────────────────────────────────────────────────────
    import csv
    data_dir = Path("outputs/data/performance_limit")
    data_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "gamma_p", "captured", "capture_time_h", "final_dist_km"])
        for r in hypergame_results:
            writer.writerow([
                f"hypergame_gamma_{r['gamma_p']}", r["gamma_p"],
                r["captured"],
                f"{r['t_capture']/3600:.4f}" if r["captured"] else "N/A",
                f"{r['final_dist']:.4f}",
            ])
        for H, r in mpc_results.items():
            writer.writerow([
                f"mpc_H{H}", "N/A", r["captured"],
                f"{r['t_capture']/3600:.4f}" if r["captured"] else "N/A",
                f"{r['final_dist']:.4f}",
            ])
    print(f"数据已保存: {data_dir}/summary.csv")


if __name__ == "__main__":
    main()
