"""
实验：不同博弈理论框架对比

四种博弈范式 vs shared_A 基线：
1. shared_A (基线)      — Nash 均衡，双方共用 A_est, γ=√2
2. hypergame_γ          — 追方暗中用 γ_p=1.2（更激进），逃方以为 γ=√2
3. hinf_robust          — H∞ 型，追方把估计误差 w 也当作对抗方
4. belief_adaptive_R    — 信念自适应 R：P_ekf 迹大时减小控制惩罚
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


# ═══════════════════════════════════════════════════════════════════════════════
# 通用仿真引擎（支持多种博弈模式）
# ═══════════════════════════════════════════════════════════════════════════════

class GameTheorySimulation:
    """支持多种博弈理论范式的闭环仿真引擎。"""

    def __init__(self, orb, X_p0, X_e0, nu0, ekf,
                 mode="baseline", gamma_p=None, hinf_theta=None,
                 belief_alpha=0.0, dt=10.0, capture_dist=0.1, seed=42):
        self.orb = orb
        self.ekf = ekf
        self.dt = dt
        self.capture_dist = capture_dist
        self.mode = mode

        # 共同参数
        self.Q = np.eye(6)
        self.R_base = np.eye(3) * 1e13
        self.gamma_e = np.sqrt(2)  # 逃方的 γ
        self.gamma_p = gamma_p if gamma_p is not None else np.sqrt(2)
        self.hinf_theta = hinf_theta
        self.belief_alpha = belief_alpha
        self.P0_trace = np.trace(ekf.P)

        self.B_p = np.zeros((6, 3))
        self.B_p[3:, :] = np.eye(3)
        self.B_e = -self.B_p
        self._B_ctrl = self.B_p.copy()

        self.state = np.zeros(13)
        self.state[0:6] = X_p0
        self.state[6:12] = X_e0
        self.state[12] = nu0

        self.rng = np.random.default_rng(seed)
        self.last_P_p = None

    def _solve_are(self, A, R_eff):
        """求解 ARE: A'P + PA - P S P + Q = 0, S = B_p R_eff^{-1} B_p'."""
        S = self.B_p @ np.linalg.inv(R_eff) @ self.B_p.T
        norm_Q = np.linalg.norm(self.Q, "fro")
        norm_S = np.linalg.norm(S, "fro")
        if norm_S < 1e-30:
            return np.zeros((6, 6))
        alpha = np.sqrt(norm_Q / norm_S)
        Q_bal = self.Q / alpha
        R_bal = R_eff / alpha
        S_bal = S / alpha
        try:
            P_bar = solve_continuous_are(A, self.B_p, Q_bal, R_bal)
            return alpha * P_bar
        except Exception:
            return None

    def run(self, t_end):
        N = int(t_end / self.dt)
        st = self.state.copy()
        t = 0.0

        dist_hist = np.zeros(N + 1)
        u_p_norm_hist = np.zeros(N + 1)
        u_e_norm_hist = np.zeros(N + 1)
        P_trace_hist = np.zeros(N + 1)
        R_scale_hist = np.zeros(N + 1)

        captured = False
        N_actual = N

        for k in range(N):
            nu = st[12]
            r_c, nu_dot, nu_ddot = self.orb.get_orbital_params(nu)

            X_p_true = st[0:6]
            X_e_true = st[6:12]
            x_true_rel = X_p_true - X_e_true
            x_est = self.ekf.x
            X_e_est = X_p_true - x_est

            A_est = self.orb.get_SDC_matrix(X_p_true, X_e_est, r_c, nu_dot, nu_ddot)
            A_true = self.orb.get_SDC_matrix(X_p_true, X_e_true, r_c, nu_dot, nu_ddot)

            P_trace = np.trace(self.ekf.P)
            P_trace_hist[k] = P_trace

            # ── 根据博弈模式计算控制 ──────────────────────────────────────

            if self.mode == "baseline":
                # 标准 Nash: 双方都用 γ=√2，共用 A_est
                R_eff = self.R_base / (1.0 - self.gamma_e**(-2))
                R_scale_hist[k] = 1.0

                P = self._solve_are(A_est, R_eff)
                if P is not None:
                    self.last_P_p = P
                elif self.last_P_p is not None:
                    P = self.last_P_p
                else:
                    P = np.zeros((6, 6))

                R_eff_inv = np.linalg.inv(R_eff)
                u_p = -R_eff_inv @ self.B_p.T @ P @ x_est
                u_e = self.gamma_e**(-2) * R_eff_inv @ self.B_e.T @ P @ x_true_rel

            elif self.mode == "hypergame_γ":
                # Hypergame: 逃方按 γ_e=√2 出牌，追方暗中用 γ_p < γ_e
                R_eff_e = self.R_base / (1.0 - self.gamma_e**(-2))
                R_eff_p = self.R_base / (1.0 - self.gamma_p**(-2))
                R_scale_hist[k] = np.trace(R_eff_e) / np.trace(R_eff_p)

                # 追方用自己的 R_eff_p 解 P_p
                P_p = self._solve_are(A_est, R_eff_p)
                if P_p is not None:
                    self.last_P_p = P_p
                elif self.last_P_p is not None:
                    P_p = self.last_P_p
                else:
                    P_p = np.zeros((6, 6))

                # 逃方的 ARE 也是用 A_est 解（它也用追方的估计视角）
                P_e = self._solve_are(A_est, R_eff_e)
                if P_e is None:
                    P_e = np.zeros((6, 6))

                R_eff_inv_p = np.linalg.inv(R_eff_p)
                R_eff_inv_e = np.linalg.inv(R_eff_e)

                u_p = -R_eff_inv_p @ self.B_p.T @ P_p @ x_est
                u_e = self.gamma_e**(-2) * R_eff_inv_e @ self.B_e.T @ P_e @ x_true_rel

            elif self.mode == "hinf_robust":
                # H∞ 型: 估计误差协方差注入 ARE，使追方更保守
                R_eff_base = self.R_base / (1.0 - self.gamma_e**(-2))
                R_eff_inv_base = np.linalg.inv(R_eff_base)

                # 标准 ARE → P_std
                P_std = self._solve_are(A_est, R_eff_base)
                if P_std is None:
                    P_std = np.zeros((6, 6))

                # 从 EKF 协方差提取位置不确定性 σ
                sigma_pos = np.sqrt(np.maximum(np.diag(self.ekf.P)[:3], 1e-12))
                # 用 3σ 对估计状态做 worst-case 偏置
                K_p_pos = R_eff_inv_base @ self.B_p.T @ P_std  # (3,6)
                K_pos = K_p_pos[:, :3]  # 只看位置分量 (3,3)
                worst_dir = np.sign(K_pos @ sigma_pos)  # (3,)
                delta_x = np.zeros(6)
                delta_x[:3] = sigma_pos * worst_dir

                x_worst = x_est + delta_x
                u_p = -R_eff_inv_base @ self.B_p.T @ P_std @ x_worst
                u_e = self.gamma_e**(-2) * R_eff_inv_base @ self.B_e.T @ P_std @ x_true_rel
                R_scale_hist[k] = np.mean(sigma_pos) / (np.mean(np.abs(x_est[:3])) + 1e-6)

            elif self.mode == "belief_adaptive_R":
                # 信念自适应 R
                trace_ratio = P_trace / (self.P0_trace + 1e-30)
                # 不确定性高 → R 缩小 → 控制更激进
                scale = np.clip(trace_ratio ** self.belief_alpha, 0.1, 5.0)
                R_adapted = self.R_base * scale
                R_eff = R_adapted / (1.0 - self.gamma_e**(-2))
                R_scale_hist[k] = scale

                P = self._solve_are(A_est, R_eff)
                if P is not None:
                    self.last_P_p = P
                elif self.last_P_p is not None:
                    P = self.last_P_p
                else:
                    P = np.zeros((6, 6))

                R_eff_inv = np.linalg.inv(R_eff)
                u_p = -R_eff_inv @ self.B_p.T @ P @ x_est
                u_e = self.gamma_e**(-2) * R_eff_inv @ self.B_e.T @ P @ x_true_rel

            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            # ── 真实状态传播 ──────────────────────────────────────────────
            sol = solve_ivp(
                self.orb.dynamics_13d, [t, t + self.dt], st,
                args=(u_p, u_e), method="RK45", rtol=1e-8, atol=1e-10,
            )
            st = sol.y[:, -1]
            t += self.dt

            # ── EKF 预测 & 更新 ───────────────────────────────────────────
            x_priori, P_priori = self.ekf.predict(A_est, self._B_ctrl, u_p, u_e, self.dt)
            z_true = RelativeStateEKF.measure(st[0:6], st[6:12], angle_only=True)
            z_meas = z_true + self.rng.multivariate_normal(np.zeros(2), self.ekf.R)
            self.ekf.update(x_priori, P_priori, z_meas)

            dist_hist[k + 1] = np.linalg.norm(st[0:3] - st[6:9])
            u_p_norm_hist[k + 1] = np.linalg.norm(u_p)
            u_e_norm_hist[k + 1] = np.linalg.norm(u_e)

            if dist_hist[k + 1] < self.capture_dist:
                print(f"  [{self.mode}] 捕获！t = {t:.1f} s ({t/3600:.2f} h)")
                N_actual = k + 1
                captured = True
                break
        else:
            print(f"  [{self.mode}] 未捕获，最终距离 = {dist_hist[N]:.3f} km")

        sl = slice(0, N_actual + 1)
        return {
            "captured": captured,
            "t_capture": t if captured else np.nan,
            "dist": dist_hist[sl],
            "u_p_norm": u_p_norm_hist[sl],
            "u_e_norm": u_e_norm_hist[sl],
            "P_trace": P_trace_hist[sl],
            "R_scale": R_scale_hist[sl],
            "final_dist": dist_hist[N_actual],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    orb = OrbitalDynamics(mu=MU, a_c=15000.0, e_c=0.5)
    X_p0 = np.array([500.0, 500.0, 500.0, 0.01, 0.01, 0.01])
    X_e0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    nu0 = 0.0
    x_rel0 = X_p0 - X_e0

    SIGMA_ANG = 0.008 * DEG2RAD
    seed = 42
    rng = np.random.default_rng(seed)

    R_meas_ao = np.diag([SIGMA_ANG**2, SIGMA_ANG**2])
    Q_proc = np.diag([5e-4, 5e-4, 5e-4, 5e-8, 5e-8, 5e-8])

    initial_dist = float(np.linalg.norm(x_rel0[:3]))
    scale_d = initial_dist / 3000.0
    sigma_pos = 10.0 * scale_d
    sigma_vel = 1e-3 * scale_d
    noise = rng.standard_normal(6) * np.array([sigma_pos]*3 + [sigma_vel]*3)

    t_end = 5.0 * orb.T_orbit
    print(f"初始相对距离: {initial_dist:.1f} km,  轨道周期: {orb.T_orbit/3600:.2f} h\n")

    # ─── 各模式参数字典 ──────────────────────────────────────────────────
    mode_configs = {
        "baseline": {
            "label": "Nash 基线 (γ=√2, 共用A)",
            "mode": "baseline",
        },
        "hypergame_2.0": {
            "label": "Hypergame (γ_p=2.0, γ_e=√2)",
            "mode": "hypergame_γ",
            "gamma_p": 2.0,
        },
        "hypergame_3.0": {
            "label": "Hypergame (γ_p=3.0, γ_e=√2)",
            "mode": "hypergame_γ",
            "gamma_p": 3.0,
        },
        "hinf_robust": {
            "label": "H∞ 鲁棒 (θ=1e3, 3σ worst-case)",
            "mode": "hinf_robust",
            "hinf_theta": 1e3,
        },
        "belief_R_0.5": {
            "label": "信念自适应R (α=0.5, 不确定时更激进)",
            "mode": "belief_adaptive_R",
            "belief_alpha": 0.5,
        },
        "belief_R_1.0": {
            "label": "信念自适应R (α=1.0, 线性缩放)",
            "mode": "belief_adaptive_R",
            "belief_alpha": 1.0,
        },
    }

    results = {}
    for key, cfg in mode_configs.items():
        print("=" * 65)
        print(f"模式: {cfg['label']}")
        print("=" * 65)

        x0_est = x_rel0 + noise
        P0 = np.diag([sigma_pos**2]*3 + [sigma_vel**2]*3)
        ekf = RelativeStateEKF(x0=x0_est.copy(), P0=P0.copy(),
                               Q=Q_proc, R=R_meas_ao, angles_only=True)

        sim = GameTheorySimulation(
            orb, X_p0, X_e0, nu0, ekf,
            mode=cfg["mode"],
            gamma_p=cfg.get("gamma_p"),
            hinf_theta=cfg.get("hinf_theta"),
            belief_alpha=cfg.get("belief_alpha", 0.0),
            dt=DT, seed=seed+1,
        )
        results[key] = sim.run(t_end)

    # ─── 汇总 ────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"{'博弈范式':<30} {'捕获':<8} {'捕获时间':<15} {'vs基线':<10}")
    print("-" * 65)
    base_time = results["baseline"]["t_capture"]
    for key, cfg in mode_configs.items():
        r = results[key]
        if r["captured"]:
            delta = (r["t_capture"] - base_time) / base_time * 100
            delta_str = f"{delta:+.1f}%" if key != "baseline" else "—"
            print(f"{cfg['label']:<30} ✅       {r['t_capture']/3600:.2f} h         {delta_str}")
        else:
            print(f"{cfg['label']:<30} ❌       N/A              —")
    print("=" * 65)

    # ─── 绘图 ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    T_orb = orb.T_orbit

    colors = ["tab:blue", "tab:red", "tab:orange", "tab:purple", "tab:green", "tab:brown"]
    styles = ["-", "--", "--", "-.", "-", "-."]

    for (key, cfg), c, ls in zip(mode_configs.items(), colors, styles):
        r = results[key]
        t_h = np.arange(len(r["dist"])) * DT / T_orb
        axes[0, 0].plot(t_h, r["dist"], color=c, linestyle=ls, linewidth=1.5,
                         label=cfg["label"])
        axes[0, 1].plot(t_h, r["u_p_norm"], color=c, linestyle=ls, alpha=0.8,
                         label=cfg["label"])
        axes[1, 0].plot(t_h, r["u_e_norm"], color=c, linestyle=ls, alpha=0.8,
                         label=cfg["label"])

    axes[0, 0].set_ylabel("相对距离 (km)")
    axes[0, 0].set_title("距离收敛对比")
    axes[0, 0].legend(fontsize=7, loc="upper right")
    axes[0, 0].grid(True, alpha=0.4)
    axes[0, 0].set_yscale("log")
    axes[0, 0].axhline(0.1, color="gray", linestyle=":", alpha=0.5)

    axes[0, 1].set_ylabel("||u_p|| (km/s²)")
    axes[0, 1].set_title("追方推力幅值")
    axes[0, 1].legend(fontsize=7, loc="upper right")
    axes[0, 1].grid(True, alpha=0.4)
    axes[0, 1].set_yscale("log")

    axes[1, 0].set_ylabel("||u_e|| (km/s²)")
    axes[1, 0].set_title("逃方推力幅值")
    axes[1, 0].set_xlabel("时间 (轨道周期)")
    axes[1, 0].legend(fontsize=7, loc="upper right")
    axes[1, 0].grid(True, alpha=0.4)
    axes[1, 0].set_yscale("log")

    # 右下：信念自适应 R 的缩放因子
    for key in ["belief_R_0.5", "belief_R_1.0"]:
        if key in results:
            r = results[key]
            t_h = np.arange(len(r["R_scale"])) * DT / T_orb
            axes[1, 1].plot(t_h, r["R_scale"], linewidth=1.5,
                            label=mode_configs[key]["label"].split("(")[0].strip())
    axes[1, 1].set_ylabel("R 缩放因子")
    axes[1, 1].set_title("信念自适应 R 缩放因子")
    axes[1, 1].set_xlabel("时间 (轨道周期)")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.4)
    axes[1, 1].axhline(1.0, color="gray", linestyle=":", alpha=0.5)

    plt.suptitle("不同博弈理论框架的追逃性能对比 (仅测角EKF)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = Path("outputs/figures/game_theory_comparison.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n对比图已保存: {out_path}")

    # ─── 保存数据 ────────────────────────────────────────────────────────
    import csv
    data_dir = Path("outputs/data/game_theory_comparison")
    data_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "label", "captured", "capture_time_h", "final_dist_km"])
        for key, cfg in mode_configs.items():
            r = results[key]
            writer.writerow([
                key, cfg["label"], r["captured"],
                f"{r['t_capture']/3600:.4f}" if r["captured"] else "N/A",
                f"{r['final_dist']:.4f}",
            ])

    for key, r in results.items():
        n = len(r["dist"])
        stride = max(1, n // 2000)
        idx = np.arange(0, n, stride)
        t_h = np.arange(n) * DT / T_orb
        rows = [[t_h[i], r["dist"][i], r["u_p_norm"][i], r["u_e_norm"][i]]
                for i in idx]
        np.savetxt(data_dir / f"{key}.csv", np.array(rows),
                   delimiter=",", header="time_orbits,dist_km,u_p_norm,u_e_norm",
                   comments="", fmt=["%.6f"] + ["%.8e"]*3)
    print(f"数据已保存: {data_dir}/")


if __name__ == "__main__":
    main()
