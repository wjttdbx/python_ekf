"""
实验：追逃双方是否可以使用不同的 A 矩阵？

对比三种模式：
1. shared_A (现状):   双方共用 A_SDC(X_p_true, X_e_est)，同一个 P
2. dual_A:            追方用 A_SDC(X_p_true, X_e_est) → P_p，逃方用 A_SDC(X_p_true, X_e_true) → P_e
3. omniscient (全知): 无 EKF，双方都用真实 A_SDC + 真实状态（上界）

问题：dual_A 打破了博弈的共同认知假设，在工程上还能追到吗？
"""
import zhplot
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.linalg import solve_continuous_are

from aerospace.dynamics.nerm import OrbitalDynamics
from aerospace.control.sdre import SDREGameController
from aerospace.estimation.ekf import RelativeStateEKF
from aerospace.simulation.nerm_ekf_sdre import EKFSDRESimulation

DEG2RAD = np.pi / 180.0
MU = 3.986e5
DT = 10.0

# ─── 双 A 矩阵控制器 ──────────────────────────────────────────────────────────
class DualAMatrixController:
    """追方和逃方各自使用独立的 A_SDC 求解 ARE，得到各自的 P 矩阵。"""

    def __init__(self, Q, R, gamma=np.sqrt(2)):
        self.Q = Q
        self.R = R
        self.gamma = gamma
        self.R_inv = np.linalg.inv(R)
        self.R_eff_inv = self.R_inv * (1.0 - gamma**(-2))
        self.R_eff = R / (1.0 - gamma**(-2))
        self.B_p = np.zeros((6, 3))
        self.B_p[3:, :] = np.eye(3)
        self.B_e = -self.B_p
        self.S = self.B_p @ self.R_eff_inv @ self.B_p.T

        self.last_P_p = None  # 追方的 P
        self.last_P_e = None  # 逃方的 P
        self._fallback_count = 0

    def _solve_are_for_A(self, A):
        """为给定的 A 矩阵求解 ARE，返回 P。"""
        norm_Q = np.linalg.norm(self.Q, "fro")
        norm_S = np.linalg.norm(self.S, "fro")
        if norm_S < 1e-30:
            return np.zeros((6, 6))
        alpha = np.sqrt(norm_Q / norm_S)
        Q_bal = self.Q / alpha
        R_bal = self.R_eff / alpha
        try:
            P_bar = solve_continuous_are(A, self.B_p, Q_bal, R_bal)
            return alpha * P_bar
        except Exception:
            return None

    def compute_control(self, A_p: np.ndarray, A_e: np.ndarray,
                        x_rel: np.ndarray, x_rel_e: np.ndarray,
                        solve_are: bool = True):
        """使用两个不同的 A 矩阵分别计算追方和逃方控制。

        A_p: 追方视角的 SDC 矩阵（基于估计状态）
        A_e: 逃方视角的 SDC 矩阵（基于真实状态）
        """
        if solve_are:
            # 追方的 P
            P_p = self._solve_are_for_A(A_p)
            if P_p is not None:
                self.last_P_p = P_p
            elif self.last_P_p is not None:
                P_p = self.last_P_p
                self._fallback_count += 1
            else:
                P_p = np.zeros((6, 6))

            # 逃方的 P
            P_e = self._solve_are_for_A(A_e)
            if P_e is not None:
                self.last_P_e = P_e
            elif self.last_P_e is not None:
                P_e = self.last_P_e
            else:
                P_e = np.zeros((6, 6))
        else:
            P_p = self.last_P_p
            P_e = self.last_P_e

        u_p = -self.R_eff_inv @ self.B_p.T @ P_p @ x_rel
        u_e = self.gamma**(-2) * self.R_eff_inv @ self.B_e.T @ P_e @ x_rel_e

        return u_p, u_e


# ─── 带双 A 矩阵的仿真引擎 ───────────────────────────────────────────────────
class DualASimulation:
    """支持双 A 矩阵的闭环仿真。"""

    def __init__(self, dynamics, controller, ekf, X_p0, X_e0, nu0,
                 dt=10.0, rng=None, mode="shared_A"):
        self.dynamics = dynamics
        self.controller = controller
        self.ekf = ekf
        self.dt = dt
        self.rng = rng
        self.mode = mode  # "shared_A" | "dual_A" | "omniscient"

        self.state = np.zeros(13)
        self.state[0:6] = X_p0
        self.state[6:12] = X_e0
        self.state[12] = nu0

        self._B_ctrl = np.zeros((6, 3))
        self._B_ctrl[3:, :] = np.eye(3)

    def run(self, t_end):
        N = int(t_end / self.dt)
        state = self.state.copy()
        t = 0.0

        dist_hist = np.zeros(N + 1)
        u_p_norm_hist = np.zeros(N + 1)
        u_e_norm_hist = np.zeros(N + 1)
        P_diff_hist = np.zeros(N + 1)  # ||P_p - P_e|| in dual mode

        captured = False
        N_actual = N

        for k in range(N):
            nu = state[12]
            r_c, nu_dot, nu_ddot = self.dynamics.get_orbital_params(nu)

            X_p_true = state[0:6]
            X_e_true = state[6:12]
            x_true_rel = X_p_true - X_e_true

            A_true = self.dynamics.get_SDC_matrix(X_p_true, X_e_true, r_c, nu_dot, nu_ddot)

            if self.mode == "omniscient":
                # 全知：无 EKF，真实 A + 真实状态
                x_ctrl = x_true_rel
                self.ekf.x = x_true_rel.copy()

                if isinstance(self.controller, DualAMatrixController):
                    u_p, u_e = self.controller.compute_control(
                        A_true, A_true, x_ctrl, x_true_rel
                    )
                else:
                    u_p, u_e = self.controller.compute_control(A_true, x_ctrl, x_rel_e=x_true_rel)

                P_diff_hist[k] = 0.0

            elif self.mode == "true_A":
                # EKF 估计状态 + 真实 A 矩阵（双方都用 A_true → 同一个 P）
                x_ctrl = self.ekf.x

                if isinstance(self.controller, DualAMatrixController):
                    u_p, u_e = self.controller.compute_control(
                        A_true, A_true, x_ctrl, x_true_rel
                    )
                else:
                    u_p, u_e = self.controller.compute_control(A_true, x_ctrl, x_rel_e=x_true_rel)

                P_diff_hist[k] = 0.0

            else:
                # EKF 估计状态 + 估计 A 矩阵
                x_ctrl = self.ekf.x
                X_e_est = X_p_true - x_ctrl
                A_est = self.dynamics.get_SDC_matrix(X_p_true, X_e_est, r_c, nu_dot, nu_ddot)

                if self.mode == "shared_A":
                    # 双方共用 A_est
                    if isinstance(self.controller, DualAMatrixController):
                        u_p, u_e = self.controller.compute_control(
                            A_est, A_est, x_ctrl, x_true_rel
                        )
                    else:
                        u_p, u_e = self.controller.compute_control(
                            A_est, x_ctrl, x_rel_e=x_true_rel
                        )
                    P_diff_hist[k] = 0.0

                elif self.mode == "dual_A":
                    # 追方用 A_est→P_p，逃方用 A_true→P_e
                    u_p, u_e = self.controller.compute_control(
                        A_est, A_true, x_ctrl, x_true_rel
                    )

                    if (self.controller.last_P_p is not None and
                        self.controller.last_P_e is not None):
                        P_diff_hist[k] = np.linalg.norm(
                            self.controller.last_P_p - self.controller.last_P_e, 'fro'
                        )

            # 真实状态传播
            from scipy.integrate import solve_ivp
            sol = solve_ivp(
                self.dynamics.dynamics_13d, [t, t + self.dt], state,
                args=(u_p, u_e), method="RK45", rtol=1e-8, atol=1e-10,
            )
            state = sol.y[:, -1]
            t += self.dt

            # EKF 更新
            if self.rng is not None and self.mode != "omniscient":
                # 所有非全知模式都使用 A_est 做 EKF 预测
                # （true_A 模式虽然控制用 A_true，但 EKF 预测仍用 A_est）
                X_e_est = X_p_true - self.ekf.x
                A_est_for_ekf = self.dynamics.get_SDC_matrix(X_p_true, X_e_est, r_c, nu_dot, nu_ddot)
                x_priori, P_priori = self.ekf.predict(A_est_for_ekf, self._B_ctrl, u_p, u_e, self.dt)
                z_true = RelativeStateEKF.measure(state[0:6], state[6:12],
                                                   angle_only=(self.ekf.R.shape[0] == 2))
                z_meas = z_true + self.rng.multivariate_normal(np.zeros(self.ekf.R.shape[0]), self.ekf.R)
                self.ekf.update(x_priori, P_priori, z_meas)
            elif self.rng is None:
                self.ekf.x = state[0:6] - state[6:12]

            dist_hist[k + 1] = np.linalg.norm(state[0:3] - state[6:9])
            u_p_norm_hist[k + 1] = np.linalg.norm(u_p)
            u_e_norm_hist[k + 1] = np.linalg.norm(u_e)

            if dist_hist[k + 1] < 0.1:
                print(f"  [{self.mode}] 捕获！t = {t:.1f} s ({t/3600:.2f} h)")
                N_actual = k + 1
                captured = True
                break
        else:
            print(f"  [{self.mode}] 未捕获，最终距离 = {dist_hist[N]:.3f} km")

        return {
            "captured": captured,
            "t_capture": t if captured else np.nan,
            "dist": dist_hist[:N_actual+1],
            "u_p_norm": u_p_norm_hist[:N_actual+1],
            "u_e_norm": u_e_norm_hist[:N_actual+1],
            "P_diff": P_diff_hist[:N_actual+1],
            "final_dist": dist_hist[N_actual],
        }


def main():
    # ─── 使用大椭圆轨道 + 大初始距离，放大 EKF 估计误差 ────────────────────
    orb = OrbitalDynamics(mu=MU, a_c=15000.0, e_c=0.5)

    # 初始 LVLH 状态：追方在逃方 500 km 外
    X_p0 = np.array([500.0, 500.0, 500.0, 0.01, 0.01, 0.01])
    X_e0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    nu0 = 0.0

    x_rel0 = X_p0 - X_e0
    initial_dist = float(np.linalg.norm(x_rel0[:3]))
    print(f"初始相对距离: {initial_dist:.1f} km")
    print(f"轨道周期: {orb.T_orbit/3600:.2f} h\n")

    Q_ctrl = np.eye(6)
    R_ctrl = np.eye(3) * 1e13
    gamma = np.sqrt(2)

    # ─── 仅测角 EKF（无距离信息 → 估计误差更大）────────────────────────────
    SIGMA_ANG = 0.008 * DEG2RAD
    seed = 42
    rng = np.random.default_rng(seed)

    R_meas_ao = np.diag([SIGMA_ANG**2, SIGMA_ANG**2])  # 仅测角 2×2
    Q_proc = np.diag([5e-4, 5e-4, 5e-4, 5e-8, 5e-8, 5e-8])

    scale = initial_dist / 3000.0
    sigma_pos = 10.0 * scale
    sigma_vel = 1e-3 * scale
    noise = rng.standard_normal(6) * np.array([sigma_pos]*3 + [sigma_vel]*3)
    x0_est = x_rel0 + noise

    results = {}

    # ─── 模式 0: 全知（上界）─────────────────────────────────────────────────
    print("=" * 60)
    print("模式 0: 全知 (omniscient) — 无 EKF，真实 A + 真实状态")
    print("=" * 60)
    ekf_omni = RelativeStateEKF(
        x0=x_rel0.copy(),
        P0=np.diag([1.0]*3 + [1e-4]*3),
        Q=np.zeros((6, 6)),
        R=np.diag([1e10, 1e-30, 1e-30]),
        angles_only=False,
    )
    ctrl_omni = SDREGameController(Q=Q_ctrl, R=R_ctrl, gamma=gamma)
    sim_omni = DualASimulation(
        orb, ctrl_omni, ekf_omni, X_p0, X_e0, nu0,
        dt=DT, rng=None, mode="omniscient",
    )
    results["omniscient"] = sim_omni.run(t_end=5.0 * orb.T_orbit)

    # ─── 模式 1: 共享 A + 仅测角（现状，信息最贫乏）─────────────────────────
    print("\n" + "=" * 60)
    print("模式 1: shared_A (现状) — 仅测角EKF，共用 A_est→P")
    print("=" * 60)
    ekf_shared = RelativeStateEKF(
        x0=x0_est.copy(),
        P0=np.diag([sigma_pos**2]*3 + [sigma_vel**2]*3),
        Q=Q_proc, R=R_meas_ao, angles_only=True,
    )
    ctrl_shared = DualAMatrixController(Q=Q_ctrl, R=R_ctrl, gamma=gamma)
    sim_shared = DualASimulation(
        orb, ctrl_shared, ekf_shared, X_p0, X_e0, nu0,
        dt=DT, rng=np.random.default_rng(seed+1), mode="shared_A",
    )
    results["shared_A"] = sim_shared.run(t_end=5.0 * orb.T_orbit)

    # ─── 模式 2: 双 A + 仅测角（实验）───────────────────────────────────────
    print("\n" + "=" * 60)
    print("模式 2: dual_A (实验) — 仅测角EKF，追方 A_est→P_p，逃方 A_true→P_e")
    print("=" * 60)
    ekf_dual = RelativeStateEKF(
        x0=x0_est.copy(),
        P0=np.diag([sigma_pos**2]*3 + [sigma_vel**2]*3),
        Q=Q_proc, R=R_meas_ao, angles_only=True,
    )
    ctrl_dual = DualAMatrixController(Q=Q_ctrl, R=R_ctrl, gamma=gamma)
    sim_dual = DualASimulation(
        orb, ctrl_dual, ekf_dual, X_p0, X_e0, nu0,
        dt=DT, rng=np.random.default_rng(seed+1), mode="dual_A",
    )
    results["dual_A"] = sim_dual.run(t_end=5.0 * orb.T_orbit)

    # ─── 模式 3: 全知 A + 仅测角状态（对照）─────────────────────────────────
    # 追方用真实 A_SDC，逃方也用真实 A_SDC，但追方反馈用 EKF 估计状态
    print("\n" + "=" * 60)
    print("模式 3: true_A (对照) — 仅测角EKF，双方都用真实 A_true→P")
    print("=" * 60)
    ekf_trueA = RelativeStateEKF(
        x0=x0_est.copy(),
        P0=np.diag([sigma_pos**2]*3 + [sigma_vel**2]*3),
        Q=Q_proc, R=R_meas_ao, angles_only=True,
    )
    ctrl_trueA = DualAMatrixController(Q=Q_ctrl, R=R_ctrl, gamma=gamma)
    sim_trueA = DualASimulation(
        orb, ctrl_trueA, ekf_trueA, X_p0, X_e0, nu0,
        dt=DT, rng=np.random.default_rng(seed+1), mode="true_A",
    )
    results["true_A"] = sim_trueA.run(t_end=5.0 * orb.T_orbit)

    # ─── 汇总与绘图 ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("结果汇总")
    print("=" * 60)
    for mode, r in results.items():
        status = f"捕获 {r['t_capture']/3600:.2f} h" if r['captured'] else f"未捕获 (最终 {r['final_dist']:.3f} km)"
        print(f"  {mode:<15}: {status}")

    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    T_orb = orb.T_orbit
    colors = {"omniscient": "black", "shared_A": "tab:blue", "dual_A": "tab:red", "true_A": "tab:green"}
    labels = {"omniscient": "全知 (上界)", "shared_A": "共享A (现状)", "dual_A": "双A (实验)",
              "true_A": "真实A+EKF状态 (对照)"}

    for mode, r in results.items():
        t_h = np.arange(len(r["dist"])) * DT / T_orb
        axes[0, 0].plot(t_h, r["dist"], color=colors[mode], label=labels[mode])
        axes[0, 1].plot(t_h, r["u_p_norm"], color=colors[mode], label=labels[mode])
        axes[1, 0].plot(t_h, r["u_e_norm"], color=colors[mode], label=labels[mode])
        if mode == "dual_A":
            axes[1, 1].plot(t_h, r["P_diff"], color=colors[mode], label="||P_p - P_e||_F")

    axes[0, 0].set_ylabel("相对距离 (km)")
    axes[0, 0].set_title("距离收敛对比")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].set_yscale("log")
    axes[0, 0].axhline(0.1, color="gray", linestyle="--", alpha=0.5)

    axes[0, 1].set_ylabel("||u_p|| (km/s²)")
    axes[0, 1].set_title("追方推力幅值")
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].set_yscale("log")

    axes[1, 0].set_ylabel("||u_e|| (km/s²)")
    axes[1, 0].set_title("逃方推力幅值")
    axes[1, 0].set_xlabel("时间 (轨道周期)")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].set_yscale("log")

    axes[1, 1].set_ylabel("Frobenius 范数")
    axes[1, 1].set_title("P 矩阵差异 ||P_p - P_e||_F (仅 dual_A)")
    axes[1, 1].set_xlabel("时间 (轨道周期)")
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].set_yscale("log")

    plt.suptitle("追逃双方使用不同 A 矩阵的影响 (仅测角EKF, 大椭圆轨道)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = Path("outputs/figures/dual_a_matrix_comparison.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n对比图已保存: {out_path}")

    # ─── 保存数值数据为 CSV ────────────────────────────────────────────────────
    import csv
    data_dir = Path("outputs/data/dual_a_matrix")
    data_dir.mkdir(parents=True, exist_ok=True)

    # 保存汇总表
    with open(data_dir / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "captured", "capture_time_h", "final_dist_km", "n_steps"])
        for mode, r in results.items():
            writer.writerow([
                mode, r["captured"],
                f"{r['t_capture']/3600:.4f}" if r["captured"] else "N/A",
                f"{r['final_dist']:.4f}",
                len(r["dist"]),
            ])

    # 保存每个模式的时间序列（均匀降采样到至多 2000 点）
    for mode, r in results.items():
        n = len(r["dist"])
        stride = max(1, n // 2000)
        idx = np.arange(0, n, stride)
        t_h = np.arange(n) * DT / T_orb

        rows = []
        for i in idx:
            row = [t_h[i], r["dist"][i], r["u_p_norm"][i], r["u_e_norm"][i]]
            if mode == "dual_A":
                row.append(r["P_diff"][i])
            rows.append(row)

        header = ["time_orbits", "dist_km", "u_p_norm", "u_e_norm"]
        if mode == "dual_A":
            header.append("P_diff_frobenius")

        np.savetxt(data_dir / f"{mode}.csv", np.array(rows),
                   delimiter=",", header=",".join(header), comments="",
                   fmt=["%.6f"] + ["%.8e"] * (len(header) - 1))

    names = ", ".join(p.name for p in sorted(data_dir.glob("*.csv")))
    print(f"数值数据已保存: {data_dir}/")
    print(f"  {names}")


if __name__ == "__main__":
    main()
