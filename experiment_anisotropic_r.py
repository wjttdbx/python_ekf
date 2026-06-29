import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.linalg import solve_continuous_are

from aerospace.dynamics.nerm import OrbitalDynamics
from aerospace.control.sdre import SDREGameController
from aerospace.estimation.ekf import RelativeStateEKF
from aerospace.simulation.nerm_ekf_sdre import EKFSDRESimulation

# ─── 常量与基准设置 ──────────────────────────────────────────────────────────
DEG2RAD = np.pi / 180.0
MU = 3.986e5
A_C = 15000.0
E_C = 0.5
DT = 10.0
T_END_FACTOR = 10.0  # 运行 10 个轨道周期以保证捕获

# 初始状态
X_P0 = np.array([500.0, 500.0, 500.0, 0.01, 0.01, 0.01])
X_E0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
NU0 = 0.0

# EKF参数
SIGMA_ANG = 0.008 * DEG2RAD
SIGMA_RANGE = 0.01  # 10m


# ─── 1. 各项异性追方专用控制器定义 ─────────────────────────────────────────────
class CustomPursuerSDREController(SDREGameController):
    """追方使用自定义 R 矩阵，逃方固定使用基准各向同性 R=1e13 求解，以实现不平衡博弈对抗"""
    def __init__(self, Q, get_R_p_func, gamma=np.sqrt(2)):
        # 基类初始化使用 R=1e13 矩阵，主要供计算逃方控制 u_e 使用
        super().__init__(Q=Q, R=np.eye(3) * 1e13, gamma=gamma)
        self.get_R_p_func = get_R_p_func
        self.last_P_p = None

    def compute_control(self, A_SDC: np.ndarray, x_rel: np.ndarray,
                        t: float = None, solve_are: bool = True,
                        x_rel_e: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        # 1. 求解逃方控制 u_e：直接使用基类计算（其使用基准各向同性 R=1e13）
        _, u_e = super().compute_control(A_SDC, x_rel, t, solve_are, x_rel_e)
        
        # 2. 获取追方的 R 矩阵
        R_p = self.get_R_p_func(x_rel)
        R_inv_p = np.linalg.inv(R_p)
        R_eff_inv_p = R_inv_p * (1.0 - self.gamma**(-2))
        R_eff_p = np.linalg.inv(R_eff_inv_p)
        S_p = self.B_p @ R_eff_inv_p @ self.B_p.T
        
        # 3. 求解追方专用的 Riccati 方程 (P_p)
        if solve_are:
            norm_Q = np.linalg.norm(self.Q, "fro")
            norm_S = np.linalg.norm(S_p, "fro")
            alpha = np.sqrt(norm_Q / norm_S)
            Q_bal = self.Q / alpha
            R_bal = R_eff_p / alpha
            try:
                P_bar = solve_continuous_are(A_SDC, self.B_p, Q_bal, R_bal)
                self.last_P_p = alpha * P_bar
            except Exception as e:
                if self.last_P_p is None:
                    self.last_P_p = np.zeros((6, 6))
        
        P_p = self.last_P_p if self.last_P_p is not None else np.zeros((6, 6))
        
        # 4. 计算追方的推力 u_p
        u_p = - R_eff_inv_p @ self.B_p.T @ P_p @ x_rel
        
        return u_p, u_e


# ─── 2. 仿真运行辅助函数 ────────────────────────────────────────────────────────
def run_sim(ctrl_type="baseline", seed=42):
    orb = OrbitalDynamics(mu=MU, a_c=A_C, e_c=E_C)
    
    # 状态初始化
    x0_est = X_P0 - X_E0
    initial_dist = float(np.linalg.norm(x0_est[:3]))
    
    # EKF 估计器 (使用有偏估计/噪声)
    rng = np.random.default_rng(seed)
    R_meas = np.diag([SIGMA_RANGE**2, SIGMA_ANG**2, SIGMA_ANG**2])  # 距离+角度 EKF
    Q_proc = np.diag([5e-4, 5e-4, 5e-4, 5e-8, 5e-8, 5e-8])
    sigma_pos = initial_dist * SIGMA_ANG
    sigma_vel = 1.0 * SIGMA_ANG
    P0 = np.diag([sigma_pos**2]*3 + [sigma_vel**2]*3)
    
    # 引入初始估计误差
    noise = rng.standard_normal(6) * np.array([sigma_pos]*3 + [sigma_vel]*3)
    x0_est_noisy = x0_est + noise
    ekf = RelativeStateEKF(x0=x0_est_noisy, P0=P0, Q=Q_proc, R=R_meas, angles_only=False)
    
    # 控制器选择
    Q_ctrl = np.eye(6)
    if ctrl_type == "baseline":
        # 各向同性基准: R = 1e13 * I
        ctrl = SDREGameController(Q=Q_ctrl, R=np.eye(3) * 1e13, gamma=np.sqrt(2))
    elif ctrl_type == "anisotropic_los":
        # 视线自适应各向异性（沿视线方向 R=2e12 鼓励逼近，垂直视线方向 R=1e13 抑制横向修正）
        def get_R_los(x_rel):
            r_vec = x_rel[:3]
            dist = np.linalg.norm(r_vec)
            n = r_vec / dist if dist > 1e-6 else np.array([1.0, 0.0, 0.0])
            n = n.reshape(3, 1)
            R_para = 2e12
            R_perp = 1e13
            return R_perp * np.eye(3) + (R_para - R_perp) * (n @ n.T)
        
        ctrl = CustomPursuerSDREController(Q=Q_ctrl, get_R_p_func=get_R_los, gamma=np.sqrt(2))
    elif ctrl_type == "anisotropic_axes":
        # 静态沿迹高惩罚: R = diag(1e12, 1e14, 1e12) (抑制沿轨纠正以利用自然漂移，允许径向冲锋)
        def get_R_axes(x_rel):
            return np.diag([1e12, 1e14, 1e12])
        
        ctrl = CustomPursuerSDREController(Q=Q_ctrl, get_R_p_func=get_R_axes, gamma=np.sqrt(2))
    else:
        raise ValueError(f"Unknown controller type: {ctrl_type}")

    sim = EKFSDRESimulation(
        dynamics=orb, controller=ctrl, ekf=ekf,
        X_p0=X_P0, X_e0=X_E0, nu0=NU0,
        dt=DT, are_interval=1, rng=rng
    )
    
    result = sim.run(t_end=T_END_FACTOR * orb.T_orbit)
    
    # 提取指标
    cap_idx = np.argmax(result.dist_history < 0.1) if result.captured else -1
    t_cap = result.t[cap_idx] if result.captured else np.nan
    energy = float(np.trapezoid(np.sum(result.u_p_history**2, axis=0), result.t))
    
    return result, t_cap, energy, orb


# ─── 3. 主程序与对比 ──────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("各向异性控制惩罚矩阵 (Anisotropic R) 对比实验 (追逃不对称)")
    print("=" * 70)
    
    results = {}
    cases = {
        "baseline": "基准各向同性 (Isotropic R=1e13)",
        "anisotropic_los": "视线自适应各向异性 (R_para=2e12, R_perp=1e13)",
        "anisotropic_axes": "静态LVLH轴各向异性 (R=diag(1e12, 1e14, 1e12))"
    }
    
    for case_key, label in cases.items():
        print(f"正在运行 {label} ...")
        res, t_cap, energy, orb = run_sim(ctrl_type=case_key, seed=42)
        results[case_key] = {
            "result": res,
            "t_cap": t_cap,
            "energy": energy
        }
        status = f"捕获成功 (时间: {t_cap/3600:.2f} h)" if res.captured else "捕获失败"
        print(f"  -> 结果: {status} | 控制能量 (燃料): {energy:.3e}")
        
    # ── 绘图对比 ──
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    T_orbit = orb.T_orbit
    
    colors = {
        "baseline": "tab:blue",
        "anisotropic_los": "tab:orange",
        "anisotropic_axes": "tab:green"
    }
    
    for case_key, data in results.items():
        res = data["result"]
        t_hours = res.t / T_orbit
        
        # 1. 距离时序
        axes[0].plot(t_hours, res.dist_history, color=colors[case_key], label=cases[case_key], linewidth=1.5)
        
        # 2. 推力幅值时序
        u_p_norm = np.linalg.norm(res.u_p_history, axis=0)
        axes[1].plot(t_hours, u_p_norm, color=colors[case_key], label=cases[case_key], alpha=0.8)
        
    axes[0].axhline(0.1, color="red", linestyle="--", label="捕获阈值 (100 m)")
    axes[0].set_ylabel("相对距离 (km)")
    axes[0].set_yscale("log")
    axes[0].grid(True, which="both", alpha=0.4)
    axes[0].legend()
    axes[0].set_title("各向异性控制惩罚下距离逼近对比 (追逃不对称)")
    
    axes[1].set_ylabel("推力幅值 ||u_p|| (km/s²)")
    axes[1].set_xlabel("时间 (轨道周期)")
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", alpha=0.4)
    axes[1].legend()
    axes[1].set_title("推力幅值对比")
    
    plt.tight_layout()
    out_path = Path("nerm_anisotropic_compare.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n对比图已保存至: {out_path.absolute()}")
    
    # ── 打印总结表 ──
    print("\n" + "="*80)
    print(f"{'控制惩罚策略':<35} | {'捕获时间 (h)':<15} | {'控制能量 (相对比值)':<15}")
    print("-" * 80)
    base_energy = results["baseline"]["energy"]
    for case_key, data in results.items():
        t_str = f"{data['t_cap']/3600:.2f} h" if not np.isnan(data['t_cap']) else "N/A"
        energy_ratio = data['energy'] / base_energy
        print(f"{cases[case_key]:<35} | {t_str:<15} | {energy_ratio:14.2%}")
    print("="*80)


if __name__ == "__main__":
    main()
