"""
实验：为何"仅测角 EKF"反而比"距离+角度 EKF"捕获更快？

假设：仅测角时距离方向不可观测，EKF 估计沿视线方向产生系统偏差，
      导致 SDRE 控制器看到的相对状态与真值不同，从而改变追捕策略。

实验设计：
  1. 多种子 Monte Carlo (N_SEED=20)，对比 angle_only / range_angle 的：
     - 捕获时间分布
     - 末端距离分布
     - 控制能量 (∫||u_p||² dt) 分布
  2. 单次详细诊断：
     - 估计距离 vs 真实距离
     - 估计速度模 vs 真实速度模
     - 距离估计误差随时间
     - 视线方向 (LOS) 上的偏差分量 vs 横向分量
     - 控制加速度大小 ||u_p|| 随时间

以 scenario_1（同平面同轨道）为基准。
"""
import zhplot
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from aerospace.dynamics.nerm import OrbitalDynamics
from aerospace.control.sdre import SDREGameController
from aerospace.estimation.ekf import RelativeStateEKF
from aerospace.simulation.nerm_ekf_sdre import EKFSDRESimulation

from run_scenarios import (
    SCENARIOS, MU, DEG2RAD, SIGMA_ANG, SIGMA_DIST, SIGMA_RANGE,
    REF_DIST, REF_SIGMA_POS, REF_SIGMA_VEL, GAMMA,
    eci_to_lvlh_scenario, create_ekf,
)

N_SEED = 20
SCENARIO_KEY = "scenario_1"
OUT_DIR = Path("outputs/figures/experiment_angle_vs_range")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_one(cfg: dict, mode: str, seed: int):
    """运行一次仿真，返回 result"""
    X_p0, X_e0, nu0 = eci_to_lvlh_scenario(cfg)
    x_rel0 = X_p0 - X_e0
    initial_dist = float(np.linalg.norm(x_rel0[:3]))

    orb = OrbitalDynamics(mu=MU, a_c=cfg["chief_orbit"]["a"], e_c=cfg["chief_orbit"]["e"])
    ctrl = SDREGameController(Q=np.eye(6), R=np.eye(3) * 1e13, gamma=cfg.get("gamma", GAMMA))

    rng_init = np.random.default_rng(seed)
    scale = initial_dist / REF_DIST
    noise = rng_init.standard_normal(6) * np.array([REF_SIGMA_POS * scale] * 3 + [REF_SIGMA_VEL * scale] * 3)
    x0_est = x_rel0 + noise

    ekf = create_ekf(mode=mode, x0=x0_est, initial_dist=initial_dist)
    rng = np.random.default_rng(seed + 1000)

    sim = EKFSDRESimulation(
        dynamics=orb, controller=ctrl, ekf=ekf,
        X_p0=X_p0, X_e0=X_e0, nu0=nu0,
        dt=10.0, are_interval=1, rng=rng,
    )
    result = sim.run(t_end=5.0 * orb.T_orbit)
    return result, orb


def metrics(result, dt=10.0):
    """从 result 提取指标"""
    captured = bool(result.captured)
    if captured:
        # 第一个达到捕获阈值的时刻
        capture_idx = np.argmax(result.dist_history < 0.1)
        t_capture = result.t[capture_idx]
    else:
        t_capture = np.nan
    final_dist = float(result.dist_history[-1])
    # 控制能量 ∫||u_p||² dt（梯形）
    u_norm_sq = np.sum(result.u_p_history**2, axis=0)
    energy = float(np.trapezoid(u_norm_sq, result.t))
    return {
        "captured": captured,
        "t_capture": t_capture,
        "final_dist": final_dist,
        "energy": energy,
    }


def monte_carlo(cfg: dict):
    """多种子 Monte Carlo"""
    results = {"angle_only": [], "range_angle": []}
    for seed in range(N_SEED):
        for mode in results.keys():
            r, _ = run_one(cfg, mode, seed)
            results[mode].append(metrics(r))
            print(f"  seed={seed:2d} mode={mode:12s} captured={r.captured} t_cap={metrics(r)['t_capture']:.0f}s energy={metrics(r)['energy']:.2e}")
    return results


def plot_monte_carlo(mc, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    modes = ["angle_only", "range_angle"]
    labels = ["仅测角", "距离+角度"]
    colors = ["tab:orange", "tab:green"]

    # 捕获时间
    ax = axes[0]
    for mode, label, c in zip(modes, labels, colors):
        ts = [m["t_capture"] / 3600 for m in mc[mode] if not np.isnan(m["t_capture"])]
        if ts:
            ax.hist(ts, bins=10, alpha=0.5, label=f"{label} (n={len(ts)})", color=c)
    ax.set_xlabel("捕获时间 (h)")
    ax.set_ylabel("频次")
    ax.set_title(f"捕获时间分布 ({N_SEED} 种子)")
    ax.legend(); ax.grid(True)

    # 末端距离
    ax = axes[1]
    for mode, label, c in zip(modes, labels, colors):
        ds = [m["final_dist"] for m in mc[mode]]
        ax.hist(ds, bins=10, alpha=0.5, label=label, color=c)
    ax.set_xlabel("末端相对距离 (km)")
    ax.set_ylabel("频次")
    ax.set_title("末端距离分布")
    ax.set_yscale("log")
    ax.legend(); ax.grid(True)

    # 控制能量
    ax = axes[2]
    for mode, label, c in zip(modes, labels, colors):
        es = [m["energy"] for m in mc[mode]]
        ax.hist(es, bins=10, alpha=0.5, label=label, color=c)
    ax.set_xlabel("控制能量 ∫||u_p||² dt")
    ax.set_ylabel("频次")
    ax.set_title("控制能量分布")
    ax.legend(); ax.grid(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Monte Carlo 图已保存: {out_path}")


def diagnose_single(cfg: dict, seed: int = 0):
    """单次详细诊断：分解视线方向和横向误差"""
    r_a, orb = run_one(cfg, "angle_only", seed)
    r_r, _ = run_one(cfg, "range_angle", seed)

    T = orb.T_orbit
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))

    for col, (r, label, c) in enumerate([(r_a, "仅测角", "tab:orange"), (r_r, "距离+角度", "tab:green")]):
        # 真实相对状态
        x_true = r.states[0:6, :] - r.states[6:12, :]
        # 估计与真值
        est_pos = r.x_est_history[:3, :]
        true_pos = x_true[:3, :]
        est_vel = r.x_est_history[3:6, :]
        true_vel = x_true[3:6, :]

        true_dist = np.linalg.norm(true_pos, axis=0)
        est_dist = np.linalg.norm(est_pos, axis=0)

        # 视线方向 (单位向量)
        los = true_pos / np.maximum(true_dist, 1e-9)
        # 误差向量
        err = est_pos - true_pos
        # 视线分量 (径向)
        err_los = np.sum(err * los, axis=0)
        # 横向分量 (剩余)
        err_lat = np.linalg.norm(err - err_los * los, axis=0)

        # 控制加速度幅值
        u_norm = np.linalg.norm(r.u_p_history, axis=0)

        # 第一行：估计距离 vs 真实距离
        ax = axes[0, col]
        ax.plot(r.t / T, true_dist, label="真实", color="k")
        ax.plot(r.t / T, est_dist, label="估计", color=c, linestyle="--")
        ax.set_xlabel("时间 (轨道周期)")
        ax.set_ylabel("相对距离 (km)")
        ax.set_title(f"{label}：估计距离 vs 真实距离")
        ax.legend(); ax.grid(True)

        # 第二行：误差分解（视线方向 / 横向）
        ax = axes[1, col]
        ax.plot(r.t / T, err_los, label="视线方向偏差 (径向)", color="tab:red")
        ax.plot(r.t / T, err_lat, label="横向偏差 (切向)", color="tab:blue")
        ax.axhline(0, color="k", linewidth=0.8, linestyle=":")
        ax.set_xlabel("时间 (轨道周期)")
        ax.set_ylabel("位置估计偏差 (km)")
        ax.set_title(f"{label}：偏差按视线分解")
        ax.legend(); ax.grid(True)

        # 第三行：控制加速度大小
        ax = axes[2, col]
        ax.plot(r.t / T, u_norm, color=c)
        ax.set_xlabel("时间 (轨道周期)")
        ax.set_ylabel("‖u_p‖ (km/s²)")
        ax.set_title(f"{label}：追踪星推力大小")
        ax.set_yscale("log")
        ax.grid(True)

    fig.suptitle(f"单次诊断 (seed={seed}) — {cfg['name']}", fontsize=13)
    fig.tight_layout()
    out_path = OUT_DIR / "diagnose_single.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  单次诊断图已保存: {out_path}")
    return r_a, r_r


def plot_summary(mc):
    """打印汇总统计"""
    print("\n" + "="*70)
    print("Monte Carlo 汇总:")
    print("="*70)
    for mode in ["angle_only", "range_angle"]:
        ts = [m["t_capture"] for m in mc[mode] if not np.isnan(m["t_capture"])]
        es = [m["energy"] for m in mc[mode]]
        ds = [m["final_dist"] for m in mc[mode]]
        cap_rate = sum(m["captured"] for m in mc[mode]) / len(mc[mode])
        print(f"\n[{mode}]")
        print(f"  捕获率:        {cap_rate*100:.1f}%")
        if ts:
            print(f"  捕获时间均值:   {np.mean(ts)/3600:.2f} h  (std {np.std(ts)/3600:.2f})")
        print(f"  末端距离均值:   {np.mean(ds):.4f} km  (std {np.std(ds):.4f})")
        print(f"  控制能量均值:   {np.mean(es):.3e}  (std {np.std(es):.3e})")


if __name__ == "__main__":
    cfg = SCENARIOS[SCENARIO_KEY]
    print(f"实验场景: {cfg['name']}")

    # 1. Monte Carlo
    print("\n--- 1. 多种子 Monte Carlo ---")
    mc = monte_carlo(cfg)
    plot_monte_carlo(mc, OUT_DIR / "monte_carlo.png")
    plot_summary(mc)

    # 2. 单次详细诊断
    print("\n--- 2. 单次详细诊断 (seed=0) ---")
    diagnose_single(cfg, seed=0)

    print(f"\n实验完成，输出目录: {OUT_DIR}")
