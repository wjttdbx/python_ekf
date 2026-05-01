"""
追加实验：探究"仅测角反而更快+更省能"的根因。

实验组（均基于 scenario_1，每组 10 种子）：
  A. baseline:        当前 angle_only / range_angle 两组
  B. 完美距离测量:     SIGMA_RANGE = 1 mm (1e-6 km) — 排除 10m 距离噪声影响
  C. 极小过程噪声 Q:   Q /= 100 — 检验是否 Q 主导抖动
  D. 极大过程噪声 Q:   Q *= 100 — 反向验证
  E. 共同初始估计噪声:  两模式用相同 x0_est（隔离初值随机性）
  F. 频域：估计距离 / 推力的标准差/抖动幅度

输出：
  - 多组 bar 图汇总（捕获时间、能量）
  - 估计距离时序对比 + 推力时序对比（典型种子）
  - 推力变化率 ‖du/dt‖ 时序（量化抖动）
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
    SCENARIOS, MU, SIGMA_ANG, SIGMA_DIST,
    REF_DIST, REF_SIGMA_POS, REF_SIGMA_VEL, GAMMA, eci_to_lvlh_scenario,
)

N_SEED = 10
SCENARIO_KEY = "scenario_1"
OUT_DIR = Path("outputs/figures/experiment_angle_vs_range/followup")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_ekf(mode: str, x0, init_dist, sigma_range=0.01, q_scale=1.0):
    """灵活构建 EKF：mode='angle_only'|'range_angle'，参数可调"""
    scale = init_dist / REF_DIST
    sp, sv = REF_SIGMA_POS * scale, REF_SIGMA_VEL * scale
    P0 = np.diag([sp**2]*3 + [sv**2]*3)
    Q = np.diag([5e-4, 5e-4, 5e-4, 5e-8, 5e-8, 5e-8]) * q_scale
    if mode == "angle_only":
        R = np.diag([SIGMA_DIST, SIGMA_ANG**2, SIGMA_ANG**2])
    else:
        R = np.diag([sigma_range**2, SIGMA_ANG**2, SIGMA_ANG**2])
    return RelativeStateEKF(x0=x0, P0=P0, Q=Q, R=R)


def run_one(cfg, mode, seed, sigma_range=0.01, q_scale=1.0, share_x0_seed=None):
    X_p0, X_e0, nu0 = eci_to_lvlh_scenario(cfg)
    x_rel0 = X_p0 - X_e0
    init_dist = float(np.linalg.norm(x_rel0[:3]))
    orb = OrbitalDynamics(mu=MU, a_c=cfg["chief_orbit"]["a"], e_c=cfg["chief_orbit"]["e"])
    ctrl = SDREGameController(Q=np.eye(6), R=np.eye(3) * 1e13, gamma=cfg.get("gamma", GAMMA))

    seed_init = share_x0_seed if share_x0_seed is not None else seed
    rng_init = np.random.default_rng(seed_init)
    scale = init_dist / REF_DIST
    noise = rng_init.standard_normal(6) * np.array([REF_SIGMA_POS*scale]*3 + [REF_SIGMA_VEL*scale]*3)
    x0_est = x_rel0 + noise

    ekf = make_ekf(mode, x0_est, init_dist, sigma_range=sigma_range, q_scale=q_scale)
    rng = np.random.default_rng(seed + 1000)
    sim = EKFSDRESimulation(dynamics=orb, controller=ctrl, ekf=ekf,
                            X_p0=X_p0, X_e0=X_e0, nu0=nu0,
                            dt=10.0, are_interval=1, rng=rng)
    return sim.run(t_end=5.0 * orb.T_orbit), orb


def metrics(r):
    cap_idx = np.argmax(r.dist_history < 0.1) if r.captured else -1
    t_cap = r.t[cap_idx] if r.captured else np.nan
    energy = float(np.trapezoid(np.sum(r.u_p_history**2, axis=0), r.t))
    # 推力抖动：‖Δu‖ 的均值
    du = np.diff(r.u_p_history, axis=1)
    jitter = float(np.mean(np.linalg.norm(du, axis=0)))
    return dict(t_cap=t_cap, energy=energy, jitter=jitter, captured=r.captured)


def run_group(cfg, label, **kwargs):
    out = {"angle_only": [], "range_angle": []}
    for seed in range(N_SEED):
        for mode in out:
            r, _ = run_one(cfg, mode, seed, **kwargs)
            out[mode].append(metrics(r))
    print(f"\n[{label}]")
    for mode in out:
        ts = [m["t_cap"]/3600 for m in out[mode] if not np.isnan(m["t_cap"])]
        es = [m["energy"] for m in out[mode]]
        js = [m["jitter"] for m in out[mode]]
        print(f"  {mode:12s}  t_cap={np.mean(ts):.2f}±{np.std(ts):.2f} h  "
              f"E={np.mean(es):.3e}  jitter={np.mean(js):.3e}")
    return out


def plot_groups(groups, out_path):
    """各组 bar 对比"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    labels = list(groups.keys())
    x = np.arange(len(labels))
    w = 0.35

    def vals(metric):
        ao = [np.nanmean([m[metric] for m in groups[l]["angle_only"]]) for l in labels]
        ra = [np.nanmean([m[metric] for m in groups[l]["range_angle"]]) for l in labels]
        return ao, ra

    for ax, metric, ylabel, scale in [
        (axes[0], "t_cap", "捕获时间 (h)", 1/3600),
        (axes[1], "energy", "控制能量 ∫‖u‖² dt", 1.0),
        (axes[2], "jitter", "推力抖动均值 (km/s²)", 1.0),
    ]:
        ao, ra = vals(metric)
        ao = [v * scale for v in ao]
        ra = [v * scale for v in ra]
        ax.bar(x - w/2, ao, w, label="仅测角", color="tab:orange")
        ax.bar(x + w/2, ra, w, label="距离+角度", color="tab:green")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel); ax.grid(True, alpha=0.4); ax.legend()
    fig.suptitle(f"实验对比（{N_SEED} 种子均值）", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  对比图已保存: {out_path}")


def plot_timeseries(cfg, out_path):
    """单种子时序：估计距离、推力幅值、推力变化率"""
    seed = 0
    r_a, orb = run_one(cfg, "angle_only", seed)
    r_r, _ = run_one(cfg, "range_angle", seed)
    T = orb.T_orbit
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    for r, label, c in [(r_a, "仅测角", "tab:orange"), (r_r, "距离+角度", "tab:green")]:
        x_true = r.states[0:6, :] - r.states[6:12, :]
        true_d = np.linalg.norm(x_true[:3, :], axis=0)
        est_d = np.linalg.norm(r.x_est_history[:3, :], axis=0)
        u = np.linalg.norm(r.u_p_history, axis=0)
        du = np.linalg.norm(np.diff(r.u_p_history, axis=1), axis=0)

        axes[0].plot(r.t / T, est_d - true_d, color=c, label=f"{label} (估计-真实)")
        axes[1].plot(r.t / T, u, color=c, label=label)
        axes[2].plot(r.t[1:] / T, du, color=c, label=label, alpha=0.8)

    axes[0].axhline(0, color="k", linewidth=0.6, linestyle=":")
    axes[0].set_ylabel("距离估计偏差 (km)"); axes[0].legend(); axes[0].grid(True)
    axes[0].set_title("距离估计偏差 (估计 − 真实)")
    axes[1].set_ylabel("‖u_p‖ (km/s²)"); axes[1].set_yscale("log"); axes[1].legend(); axes[1].grid(True)
    axes[1].set_title("追踪星推力幅值")
    axes[2].set_xlabel("时间 (轨道周期)"); axes[2].set_ylabel("‖Δu_p‖ (km/s²)")
    axes[2].set_yscale("log"); axes[2].legend(); axes[2].grid(True)
    axes[2].set_title("推力变化率（抖动指标）")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  时序图已保存: {out_path}")


if __name__ == "__main__":
    cfg = SCENARIOS[SCENARIO_KEY]
    print(f"实验场景: {cfg['name']}（{N_SEED} 种子）")

    groups = {}
    groups["A. baseline"]      = run_group(cfg, "A. baseline")
    groups["B. perfect range"] = run_group(cfg, "B. perfect range", sigma_range=1e-6)
    groups["C. low Q (×0.01)"] = run_group(cfg, "C. low Q",  q_scale=0.01)
    groups["D. high Q (×100)"] = run_group(cfg, "D. high Q", q_scale=100.0)
    groups["E. shared x0"]     = run_group(cfg, "E. shared x0", share_x0_seed=999)

    plot_groups(groups, OUT_DIR / "groups.png")
    plot_timeseries(cfg, OUT_DIR / "timeseries.png")
    print(f"\n完成，输出目录: {OUT_DIR}")
