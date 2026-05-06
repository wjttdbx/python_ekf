"""
2D 追加实验：探究"仅测角 vs 距离+角度 vs 理想 SDRE"的面内行为

实验组（均基于 scenario_2d_1，每组 10 种子）：
  F. ideal SDRE:      无 EKF，无噪声，使用真实相对状态
  A. baseline:        当前 angle_only / range_angle 两组
  B. 完美距离测量:     SIGMA_RANGE = 1 mm (1e-6 km) — 排除距离噪声影响
  C. 极小过程噪声 Q:   Q /= 100 — 检验是否 Q 主导抖动
  D. 极大过程噪声 Q:   Q *= 100 — 反向验证
  E. 共同初始估计噪声:  两模式用相同 x0_est（隔离初值随机性）

输出：
  - 多组 bar 图汇总（捕获时间、能量、抖动）
  - 估计距离时序对比 + 推力时序对比（典型种子）
"""
import zhplot
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from aerospace.dynamics.nerm_2d import OrbitalDynamics2D
from aerospace.control.sdre_2d import SDREGameController2D
from aerospace.estimation.ekf_2d import RelativeStateEKF2D
from aerospace.simulation.nerm_ekf_sdre_2d import EKFSDRESimulation2D
from run_scenarios import MU, SIGMA_ANG, REF_DIST, REF_SIGMA_POS, REF_SIGMA_VEL, GAMMA

N_SEED = 10
OUT_DIR = Path("outputs/figures/experiment_2d")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 2D 场景定义（LVLH 4D 面内状态） ──────────────────────────────────────────
SCENARIOS_2D = {
    "scenario_2d_1": {
        "name": "2D-1: 同轨道同平面",
        "X_p0": np.array([100.0, 500.0, 0.01, 0.01]),
        "X_e0": np.array([0.0, 0.0, 0.0, 0.0]),
        "nu0": 0.0,
        "chief_orbit": {"a": 15000.0, "e": 0.5},
        "gamma": np.sqrt(2),
    },
    "scenario_2d_2": {
        "name": "2D-2: 偏移轨道",
        "X_p0": np.array([200.0, -200.0, 0.005, -0.005]),
        "X_e0": np.array([0.0, 0.0, 0.0, 0.0]),
        "nu0": np.pi / 4,
        "chief_orbit": {"a": 15000.0, "e": 0.5},
        "gamma": np.sqrt(2),
    },
    "scenario_2d_3": {
        "name": "2D-3: 大初始距离",
        "X_p0": np.array([300.0, -1200.0, 0.01, 0.01]),
        "X_e0": np.array([0.0, 0.0, 0.0, 0.0]),
        "nu0": 0.0,
        "chief_orbit": {"a": 15000.0, "e": 0.5},
        "gamma": np.sqrt(2),
    },
}

SIGMA_RANGE = 0.01  # 距离测量噪声 1σ (km = 10 m)
SCENARIO_KEY = "scenario_2d_1"


def make_ekf_2d(mode: str, x0, init_dist, sigma_range=0.01, q_scale=1.0, sigma_ang=None):
    """灵活构建 2D EKF：mode='angle_only'|'range_angle'，参数可调"""
    if sigma_ang is None:
        sigma_ang = SIGMA_ANG
    scale = init_dist / REF_DIST
    sp, sv = REF_SIGMA_POS * scale, REF_SIGMA_VEL * scale
    P0 = np.diag([sp**2]*2 + [sv**2]*2)
    Q = np.diag([5e-4, 5e-4, 5e-8, 5e-8]) * q_scale
    if mode == "angle_only":
        R = np.diag([sigma_ang**2])
    else:
        R = np.diag([sigma_range**2, sigma_ang**2])
    return RelativeStateEKF2D(x0=x0, P0=P0, Q=Q, R=R)


def run_one_2d(cfg, mode, seed, sigma_range=0.01, q_scale=1.0, share_x0_seed=None,
               e_override=None, gamma_override=None, sigma_ang=None):
    X_p0 = cfg["X_p0"].copy()
    X_e0 = cfg["X_e0"].copy()
    nu0 = cfg["nu0"]
    x_rel0 = X_p0 - X_e0
    init_dist = float(np.linalg.norm(x_rel0[:2]))

    e_c = e_override if e_override is not None else cfg["chief_orbit"]["e"]
    gamma = gamma_override if gamma_override is not None else cfg.get("gamma", GAMMA)
    orb = OrbitalDynamics2D(mu=MU, a_c=cfg["chief_orbit"]["a"], e_c=e_c)
    ctrl = SDREGameController2D(Q=np.eye(4), R=np.eye(2) * 1e13, gamma=gamma)

    if mode == "ideal":
        ekf = make_ekf_2d("angle_only", x_rel0, init_dist, sigma_ang=sigma_ang)
        sim = EKFSDRESimulation2D(dynamics=orb, controller=ctrl, ekf=ekf,
                                  X_p0=X_p0, X_e0=X_e0, nu0=nu0,
                                  dt=10.0, are_interval=1, rng=None)
        return sim.run(t_end=5.0 * orb.T_orbit), orb

    seed_init = share_x0_seed if share_x0_seed is not None else seed
    rng_init = np.random.default_rng(seed_init)
    scale = init_dist / REF_DIST
    noise = rng_init.standard_normal(4) * np.array([REF_SIGMA_POS*scale]*2 + [REF_SIGMA_VEL*scale]*2)
    x0_est = x_rel0 + noise

    ekf = make_ekf_2d(mode, x0_est, init_dist, sigma_range=sigma_range, q_scale=q_scale, sigma_ang=sigma_ang)
    rng = np.random.default_rng(seed + 1000)
    sim = EKFSDRESimulation2D(dynamics=orb, controller=ctrl, ekf=ekf,
                              X_p0=X_p0, X_e0=X_e0, nu0=nu0,
                              dt=10.0, are_interval=1, rng=rng)
    return sim.run(t_end=5.0 * orb.T_orbit), orb


def metrics_2d(r):
    cap_idx = np.argmax(r.dist_history < 0.1) if r.captured else -1
    t_cap = r.t[cap_idx] if r.captured else np.nan
    energy = float(np.trapezoid(np.sum(r.u_p_history**2, axis=0), r.t))
    du = np.diff(r.u_p_history, axis=1)
    jitter = float(np.mean(np.linalg.norm(du, axis=0)))
    return dict(t_cap=t_cap, energy=energy, jitter=jitter, captured=r.captured)


def run_group_2d(cfg, label, modes=("angle_only", "range_angle"), **kwargs):
    out = {m: [] for m in modes}
    for seed in range(N_SEED):
        for mode in out:
            r, _ = run_one_2d(cfg, mode, seed, **kwargs)
            out[mode].append(metrics_2d(r))
    print(f"\n[{label}]")
    for mode in out:
        ts = [m["t_cap"]/3600 for m in out[mode] if not np.isnan(m["t_cap"])]
        es = [m["energy"] for m in out[mode]]
        js = [m["jitter"] for m in out[mode]]
        print(f"  {mode:12s}  t_cap={np.mean(ts):.2f}±{np.std(ts):.2f} h  "
              f"E={np.mean(es):.3e}  jitter={np.mean(js):.3e}")
    return out


def plot_groups_2d(groups, out_path):
    """各组 bar 对比（含理想 SDRE）"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    labels = list(groups.keys())
    x = np.arange(len(labels))

    mode_names = {"angle_only": "仅测角", "range_angle": "距离+角度", "ideal": "理想SDRE (无EKF)"}
    mode_colors = {"angle_only": "tab:orange", "range_angle": "tab:green", "ideal": "tab:blue"}
    all_modes_set = set()
    for g in groups.values():
        all_modes_set.update(g.keys())
    all_modes = sorted(all_modes_set, key=lambda m: {"ideal": 0, "angle_only": 1, "range_angle": 2}.get(m, 99))
    n_modes = len(all_modes)
    w = 0.8 / n_modes

    def vals(metric):
        result = {}
        for m in all_modes:
            row = []
            for l in labels:
                if m in groups[l]:
                    row.append(np.nanmean([s[metric] for s in groups[l][m]]))
                else:
                    row.append(np.nan)
            result[m] = row
        return result

    for ax, metric, ylabel, scale in [
        (axes[0], "t_cap", "捕获时间 (h)", 1/3600),
        (axes[1], "energy", "控制能量 ∫‖u‖² dt", 1.0),
        (axes[2], "jitter", "推力抖动均值 (km/s²)", 1.0),
    ]:
        vv = vals(metric)
        for i, mode in enumerate(all_modes):
            offset = (i - (n_modes - 1) / 2) * w
            ax.bar(x + offset, [v * scale for v in vv[mode]], w,
                   label=mode_names.get(mode, mode), color=mode_colors.get(mode))
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel); ax.grid(True, alpha=0.4); ax.legend()
    fig.suptitle(f"2D 实验对比（{N_SEED} 种子均值）", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  对比图已保存: {out_path}")


def plot_timeseries_2d(cfg, out_path):
    """单种子时序：估计距离、推力幅值、推力变化率（含理想 SDRE）"""
    seed = 0
    r_ideal, orb = run_one_2d(cfg, "ideal", seed)
    r_a, _ = run_one_2d(cfg, "angle_only", seed)
    r_r, _ = run_one_2d(cfg, "range_angle", seed)
    T = orb.T_orbit
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    # 理想 SDRE 用作基准
    u_ideal = np.linalg.norm(r_ideal.u_p_history, axis=0)
    du_ideal = np.linalg.norm(np.diff(r_ideal.u_p_history, axis=1), axis=0)

    for r, label, c in [(r_a, "仅测角", "tab:orange"), (r_r, "距离+角度", "tab:green")]:
        x_true = r.states[0:4, :] - r.states[4:8, :]
        true_d = np.linalg.norm(x_true[:2, :], axis=0)
        est_d = np.linalg.norm(r.x_est_history[:2, :], axis=0)
        u = np.linalg.norm(r.u_p_history, axis=0)
        du = np.linalg.norm(np.diff(r.u_p_history, axis=1), axis=0)

        axes[0].plot(r.t / T, est_d - true_d, color=c, label=f"{label} (估计-真实)")
        axes[1].plot(r.t / T, u, color=c, label=label)
        axes[2].plot(r.t[1:] / T, du, color=c, label=label, alpha=0.8)

    # 理想 SDRE 推力 & 抖动
    axes[1].plot(r_ideal.t / T, u_ideal, color="tab:blue", label="理想SDRE (无EKF)", linewidth=1.2)
    axes[2].plot(r_ideal.t[1:] / T, du_ideal, color="tab:blue", label="理想SDRE (无EKF)", alpha=0.8)

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
    cfg = SCENARIOS_2D[SCENARIO_KEY]
    print(f"实验场景: {cfg['name']}（{N_SEED} 种子）")

    groups = {}
    groups["F. ideal SDRE"]    = run_group_2d(cfg, "F. ideal SDRE (无EKF, 无噪声)", modes=("ideal",))
    groups["A. baseline"]      = run_group_2d(cfg, "A. baseline")
    groups["B. perfect range"] = run_group_2d(cfg, "B. perfect range", sigma_range=1e-6)
    groups["C. low Q (×0.01)"] = run_group_2d(cfg, "C. low Q", q_scale=0.01)
    groups["D. high Q (×100)"] = run_group_2d(cfg, "D. high Q", q_scale=100.0)
    groups["E. shared x0"]     = run_group_2d(cfg, "E. shared x0", share_x0_seed=999)

    plot_groups_2d(groups, OUT_DIR / "groups.png")
    plot_timeseries_2d(cfg, OUT_DIR / "timeseries.png")

    # ── 参数扫描 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("参数扫描实验")
    print("=" * 70)

    # 1. 偏心率扫描
    sweep_groups = {}
    for e_val in [0.0, 0.2, 0.3, 0.5, 0.7]:
        label = f"e={e_val:.1f}"
        sweep_groups[label] = run_group_2d(cfg, f"e={e_val:.1f}", e_override=e_val)
    plot_groups_2d(sweep_groups, OUT_DIR / "sweep_e.png")

    # 2. 角度噪声扫描
    sweep_groups = {}
    for sig_ang_deg in [0.001, 0.004, 0.008, 0.02, 0.05]:
        sig_ang_rad = sig_ang_deg * np.pi / 180.0
        label = f"σ_ang={sig_ang_deg}°"
        sweep_groups[label] = run_group_2d(cfg, f"σ_ang={sig_ang_deg}°", sigma_ang=sig_ang_rad)
    plot_groups_2d(sweep_groups, OUT_DIR / "sweep_sigma_ang.png")

    # 3. gamma 扫描
    sweep_groups = {}
    for gam in [1.1, 1.2, np.sqrt(2), 2.0, 5.0]:
        label = f"γ={gam:.2f}"
        sweep_groups[label] = run_group_2d(cfg, f"γ={gam:.2f}", gamma_override=gam)
    plot_groups_2d(sweep_groups, OUT_DIR / "sweep_gamma.png")

    print(f"\n完成，输出目录: {OUT_DIR}")
