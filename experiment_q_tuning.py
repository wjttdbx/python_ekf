"""
Q 矩阵速度惩罚调优实验 — 主动设计代价函数 vs 被动接受仅测角 EKF"红利"

扫描 α ∈ (0,1]，令 Q = diag([1, 1, α, α])，在理想 SDRE（无 EKF,无噪声）
下求解 ARE 并闭环仿真，记录捕获时间 T 和控制能量 ∫‖u‖² dt。

比较：
- 帕累托前沿 ← 理想 SDRE 扫描 α
- 仅测角 EKF ← 被动红利基准
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

OUT_DIR = Path("outputs/figures/q_tuning")
OUT_DIR.mkdir(parents=True, exist_ok=True)

cfg = {
    "X_p0": np.array([100.0, 500.0, 0.01, 0.01]),
    "X_e0": np.array([0.0, 0.0, 0.0, 0.0]),
    "nu0": 0.0,
    "chief_orbit": {"a": 15000.0, "e": 0.5},
    "gamma": np.sqrt(2),
}
X_p0, X_e0, nu0 = cfg["X_p0"], cfg["X_e0"], cfg["nu0"]
x_rel0 = X_p0 - X_e0
init_dist = float(np.linalg.norm(x_rel0[:2]))
R_ctrl = np.eye(2) * 1e13
gamma = cfg["gamma"]

orb = OrbitalDynamics2D(mu=MU, a_c=cfg["chief_orbit"]["a"], e_c=cfg["chief_orbit"]["e"])

# ── 基准：仅测角 EKF（10 种子均值） ─────────────────────────────────────
SIGMA_RANGE = 0.01
ao_ts, ao_es = [], []
for seed in range(10):
    scale_r = init_dist / REF_DIST
    sp, sv = REF_SIGMA_POS * scale_r, REF_SIGMA_VEL * scale_r
    P0 = np.diag([sp**2]*2 + [sv**2]*2)
    Q_ekf = np.diag([5e-4, 5e-4, 5e-8, 5e-8])
    R_ekf = np.diag([SIGMA_ANG**2])

    rng_init = np.random.default_rng(seed)
    noise = rng_init.standard_normal(4) * np.array([REF_SIGMA_POS*scale_r]*2 + [REF_SIGMA_VEL*scale_r]*2)
    x0_est = x_rel0 + noise
    ekf = RelativeStateEKF2D(x0=x0_est, P0=P0, Q=Q_ekf, R=R_ekf)
    ctrl = SDREGameController2D(Q=np.eye(4), R=R_ctrl, gamma=gamma)
    rng = np.random.default_rng(seed + 1000)
    sim = EKFSDRESimulation2D(dynamics=orb, controller=ctrl, ekf=ekf,
                              X_p0=X_p0, X_e0=X_e0, nu0=nu0,
                              dt=10.0, are_interval=1, rng=rng)
    r = sim.run(t_end=5.0 * orb.T_orbit)
    if r.captured:
        idx = np.argmax(r.dist_history < 0.1)
        ao_ts.append(r.t[idx] / 3600)
        ao_es.append(float(np.trapezoid(np.sum(r.u_p_history[:, :idx+1]**2, axis=0), r.t[:idx+1])))

ao_t_mean = np.mean(ao_ts)
ao_e_mean = np.mean(ao_es)
print(f"仅测角 EKF 基准: T={ao_t_mean:.2f}±{np.std(ao_ts):.2f}h  E={ao_e_mean:.3e}±{np.std(ao_es):.3e}")

# ── R 扫描：理想 SDRE ────────────────────────────────────────────────────
# R = 1e13 是原始值，减小 R → 降低控制惩罚 → 允许更大推力 → 更快捕获
R_vals = [5e10, 1e11, 5e11, 1e12, 5e12, 1e13, 5e13]
results = []  # [(R_val, T_h, E, u_mean, u_peak)]

for R_val in R_vals:
    R_scan = np.eye(2) * R_val
    ctrl = SDREGameController2D(Q=np.eye(4), R=R_scan, gamma=gamma)
    ekf = RelativeStateEKF2D(
        x0=x_rel0,
        P0=np.diag([(REF_SIGMA_POS*init_dist/REF_DIST)**2]*2 + [(REF_SIGMA_VEL*init_dist/REF_DIST)**2]*2),
        Q=np.diag([5e-4, 5e-4, 5e-8, 5e-8]),
        R=np.diag([SIGMA_ANG**2]),
    )
    sim = EKFSDRESimulation2D(dynamics=orb, controller=ctrl, ekf=ekf,
                              X_p0=X_p0, X_e0=X_e0, nu0=nu0,
                              dt=10.0, are_interval=1, rng=None)
    r = sim.run(t_end=5.0 * orb.T_orbit)

    if r.captured:
        idx = np.argmax(r.dist_history < 0.1)
        T_h = r.t[idx] / 3600
        E = float(np.trapezoid(np.sum(r.u_p_history[:, :idx+1]**2, axis=0), r.t[:idx+1]))
        u_mag = np.linalg.norm(r.u_p_history[:, :idx+1], axis=0)
    else:
        T_h = r.t[-1] / 3600
        E = float(np.trapezoid(np.sum(r.u_p_history**2, axis=0), r.t))
        u_mag = np.linalg.norm(r.u_p_history, axis=0)
    results.append((R_val, T_h, E, float(np.mean(u_mag)), float(np.max(u_mag))))
    print(f"R={R_val:.1e}: T={T_h:.2f}h  E={E:.3e}  mean_u={np.mean(u_mag):.3e}  peak_u={np.max(u_mag):.3e}  captured={r.captured}")

# ── 帕累托前沿图 ──────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

R_arr = np.array([r[0] for r in results])
Ts = np.array([r[1] for r in results])
Es = np.array([r[2] for r in results])
peak_us = np.array([r[4] for r in results])

# 左图: 帕累托前沿
ax1.plot(Ts, Es, "o-", color="tab:blue", linewidth=1.5, markersize=6, label="理想 SDRE (调 R)")
for r_val, t, e in zip(R_arr, Ts, Es):
    ax1.annotate(f"R={r_val:.0e}", (t, e), textcoords="offset points", xytext=(8, -4), fontsize=7)
ax1.scatter([ao_t_mean], [ao_e_mean], marker="*", s=200, color="tab:orange",
            zorder=5, label=f"仅测角 EKF (10种子均值)")
ax1.set_xlabel("捕获时间 T (h)"); ax1.set_ylabel("控制能量 ∫‖u‖² dt")
ax1.set_title("时间-能量 帕累托前沿")
ax1.legend(); ax1.grid(True, alpha=0.4)

# 右图: 峰值推力 vs R
ax2.semilogx(R_arr, peak_us, "s-", color="tab:red", linewidth=1.5, markersize=6)
ax2.set_xlabel("控制惩罚 R"); ax2.set_ylabel("峰值推力 ‖u_p‖ (km/s²)")
ax2.set_title("峰值推力随 R 变化")
ax2.grid(True, alpha=0.4)

fig.suptitle("R 控制惩罚调优 — 理想 SDRE vs 仅测角 EKF", fontsize=13)
fig.tight_layout()
fig.savefig(OUT_DIR / "pareto_frontier.png", dpi=150); plt.close(fig)
print(f"\n帕累托前沿图已保存: {OUT_DIR / 'pareto_frontier.png'}")

# ── 时序对比：最优 α vs 仅测角 vs 原始 ──────────────────────────────────
best_idx = np.argmin(Ts)
best_R = R_arr[best_idx]
print(f"\n最优 R = {best_R:.1e}, T = {Ts[best_idx]:.2f}h")

# 跑一次最优 R 仿真用作时序图
R_best = np.eye(2) * best_R
ctrl_best = SDREGameController2D(Q=np.eye(4), R=R_best, gamma=gamma)
ekf_best = RelativeStateEKF2D(x0=x_rel0,
    P0=np.diag([(REF_SIGMA_POS*init_dist/REF_DIST)**2]*2 + [(REF_SIGMA_VEL*init_dist/REF_DIST)**2]*2),
    Q=np.diag([5e-4, 5e-4, 5e-8, 5e-8]),
    R=np.diag([SIGMA_ANG**2]))
sim_best = EKFSDRESimulation2D(dynamics=orb, controller=ctrl_best, ekf=ekf_best,
                               X_p0=X_p0, X_e0=X_e0, nu0=nu0,
                               dt=10.0, are_interval=1, rng=None)
r_best = sim_best.run(t_end=5.0 * orb.T_orbit)

# 原始 α=1
ctrl_orig = SDREGameController2D(Q=np.eye(4), R=R_ctrl, gamma=gamma)
ekf_orig = RelativeStateEKF2D(x0=x_rel0,
    P0=np.diag([(REF_SIGMA_POS*init_dist/REF_DIST)**2]*2 + [(REF_SIGMA_VEL*init_dist/REF_DIST)**2]*2),
    Q=np.diag([5e-4, 5e-4, 5e-8, 5e-8]),
    R=np.diag([SIGMA_ANG**2]))
sim_orig = EKFSDRESimulation2D(dynamics=orb, controller=ctrl_orig, ekf=ekf_orig,
                               X_p0=X_p0, X_e0=X_e0, nu0=nu0,
                               dt=10.0, are_interval=1, rng=None)
r_orig = sim_orig.run(t_end=5.0 * orb.T_orbit)

# 仅测角
seed = 0
rng_init = np.random.default_rng(seed)
scale_r = init_dist / REF_DIST
noise = rng_init.standard_normal(4) * np.array([REF_SIGMA_POS*scale_r]*2 + [REF_SIGMA_VEL*scale_r]*2)
ekf_ao = RelativeStateEKF2D(x0=x_rel0 + noise,
    P0=np.diag([(REF_SIGMA_POS*scale_r)**2]*2 + [(REF_SIGMA_VEL*scale_r)**2]*2),
    Q=np.diag([5e-4, 5e-4, 5e-8, 5e-8]),
    R=np.diag([SIGMA_ANG**2]))
ctrl_ao = SDREGameController2D(Q=np.eye(4), R=R_ctrl, gamma=gamma)
rng_ao = np.random.default_rng(seed + 1000)
sim_ao = EKFSDRESimulation2D(dynamics=orb, controller=ctrl_ao, ekf=ekf_ao,
                             X_p0=X_p0, X_e0=X_e0, nu0=nu0,
                             dt=10.0, are_interval=1, rng=rng_ao)
r_ao = sim_ao.run(t_end=5.0 * orb.T_orbit)

T_orbit = orb.T_orbit
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# 距离下降
for ax in [axes[0, 0]]:
    for r, label, c, ls in [
        (r_orig, f"原始 R=1e13 T={r_orig.t[-1]/3600:.1f}h", "tab:green", "-"),
        (r_best, f"最优 R={best_R:.0e} T={r_best.t[-1]/3600:.1f}h", "tab:blue", "-"),
        (r_ao,   f"仅测角 EKF T={r_ao.t[-1]/3600:.1f}h", "tab:orange", "--"),
    ]:
        x_true = r.states[0:2, :] - r.states[4:6, :]
        ax.plot(r.t / T_orbit, np.linalg.norm(x_true, axis=0), color=c, linestyle=ls, label=label, linewidth=1.2)
    ax.axhline(0.1, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylabel("相对距离 (km)"); ax.set_yscale("log")
    ax.set_title("相对距离下降曲线")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# 推力幅值
for ax in [axes[0, 1]]:
    for r, label, c, ls in [
        (r_orig, f"原始 R=1e13", "tab:green", "-"),
        (r_best, f"最优 R={best_R:.0e}", "tab:blue", "-"),
        (r_ao,   f"仅测角 EKF", "tab:orange", "--"),
    ]:
        ax.plot(r.t / T_orbit, np.linalg.norm(r.u_p_history, axis=0), color=c, linestyle=ls, label=label, linewidth=1.0)
    ax.set_ylabel("‖u_p‖ (km/s²)"); ax.set_yscale("log")
    ax.set_title("推力幅值时序")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# 累计能量
for ax in [axes[1, 0]]:
    for r, label, c, ls in [
        (r_orig, f"原始 R=1e13", "tab:green", "-"),
        (r_best, f"最优 R={best_R:.0e}", "tab:blue", "-"),
        (r_ao,   f"仅测角 EKF", "tab:orange", "--"),
    ]:
        cum_e = np.cumsum(np.sum(r.u_p_history**2, axis=0)) * (r.t[1] - r.t[0])
        ax.plot(r.t / T_orbit, cum_e, color=c, linestyle=ls, label=label, linewidth=1.0)
    ax.set_xlabel("时间 (轨道周期)"); ax.set_ylabel("累计 ∫‖u‖² dt")
    ax.set_title("累计控制能量")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# 推力抖动
for ax in [axes[1, 1]]:
    for r, label, c, ls in [
        (r_orig, f"原始 R=1e13", "tab:green", "-"),
        (r_best, f"最优 R={best_R:.0e}", "tab:blue", "-"),
        (r_ao,   f"仅测角 EKF", "tab:orange", "--"),
    ]:
        du = np.linalg.norm(np.diff(r.u_p_history, axis=1), axis=0)
        ax.plot(r.t[1:] / T_orbit, du, color=c, linestyle=ls, alpha=0.7, linewidth=0.8)
    ax.set_xlabel("时间 (轨道周期)"); ax.set_ylabel("‖Δu_p‖ (km/s²)")
    ax.set_yscale("log")
    ax.set_title("推力抖动")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

fig.suptitle(f"R 控制惩罚调优 vs 仅测角 EKF vs 原始 R=1e13（最佳 R={best_R:.0e}）", fontsize=13)
fig.tight_layout()
fig.savefig(OUT_DIR / "timeseries_compare.png", dpi=150); plt.close(fig)
print(f"时序对比图已保存: {OUT_DIR / 'timeseries_compare.png'}")

# ── 打印汇总 ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("汇总: R 扫描 vs 仅测角 EKF 基准")
print("=" * 70)
print(f"仅测角 EKF: T={ao_t_mean:.2f}h  E={ao_e_mean:.3e}  (10种子均值)")
print()
print(f"{'R':>10s}  {'T(h)':>7s}  {'E':>11s}  {'mean_u':>10s}  {'peak_u':>10s}")
print("-" * 55)
for r in results:
    print(f"{r[0]:10.1e}  {r[1]:7.2f}  {r[2]:11.3e}  {r[3]:10.3e}  {r[4]:10.3e}")

# 找出 T 不大于仅测角 EKF 的 R
print()
for r in results:
    if r[1] <= ao_t_mean:
        print(f"R={r[0]:.1e}: T={r[1]:.2f}h ≤ 仅测角 {ao_t_mean:.2f}h,  E={r[2]:.3e} vs {ao_e_mean:.3e}")
