"""
状态依赖 R(r) 调优 — 直接参数化 + 网格搜索 + 闭环评测

R(r) = R_min + (R_max - R_min) * sigmoid((r - r_0) / w)

直观: 远处 (r 大) → R 小 → 激进控制
      近处 (r 小) → R 大 → 平稳捕获
"""

import zhplot
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import solve_ivp

from aerospace.dynamics.nerm_2d import OrbitalDynamics2D
from aerospace.control.sdre_2d import SDREGameController2D
from aerospace.estimation.ekf_2d import RelativeStateEKF2D
from aerospace.simulation.nerm_ekf_sdre_2d import EKFSDRESimulation2D
from run_scenarios import MU, SIGMA_ANG, REF_DIST, REF_SIGMA_POS, REF_SIGMA_VEL, GAMMA

OUT_DIR = Path("outputs/figures/state_dep_R")
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
orb = OrbitalDynamics2D(mu=MU, a_c=15000.0, e_c=0.5)
gamma = np.sqrt(2)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


class StateDependentRController:
    """在每个仿真步根据当前相对距离动态构造 R 矩阵"""

    def __init__(self, R_min, R_max, r_0, w, gamma_val=np.sqrt(2)):
        self.R_min = R_min
        self.R_max = R_max
        self.r_0 = r_0
        self.w = w
        self.gamma_val = gamma_val
        self._controller_cache = {}
        self._R_cache = {}

    def get_R(self, r):
        key = round(r, 1)
        if key not in self._R_cache:
            R_val = self.R_min + (self.R_max - self.R_min) * sigmoid((r - self.r_0) / self.w)
            self._R_cache[key] = float(R_val)
        return self._R_cache[key]

    def get_ctrl(self, r):
        """获取该距离对应的 SDRE 控制器 (带缓存以避免重复构造)"""
        R_val = self.get_R(r)
        key = round(R_val, 8)
        if key not in self._controller_cache:
            self._controller_cache[key] = SDREGameController2D(
                Q=np.eye(4), R=np.eye(2) * key, gamma=self.gamma_val
            )
        return self._controller_cache[key]


def run_simulation_state_dep_R(sd_ctrl: StateDependentRController):
    """使用状态依赖 R 运行理想 SDRE 闭环仿真"""
    ekf = RelativeStateEKF2D(
        x0=x_rel0,
        P0=np.diag([(REF_SIGMA_POS * init_dist / REF_DIST)**2] * 2 +
                    [(REF_SIGMA_VEL * init_dist / REF_DIST)**2] * 2),
        Q=np.diag([5e-4, 5e-4, 5e-8, 5e-8]),
        R=np.diag([SIGMA_ANG**2]),
    )

    # 手动仿真循环 (不使用 EKFSDRESimulation2D, 因为需要动态切换控制器)
    dt = 10.0
    t_end = 5.0 * orb.T_orbit
    N = int(t_end / dt)
    state = np.zeros(9)
    state[0:4] = X_p0
    state[4:8] = X_e0
    state[8] = nu0
    t = 0.0

    t_hist, u_hist, dist_hist = [0.0], [np.zeros(2)], [init_dist]
    captured = False

    for k in range(N):
        nu = state[8]
        r_c, nu_dot, nu_ddot = orb.get_orbital_params(nu)

        X_p_true = state[0:4]
        x_true_rel = state[0:4] - state[4:8]
        r = np.linalg.norm(x_true_rel[:2])

        ctrl = sd_ctrl.get_ctrl(r)
        A_SDC = orb.get_SDC_matrix(X_p_true, X_p_true - x_true_rel, r_c, nu_dot, nu_ddot)

        u_p, u_e = ctrl.compute_control(A_SDC, x_true_rel, t=t, solve_are=(k % 1 == 0),
                                        x_rel_e=x_true_rel)

        sol = solve_ivp(
            orb.dynamics_9d, [t, t + dt], state,
            args=(u_p, u_e), method="RK45", rtol=1e-8, atol=1e-10,
        )
        state = sol.y[:, -1]
        t += dt

        t_hist.append(t)
        u_hist.append(u_p.copy())
        new_dist = np.linalg.norm(state[0:2] - state[4:6])
        dist_hist.append(new_dist)

        if new_dist < 0.1:
            print(f"  捕获! t={t:.1f}s  dist={new_dist*1000:.1f}m")
            captured = True
            break

    t_arr = np.array(t_hist)
    u_arr = np.array(u_hist).T
    dist_arr = np.array(dist_hist)

    if captured:
        idx = np.argmax(dist_arr < 0.1)
        T_h = t_arr[idx] / 3600
        E = float(np.trapezoid(np.sum(u_arr[:, :idx+1]**2, axis=0), t_arr[:idx+1]))
        u_mag = np.linalg.norm(u_arr[:, :idx+1], axis=0)
        peak_u = float(np.max(u_mag))
        du = np.linalg.norm(np.diff(u_arr[:, :idx+1], axis=1), axis=0)
        jitter = float(np.mean(du)) if len(du) > 0 else 0
    else:
        T_h = t_arr[-1] / 3600
        E = float(np.trapezoid(np.sum(u_arr**2, axis=0), t_arr))
        peak_u = float(np.max(np.linalg.norm(u_arr, axis=0)))
        jitter = float(np.mean(np.linalg.norm(np.diff(u_arr, axis=1), axis=0)))

    return dict(T=T_h, E=E, peak_u=peak_u, jitter=jitter, captured=captured,
                t=t_arr, u=u_arr, dist=dist_arr)


# ===== 网格搜索 R(r) 参数 =====
print("状态依赖 R(r) 参数网格搜索...")
print("R(r) = R_min + (R_max - R_min) * sigmoid((r - r_0) / w)")
print()

R_min_vals = [5e10, 1e11, 5e11]
R_max_vals = [3e12, 5e12, 1e13]
r_0_vals = [50, 200, 500]
w_vals = [20, 100, 300]

best_T, best_params = 1e9, None
all_results = []

for R_min in R_min_vals:
    for R_max in R_max_vals:
        for r_0 in r_0_vals:
            for w in w_vals:
                if R_min >= R_max:
                    continue
                sd_ctrl = StateDependentRController(R_min, R_max, r_0, w)
                m = run_simulation_state_dep_R(sd_ctrl)
                all_results.append((R_min, R_max, r_0, w, m["T"], m["E"], m["peak_u"], m["jitter"]))

                if m["T"] < best_T and m["captured"]:
                    best_T = m["T"]
                    best_params = (R_min, R_max, r_0, w)

# 排序并打印最佳
all_results.sort(key=lambda x: x[4])  # 按 T 排序

print(f"{'R_min':>8s}  {'R_max':>8s}  {'r_0':>6s}  {'w':>6s}  {'T(h)':>7s}  {'E':>10s}  {'peak_u':>10s}  {'jitter':>10s}")
print("-" * 85)
for r in all_results[:15]:
    print(f"{r[0]:8.1e}  {r[1]:8.1e}  {r[2]:6.0f}  {r[3]:6.0f}  {r[4]:7.2f}  {r[5]:10.3e}  {r[6]:10.3e}  {r[7]:10.3e}")

print(f"\n最佳参数: R_min={best_params[0]:.1e}, R_max={best_params[1]:.1e}, r_0={best_params[2]:.0f}, w={best_params[3]:.0f}")

# ===== 与固定 R + 仅测角 EKF 对比 =====
print("\n--- 对比基准 ---")

# 仅测角 EKF
SIGMA_RANGE = 0.01
ao_ms = []
for seed in range(10):
    s = init_dist / REF_DIST; sp, sv = REF_SIGMA_POS * s, REF_SIGMA_VEL * s
    rng_init = np.random.default_rng(seed)
    noise = rng_init.standard_normal(4) * np.array([sp] * 2 + [sv] * 2)
    ekf = RelativeStateEKF2D(x0=x_rel0 + noise,
        P0=np.diag([sp**2] * 2 + [sv**2] * 2), Q=np.diag([5e-4, 5e-4, 5e-8, 5e-8]), R=np.diag([SIGMA_ANG**2]))
    ctrl = SDREGameController2D(Q=np.eye(4), R=np.eye(2) * 1e13, gamma=gamma)
    rng = np.random.default_rng(seed + 1000)
    sim = EKFSDRESimulation2D(dynamics=orb, controller=ctrl, ekf=ekf,
                              X_p0=X_p0, X_e0=X_e0, nu0=nu0, dt=10.0, are_interval=1, rng=rng)
    r = sim.run(t_end=5.0 * orb.T_orbit)
    if r.captured:
        idx = np.argmax(r.dist_history < 0.1)
        ao_ms.append((r.t[idx] / 3600, float(np.trapezoid(np.sum(r.u_p_history[:, :idx+1]**2, axis=0), r.t[:idx+1]))))

ao_T = np.mean([m[0] for m in ao_ms])
ao_E = np.mean([m[1] for m in ao_ms])

# 固定 R 扫描
fixed_Rs = [1e11, 5e11, 1e12, 2e12, 5e12, 1e13]
fixed_ms = []
for Rv in fixed_Rs:
    ctrl = SDREGameController2D(Q=np.eye(4), R=np.eye(2) * Rv, gamma=gamma)
    ekf = RelativeStateEKF2D(x0=x_rel0,
        P0=np.diag([(REF_SIGMA_POS * init_dist / REF_DIST)**2] * 2 + [(REF_SIGMA_VEL * init_dist / REF_DIST)**2] * 2),
        Q=np.diag([5e-4, 5e-4, 5e-8, 5e-8]), R=np.diag([SIGMA_ANG**2]))
    sim = EKFSDRESimulation2D(dynamics=orb, controller=ctrl, ekf=ekf,
                              X_p0=X_p0, X_e0=X_e0, nu0=nu0, dt=10.0, are_interval=1, rng=None)
    r = sim.run(t_end=5.0 * orb.T_orbit)
    if r.captured:
        idx = np.argmax(r.dist_history < 0.1)
        T_h = r.t[idx] / 3600
        E = float(np.trapezoid(np.sum(r.u_p_history[:, :idx+1]**2, axis=0), r.t[:idx+1]))
    else:
        T_h = r.t[-1] / 3600
        E = float(np.trapezoid(np.sum(r.u_p_history**2, axis=0), r.t))
    fixed_ms.append((T_h, E, Rv))

# 帕累托前沿图
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 左图: T vs E 帕累托前沿
fixed_Ts = [m[0] for m in fixed_ms]
fixed_Es = [m[1] for m in fixed_ms]
ax1 = axes[0]
ax1.plot(fixed_Ts, fixed_Es, "o-", color="tab:blue", linewidth=2, markersize=8, label="固定 R (理想 SDRE)")
for i, (T, E, Rv) in enumerate(fixed_ms):
    ax1.annotate(f"R={Rv:.0e}", (T, E), textcoords="offset points", xytext=(8, 6), fontsize=8, color="tab:blue")
ax1.scatter([ao_T], [ao_E], marker="*", s=200, color="tab:orange", zorder=5, label="仅测角 EKF (10种子均值)")

# 状态依赖 R 最佳点
best_Rmin, best_Rmax, best_r0, best_w = best_params
sd_ctrl_best = StateDependentRController(best_Rmin, best_Rmax, best_r0, best_w)
m_best = run_simulation_state_dep_R(sd_ctrl_best)
ax1.scatter([m_best["T"]], [m_best["E"]], marker="D", s=150, color="tab:red", zorder=6,
            label=f"状态依赖 R(r)\nR_min={best_Rmin:.0e}, R_max={best_Rmax:.0e}")

ax1.set_xlabel("捕获时间 T (h)"); ax1.set_ylabel("控制能量 ∫‖u‖² dt")
ax1.set_title("帕累托前沿: 状态依赖 R(r) vs 固定 R vs 仅测角 EKF")
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.4)

# 右图: R(r) 函数曲线
ax2 = axes[1]
r_vals = np.logspace(-1, 3, 500)
R_vals = best_Rmin + (best_Rmax - best_Rmin) * sigmoid((r_vals - best_r0) / best_w)

ax2.semilogx(r_vals, R_vals, color="tab:red", linewidth=2, label=f"状态依赖 R(r)")
ax2.axhline(y=best_Rmin, color="tab:red", linestyle=":", alpha=0.4)
ax2.axhline(y=best_Rmax, color="tab:red", linestyle=":", alpha=0.4)
ax2.axvline(x=best_r0, color="gray", linestyle=":", alpha=0.5, label=f"r_0={best_r0:.0f} km")

# 标记仅测角等效 R
ax2.axhline(y=2.5e12, color="tab:orange", linestyle="--", alpha=0.6, label="仅测角等效 R≈2.5e12")

ax2.set_xlabel("相对距离 r (km)"); ax2.set_ylabel("R(r)")
ax2.set_title(f"状态依赖 R(r) = {best_Rmin:.0e} + {best_Rmax-best_Rmin:.0e}⋅σ((r-{best_r0:.0f})/{best_w:.0f})")
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.4)

fig.suptitle("逆最优 SDRE: 状态依赖控制惩罚 R(r)", fontsize=13)
fig.tight_layout()
fig.savefig(OUT_DIR / "state_dep_R_pareto.png", dpi=150); plt.close(fig)
print(f"\n图已保存: {OUT_DIR / 'state_dep_R_pareto.png'}")

# 汇总
print("\n" + "=" * 70)
print("汇总: 状态依赖 R(r) vs 固定 R vs 仅测角 EKF")
print("=" * 70)
print(f"状态依赖 R(r): T={m_best['T']:.2f}h  E={m_best['E']:.3e}  peak_u={m_best['peak_u']:.3e}  jitter={m_best['jitter']:.3e}")
print(f"仅测角 EKF:     T={ao_T:.2f}h  E={ao_E:.3e}")
for T, E, Rv in fixed_ms:
    print(f"固定 R={Rv:.1e}:    T={T:.2f}h  E={E:.3e}")
print(f"\nR(r) 范围: [{best_Rmin:.1e}, {best_Rmax:.1e}], 过渡距离 r_0={best_r0:.0f}km")
