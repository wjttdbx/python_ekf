"""
轨迹对比: 最优控制 (CasADi) vs 固定 R SDRE vs 逆最优 R(r) SDRE

验证逆最优 R(r) 是否真能逼近真正的时间-能量最优解。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import solve_ivp

from aerospace.dynamics.nerm_2d import OrbitalDynamics2D
from aerospace.control.sdre_2d import SDREGameController2D

OUT_DIR = Path("outputs/figures/trajectory_compare")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MU = 3.986e5
GAMMA_SDRE = np.sqrt(2)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


class StateDependentRController:
    """使用 R(r) 的 SDRE 控制器"""

    def __init__(self, fit_params: dict, gamma_val=GAMMA_SDRE):
        self.R_min = fit_params["R_min"]
        self.R_max = fit_params["R_max"]
        self.r_0 = fit_params["r_0"]
        self.w = max(fit_params["w"], 1.0)
        self.gamma_val = gamma_val
        self._ctrl_cache = {}
        self.history_R = []

    def get_R(self, r):
        return float(self.R_min + (self.R_max - self.R_min) *
                     sigmoid((r - self.r_0) / self.w))

    def get_ctrl(self, r):
        R_val = self.get_R(r)
        self.history_R.append(R_val)
        key = round(np.log10(max(R_val, 1e-30)), 4)
        if key not in self._ctrl_cache:
            self._ctrl_cache[key] = SDREGameController2D(
                Q=np.eye(4), R=np.eye(2) * max(R_val, 1e-30),
                gamma=self.gamma_val,
            )
        return self._ctrl_cache[key]


def run_with_trajectory(controller, orb, X_p0, X_e0, nu0=0.0,
                        dt=10.0, t_max=None):
    """运行闭环仿真并记录完整轨迹。"""
    if t_max is None:
        t_max = 5.0 * orb.T_orbit
    state = np.zeros(9)
    state[0:4] = X_p0
    state[4:8] = X_e0
    state[8] = nu0
    t_now = 0.0
    t_hist, traj, u_hist = [0.0], [state.copy()], [np.zeros(2)]
    N = int(t_max / dt)
    for k in range(N):
        nu = state[8]
        X_p = state[0:4]
        x_rel = X_p - state[4:8]
        r_c, nu_dot, nu_ddot = orb.get_orbital_params(nu)
        if isinstance(controller, StateDependentRController):
            ctrl = controller.get_ctrl(np.linalg.norm(x_rel[:2]))
        else:
            ctrl = controller
        A_SDC = orb.get_SDC_matrix(X_p, X_p - x_rel, r_c, nu_dot, nu_ddot)
        u_p, u_e = ctrl.compute_control(A_SDC, x_rel, t=t_now, solve_are=True)
        sol = solve_ivp(orb.dynamics_9d, [t_now, t_now + dt], state, args=(u_p, u_e),
                        method="RK45", rtol=1e-8, atol=1e-10)
        state = sol.y[:, -1]
        t_now += dt
        traj.append(state.copy())
        u_hist.append(u_p.copy())
        if np.linalg.norm(state[0:2] - state[4:6]) < 0.1:
            print(f"  Captured at t={t_now:.0f}s ({t_now/3600:.2f}h)")
            break
    return np.arange(len(traj)) * dt, np.array(traj).T, np.array(u_hist).T


def main():
    X_p0 = np.array([100.0, 500.0, 0.01, 0.01])
    X_e0 = np.array([0.0, 0.0, 0.0, 0.0])
    nu0 = 0.0
    orb = OrbitalDynamics2D(mu=MU, a_c=15000.0, e_c=0.5)

    # ── 1. 加载最优控制解 ──
    sol_file = Path("outputs/optimal_control/sol_gamma_1e+07.npz")
    if not sol_file.exists():
        print("ERROR: optimal control solution not found. Run Phase 1 first.")
        return
    opt = np.load(sol_file)
    t_opt, x_opt, u_opt = opt["t"], opt["x"], opt["u"]
    dist_opt = np.sqrt(x_opt[0]**2 + x_opt[1]**2)

    # ── 2. 加载逆最优 R(r) ──
    fit_file = Path("outputs/inverse_optimal/inverse_optimal_fit.npz")
    if fit_file.exists():
        fit = np.load(fit_file)
        fit_params = dict(
            R_min=float(fit["R_min"]), R_max=float(fit["R_max"]),
            r_0=float(fit["r_0"]), w=float(fit["w"])
        )
    else:
        print("WARNING: fit not found, using defaults")
        fit_params = dict(R_min=1e12, R_max=3e12, r_0=500.0, w=200.0)

    # ── 3. 固定 R SDRE ──
    print("Running fixed-R SDRE...")
    ctrl_fixed = SDREGameController2D(Q=np.eye(4), R=np.eye(2) * 1e13, gamma=GAMMA_SDRE)
    t_fix, traj_fix, u_fix = run_with_trajectory(ctrl_fixed, orb, X_p0, X_e0, nu0)

    # ── 4. 状态依赖 R(r) SDRE ──
    print(f"Running R(r) SDRE: R_min={fit_params['R_min']:.2e}, "
          f"R_max={fit_params['R_max']:.2e}, "
          f"r_0={fit_params['r_0']:.0f}, w={fit_params['w']:.0f}")
    ctrl_rx = StateDependentRController(fit_params)
    t_rx, traj_rx, u_rx = run_with_trajectory(ctrl_rx, orb, X_p0, X_e0, nu0)
    R_history = np.array(ctrl_rx.history_R)
    # 补全：R_history 在每次 compute_control 时记录，比 t_rx 少 1 (初始时未记录)
    if len(R_history) < len(t_rx):
        R_history = np.concatenate([[R_history[0]], R_history])

    # 相对轨迹
    rel_fix = traj_fix[0:2] - traj_fix[4:6]
    rel_rx = traj_rx[0:2] - traj_rx[4:6]
    dist_fix = np.linalg.norm(rel_fix, axis=0)
    dist_rx = np.linalg.norm(rel_rx, axis=0)

    # ── 5. 绘图 ──
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Trajectory Comparison: Optimal vs Fixed-R SDRE vs R(r) SDRE",
                 fontsize=13, fontweight="bold")

    # 5a. 2D 轨迹
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(x_opt[0], x_opt[1], "k-", lw=2, label="Optimal (CasADi)")
    ax1.plot(rel_fix[0], rel_fix[1], "b--", lw=1.5, label="Fixed R=1e13 SDRE")
    ax1.plot(rel_rx[0], rel_rx[1], "r-", lw=1.5, label="R(r) SDRE")
    ax1.scatter(*x_opt[:2, 0], c="g", s=60, zorder=5, marker="o")
    ax1.scatter(0, 0, c="r", s=60, zorder=5, marker="*")
    ax1.set_xlabel("x (km)"); ax1.set_ylabel("y (km)")
    ax1.set_title("2D Relative Trajectory")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.4); ax1.set_aspect("equal")

    # 5b. 距离 vs 时间
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.semilogy(t_opt/3600, dist_opt, "k-", lw=2, label="Optimal (CasADi)")
    ax2.semilogy(t_fix/3600, dist_fix, "b--", lw=1.5, label="Fixed R=1e13 SDRE")
    ax2.semilogy(t_rx/3600, dist_rx, "r-", lw=1.5, label="R(r) SDRE")
    ax2.set_xlabel("Time (h)"); ax2.set_ylabel("Relative Distance (km)")
    ax2.set_title("Distance vs Time")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.4)

    # T & E 标注
    for label, t_arr_, u_arr_, color in [
        ("Optimal", t_opt, u_opt, "k"),
        ("Fixed R", t_fix, u_fix, "b"),
        ("R(r)", t_rx, u_rx, "r"),
    ]:
        T_h = t_arr_[-1] / 3600
        E = float(np.trapezoid(np.sum(u_arr_**2, axis=0), t_arr_))
        ax2.annotate(f"{label}\nT={T_h:.1f}h, E={E:.2e}",
                     xy=(T_h, 5), fontsize=7, color=color,
                     bbox=dict(boxstyle="round,pad=0.3", fc="w", alpha=0.8))

    # 5c. 推力范数
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(t_opt/3600, np.linalg.norm(u_opt, axis=0)*1e6, "k-", lw=1.5)
    ax3.plot(t_fix/3600, np.linalg.norm(u_fix, axis=0)*1e6, "b--", lw=1.5)
    ax3.plot(t_rx/3600, np.linalg.norm(u_rx, axis=0)*1e6, "r-", lw=1.5)
    ax3.set_xlabel("Time (h)"); ax3.set_ylabel("||u|| (um/s^2)")
    ax3.set_title("Control Magnitude")
    ax3.legend(["Optimal", "Fixed R", "R(r)"], fontsize=8)
    ax3.grid(True, alpha=0.4)

    # 5d. R(r) 历史
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.semilogy(t_rx/3600, R_history, "r-", lw=1)
    ax4.set_xlabel("Time (h)"); ax4.set_ylabel("R")
    ax4.set_title("R(r) Evolution During Simulation")
    ax4.grid(True, alpha=0.4)

    # 5e. 控制 u_x
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(t_opt/3600, u_opt[0]*1e6, "k-", lw=1)
    ax5.plot(t_fix/3600, u_fix[0]*1e6, "b--", lw=1)
    ax5.plot(t_rx/3600, u_rx[0]*1e6, "r-", lw=1)
    ax5.set_xlabel("Time (h)"); ax5.set_ylabel("u_x (um/s^2)")
    ax5.set_title("Control u_x")
    ax5.legend(["Optimal", "Fixed R", "R(r)"], fontsize=8)
    ax5.grid(True, alpha=0.4)

    # 5f. 控制 u_y
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(t_opt/3600, u_opt[1]*1e6, "k-", lw=1)
    ax6.plot(t_fix/3600, u_fix[1]*1e6, "b--", lw=1)
    ax6.plot(t_rx/3600, u_rx[1]*1e6, "r-", lw=1)
    ax6.set_xlabel("Time (h)"); ax6.set_ylabel("u_y (um/s^2)")
    ax6.set_title("Control u_y")
    ax6.legend(["Optimal", "Fixed R", "R(r)"], fontsize=8)
    ax6.grid(True, alpha=0.4)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "trajectory_comparison.png", dpi=150)
    plt.close(fig)
    print(f"\nFigure saved: {OUT_DIR / 'trajectory_comparison.png'}")

    # 汇总
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for label, t_arr_, u_arr_ in [("Optimal", t_opt, u_opt),
                                    ("Fixed R SDRE", t_fix, u_fix),
                                    ("R(r) SDRE", t_rx, u_rx)]:
        T_h = t_arr_[-1]/3600
        E = float(np.trapezoid(np.sum(u_arr_**2, axis=0), t_arr_))
        peak = float(np.max(np.linalg.norm(u_arr_, axis=0)))
        print(f"  {label:20s}: T={T_h:6.2f}h  E={E:.3e}  peak_u={peak:.3e}")


if __name__ == "__main__":
    main()
