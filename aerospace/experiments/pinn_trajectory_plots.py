"""
PINN 追逃博弈可视化脚本
=======================

分别对 2D 和 3D PINN 方法运行 SDRE 基线与 PINN 闭环仿真，
生成四组对比图：
  1. ECI 绝对轨迹
  2. 各方向相对距离分量 vs 时间
  3. 追方 u_p 各方向分量 vs 时间
  4. 博弈代价函数 J_p / J_e / J_total vs 时间 (PINN vs SDRE)

运行: uv run python -m aerospace.experiments.pinn_trajectory_plots
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
try:
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: F401
except Exception:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid

from aerospace.shared.coord_transform import (
    lvlh_to_eci, chief_eci_pos, draw_earth, plot_eci_trajectory,
)

# ── 2D imports ──
from aerospace.dynamics.nerm_2d import OrbitalDynamics2D
from aerospace.control.sdre_2d import SDREGameController2D
from aerospace.control.neural_2d import NeuralSDREController2D
from aerospace.simulation.nerm_sdre_2d import SDRESimulation2D

# ── 3D imports ──
from aerospace.dynamics.nerm import OrbitalDynamics
from aerospace.control.sdre import SDREGameController
from aerospace.control.neural import NeuralSDREController
from aerospace.simulation.nerm_sdre import SDRESimulation

# ── 常量 ──
MU = 3.986e5       # km^3/s^2
A_C = 15000.0       # km
E_C = 0.5
R_EARTH = 6371.0    # km

CHECKPOINT_2D = "checkpoints/sdre_pinn_2d/best_model.pt"
CHECKPOINT_3D = "checkpoints/sdre_pinn/best_model.pt"
OUTPUT_DIR = Path("outputs/pinn_trajectory_plots")

GAMMA = np.sqrt(2)


# ═══════════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════════

def _run_sim_2d(ctrl, Xp0, Xe0, nu0=0.0, dt=20.0):
    dyn = OrbitalDynamics2D(MU, A_C, E_C)
    sim = SDRESimulation2D(dyn, ctrl, Xp0, Xe0, nu0=nu0, dt=dt, log_interval=200)
    return sim.run()


def _run_sim_3d(ctrl, Xp0, Xe0, nu0=0.0, dt=20.0):
    dyn = OrbitalDynamics(MU, A_C, E_C)
    sim = SDRESimulation(dyn, ctrl, Xp0, Xe0, nu0=nu0, dt=dt, log_interval=200)
    return sim.run()


def _compute_cost_curves(result, Q, R, gamma, state_dim, ctrl_dim):
    """从仿真结果计算累积代价曲线 J_p, J_e, J_total。

    J_state(t)  = int_0^t  x_rel^T Q x_rel  dτ
    J_p_ctrl(t) = int_0^t  u_p^T R u_p  dτ
    J_e_ctrl(t) = int_0^t  γ^{-2} u_e^T R u_e  dτ

    J_p(t)    = J_state(t) + J_p_ctrl(t)
    J_e(t)    = J_e_ctrl(t)
    J_total(t)= J_p(t) - J_e(t)

    Returns (t_cost, J_p, J_e, J_total) — 长度 N-1 的数组。
    """
    t = result.t
    states = result.states
    u_p = result.u_p_history  # (ctrl_dim, N)
    u_e = result.u_e_history

    if state_dim == 4:
        x_rel = states[0:4] - states[4:8]  # (4, N)
    else:
        x_rel = states[0:6] - states[6:12]  # (6, N)

    N = x_rel.shape[1]
    n_u = u_p.shape[1]

    L_state = np.array([x_rel[:, k] @ Q @ x_rel[:, k] for k in range(N)])

    n = min(N, n_u)
    L_p_ctrl = np.array([u_p[:, k] @ R @ u_p[:, k] for k in range(n)])
    L_e_ctrl = np.array([gamma**(-2) * (u_e[:, k] @ R @ u_e[:, k]) for k in range(n)])

    L_state = L_state[:n]
    t_trunc = t[:n]

    J_state = cumulative_trapezoid(L_state, t_trunc, initial=0)
    J_p_ctrl = cumulative_trapezoid(L_p_ctrl, t_trunc, initial=0)
    J_e_ctrl = cumulative_trapezoid(L_e_ctrl, t_trunc, initial=0)

    J_p = J_state + J_p_ctrl
    J_e = J_e_ctrl
    J_total = J_p - J_e

    return t_trunc, J_p, J_e, J_total


# ═══════════════════════════════════════════════════════════════════
#  图 1: ECI 绝对轨迹
# ═══════════════════════════════════════════════════════════════════

def _plot_eci(result, dim_label, save_path, state_dim):
    """绘制 ECI 3D 绝对轨迹（追方 + 逃方 + 参考轨道 + 地球）。"""
    states = result.states

    if state_dim == 4:
        pos_p = np.vstack([states[0:2], np.zeros((1, states.shape[1]))])  # (3,N)
        pos_e = np.vstack([states[4:6], np.zeros((1, states.shape[1]))])
        nu = states[8]
    else:
        pos_p = states[0:3]
        pos_e = states[6:9]
        nu = states[12]

    pursuer_eci = lvlh_to_eci(pos_p, nu, A_C, E_C)
    evader_eci = lvlh_to_eci(pos_e, nu, A_C, E_C)
    chief_eci = chief_eci_pos(nu, A_C, E_C)

    plot_eci_trajectory(
        pursuer_eci, evader_eci, chief_eci,
        earth_radius=R_EARTH,
        title=f"{dim_label} PINN Pursuit-Evasion — ECI Trajectories",
        unit_label="km",
        save_path=str(save_path),
    )
    plt.close("all")
    print(f"  -> Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════
#  图 2: 各方向相对位置分量 vs 时间
# ═══════════════════════════════════════════════════════════════════

def _plot_relative_components(sdre_res, pinn_res, dim_label, save_path, state_dim):
    if state_dim == 4:
        n_ax = 2
        labels = ["Radial  Δx (km)", "Along-track  Δy (km)"]
        idx_p, idx_e = [0, 1], [4, 5]
    else:
        n_ax = 3
        labels = ["Radial  Δx (km)", "Along-track  Δy (km)", "Cross-track  Δz (km)"]
        idx_p, idx_e = [0, 1, 2], [6, 7, 8]

    fig, axes = plt.subplots(n_ax, 1, figsize=(12, 3.5 * n_ax), sharex=True)
    if n_ax == 1:
        axes = [axes]
    fig.suptitle(f"{dim_label} Relative Position Components — SDRE vs PINN", fontsize=14)

    t_s = sdre_res.t / 3600
    t_p = pinn_res.t / 3600

    for i in range(n_ax):
        ax = axes[i]
        dr_sdre = sdre_res.states[idx_p[i]] - sdre_res.states[idx_e[i]]
        dr_pinn = pinn_res.states[idx_p[i]] - pinn_res.states[idx_e[i]]
        ax.plot(t_s, dr_sdre, "b-", label="SDRE", alpha=0.8)
        ax.plot(t_p, dr_pinn, "r--", label="PINN", alpha=0.8)
        ax.set_ylabel(labels[i])
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (hours)")
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"  -> Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════
#  图 3: 追方 u_p 各方向分量 vs 时间
# ═══════════════════════════════════════════════════════════════════

def _plot_pursuer_control(sdre_res, pinn_res, dim_label, save_path, ctrl_dim):
    if ctrl_dim == 2:
        labels = ["u_px  radial (m/s²)", "u_py  along-track (m/s²)"]
    else:
        labels = ["u_px  radial (m/s²)", "u_py  along-track (m/s²)",
                  "u_pz  cross-track (m/s²)"]

    n_ax = ctrl_dim
    fig, axes = plt.subplots(n_ax, 1, figsize=(12, 3.5 * n_ax), sharex=True)
    if n_ax == 1:
        axes = [axes]
    fig.suptitle(f"{dim_label} Pursuer Control Components — SDRE vs PINN", fontsize=14)

    t_s = sdre_res.t / 3600
    t_p = pinn_res.t / 3600

    for i in range(n_ax):
        ax = axes[i]
        n_s = sdre_res.u_p_history.shape[1]
        n_p = pinn_res.u_p_history.shape[1]
        ax.plot(t_s[:n_s], sdre_res.u_p_history[i, :] * 1e3,
                "b-", label="SDRE", alpha=0.8)
        ax.plot(t_p[:n_p], pinn_res.u_p_history[i, :] * 1e3,
                "r--", label="PINN", alpha=0.8)
        ax.set_ylabel(labels[i])
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (hours)")
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"  -> Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════
#  图 4: 博弈代价函数 J 对比
# ═══════════════════════════════════════════════════════════════════

def _plot_cost_comparison(sdre_res, pinn_res, Q, R, gamma,
                          dim_label, save_path, state_dim, ctrl_dim):
    t_s, Jp_s, Je_s, Jt_s = _compute_cost_curves(sdre_res, Q, R, gamma,
                                                   state_dim, ctrl_dim)
    t_p, Jp_p, Je_p, Jt_p = _compute_cost_curves(pinn_res, Q, R, gamma,
                                                   state_dim, ctrl_dim)

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    fig.suptitle(f"{dim_label} Game Cost Functions — SDRE vs PINN", fontsize=14)

    t_s_h = t_s / 3600
    t_p_h = t_p / 3600

    # J_p
    ax = axes[0]
    ax.plot(t_s_h, Jp_s, "b-", label="SDRE", alpha=0.8)
    ax.plot(t_p_h, Jp_p, "r--", label="PINN", alpha=0.8)
    ax.set_ylabel(r"$J_p(t) = \int(x^TQx + u_p^TRu_p)\,d\tau$")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

    # J_e
    ax = axes[1]
    ax.plot(t_s_h, Je_s, "b-", label="SDRE", alpha=0.8)
    ax.plot(t_p_h, Je_p, "r--", label="PINN", alpha=0.8)
    ax.set_ylabel(r"$J_e(t) = \int \gamma^{-2} u_e^TRu_e\,d\tau$")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

    # J_total
    ax = axes[2]
    ax.plot(t_s_h, Jt_s, "b-", label="SDRE", alpha=0.8)
    ax.plot(t_p_h, Jt_p, "r--", label="PINN", alpha=0.8)
    ax.set_ylabel(r"$J(t) = J_p - J_e$")
    ax.set_xlabel("Time (hours)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"  -> Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════════

def _run_2d():
    print("\n" + "=" * 60)
    print("  2D PINN Trajectory Visualization")
    print("=" * 60)

    state_dim, ctrl_dim = 4, 2
    Q = np.eye(state_dim)
    R = np.eye(ctrl_dim) * 1e13

    Xp0 = np.array([500.0, 500.0, 0.01, 0.01])
    Xe0 = np.zeros(4)

    print("\n  Running SDRE baseline ...")
    sdre_ctrl = SDREGameController2D(Q, R, GAMMA)
    sdre_res = _run_sim_2d(sdre_ctrl, Xp0, Xe0)

    print("  Running PINN ...")
    pinn_ctrl = NeuralSDREController2D(CHECKPOINT_2D, device="cpu")
    pinn_res = _run_sim_2d(pinn_ctrl, Xp0, Xe0)

    _plot_eci(pinn_res, "2D", OUTPUT_DIR / "2d_eci_trajectory.png", state_dim)
    _plot_relative_components(sdre_res, pinn_res, "2D",
                              OUTPUT_DIR / "2d_relative_components.png", state_dim)
    _plot_pursuer_control(sdre_res, pinn_res, "2D",
                          OUTPUT_DIR / "2d_pursuer_control.png", ctrl_dim)
    _plot_cost_comparison(sdre_res, pinn_res, Q, R, GAMMA, "2D",
                          OUTPUT_DIR / "2d_cost_comparison.png",
                          state_dim, ctrl_dim)


def _run_3d():
    print("\n" + "=" * 60)
    print("  3D PINN Trajectory Visualization")
    print("=" * 60)

    state_dim, ctrl_dim = 6, 3
    Q = np.eye(state_dim)
    R = np.eye(ctrl_dim) * 1e13

    Xp0 = np.array([500.0, 500.0, 500.0, 0.01, 0.01, 0.01])
    Xe0 = np.zeros(6)

    print("\n  Running SDRE baseline ...")
    sdre_ctrl = SDREGameController(Q, R, GAMMA)
    sdre_res = _run_sim_3d(sdre_ctrl, Xp0, Xe0)

    print("  Running PINN ...")
    pinn_ctrl = NeuralSDREController(CHECKPOINT_3D, device="cpu")
    pinn_res = _run_sim_3d(pinn_ctrl, Xp0, Xe0)

    _plot_eci(pinn_res, "3D", OUTPUT_DIR / "3d_eci_trajectory.png", state_dim)
    _plot_relative_components(sdre_res, pinn_res, "3D",
                              OUTPUT_DIR / "3d_relative_components.png", state_dim)
    _plot_pursuer_control(sdre_res, pinn_res, "3D",
                          OUTPUT_DIR / "3d_pursuer_control.png", ctrl_dim)
    _plot_cost_comparison(sdre_res, pinn_res, Q, R, GAMMA, "3D",
                          OUTPUT_DIR / "3d_cost_comparison.png",
                          state_dim, ctrl_dim)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _run_2d()
    _run_3d()
    print(f"\n  All plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
