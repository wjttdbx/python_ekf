"""
NERM + EKF + SDRE 仿真结果可视化

每次仿真输出三张图：
  <base>_rel.png   — LVLH 相对运动轨迹 + 各分量时间历程 + 相对距离
  <base>_ctrl.png  — 推力分量 + 推力范数（追踪星 / 逃逸星）
  <base>_ekf.png   — EKF 误差 + 3σ 包络 + 新息（仅有噪声时有意义）
"""

import numpy as np
import matplotlib
import os

if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from aerospace.dynamics.nerm import OrbitalDynamics
from aerospace.simulation.nerm_ekf_sdre import EKFSDRESimResult

_C = ["tab:blue", "tab:orange", "tab:green"]
_POS = ["x", "y", "z"]
_VEL = ["vx", "vy", "vz"]
_CTRL = ["ux", "uy", "uz"]


# ── 内部工具 ──────────────────────────────────────────────────────────────────

def _sym_ylim(ax, margin: float = 0.1) -> None:
    """y 轴以 0 为中心，范围由数据 99 百分位决定，避免初始峰值撑开坐标轴。"""
    lines = ax.get_lines()
    if not lines:
        return
    vals = np.concatenate([ln.get_ydata() for ln in lines if len(ln.get_ydata()) > 0])
    if vals.size == 0:
        return
    lim = np.percentile(np.abs(vals), 99) * (1 + margin)
    if lim > 0:
        ax.set_ylim(-lim, lim)


def _equal_xy_lim(axes_list) -> None:
    """让多个子图共享相同的 x/y 范围（取所有轴的并集），用于投影图对齐。"""
    all_xl = [ax.get_xlim() for ax in axes_list]
    all_yl = [ax.get_ylim() for ax in axes_list]
    xmin = min(v[0] for v in all_xl); xmax = max(v[1] for v in all_xl)
    ymin = min(v[0] for v in all_yl); ymax = max(v[1] for v in all_yl)
    span = max(xmax - xmin, ymax - ymin) * 0.55
    xc = (xmin + xmax) / 2; yc = (ymin + ymax) / 2
    for ax in axes_list:
        ax.set_xlim(xc - span, xc + span)
        ax.set_ylim(yc - span, yc + span)


# ── 图 1：LVLH 相对运动 ───────────────────────────────────────────────────────

def plot_relative_motion(result: EKFSDRESimResult, title: str, out_path: str) -> None:
    """
    3×4 布局：
      行0: 3D轨迹(跨2列) | x-y投影 | y-z投影
      行1: 相对距离(对数) | 位置 x   | 位置 y  | 位置 z
      行2: 速度范数       | 速度 vx  | 速度 vy | 速度 vz
    单位：位置 km，速度 m/s
    """
    t_h  = result.t / 3600
    rel  = result.states[0:3, :] - result.states[6:9, :]          # km, (3,N)
    relv = (result.states[3:6, :] - result.states[9:12, :]) * 1000 # m/s, (3,N)
    rel_T = rel.T                                                   # (N,3)

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    gs = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    # ── 3D 轨迹
    ax3d = fig.add_subplot(gs[0, :2], projection="3d")
    ax3d.plot(rel_T[:, 0], rel_T[:, 1], rel_T[:, 2], "b-", lw=1.2)
    ax3d.scatter(*rel_T[0],  c="g", s=60, zorder=5, label="Start")
    ax3d.scatter(*rel_T[-1], c="r", s=60, marker="*", zorder=5, label="End")
    ax3d.set_xlabel("x (km)"); ax3d.set_ylabel("y (km)"); ax3d.set_zlabel("z (km)")
    ax3d.set_title("Relative Motion 3D (LVLH)"); ax3d.legend(fontsize=8)

    # ── x-y 投影
    ax_xy = fig.add_subplot(gs[0, 2])
    ax_xy.plot(rel_T[:, 0], rel_T[:, 1], "b-", lw=1.2)
    ax_xy.scatter(*rel_T[0, :2],  c="g", s=40)
    ax_xy.scatter(*rel_T[-1, :2], c="r", s=50, marker="*")
    ax_xy.set_xlabel("x (km)"); ax_xy.set_ylabel("y (km)")
    ax_xy.set_title("x-y  (radial / along-track)")
    ax_xy.grid(True, alpha=0.4); ax_xy.set_aspect("equal")

    # ── y-z 投影
    ax_yz = fig.add_subplot(gs[0, 3])
    ax_yz.plot(rel_T[:, 1], rel_T[:, 2], "b-", lw=1.2)
    ax_yz.scatter(*rel_T[0, 1:],  c="g", s=40)
    ax_yz.scatter(*rel_T[-1, 1:], c="r", s=50, marker="*")
    ax_yz.set_xlabel("y (km)"); ax_yz.set_ylabel("z (km)")
    ax_yz.set_title("y-z  (along-track / cross-track)")
    ax_yz.grid(True, alpha=0.4); ax_yz.set_aspect("equal")

    _equal_xy_lim([ax_xy, ax_yz])

    # ── 相对距离（对数）
    ax_d = fig.add_subplot(gs[1, 0])
    ax_d.semilogy(t_h, result.dist_history, "k-", lw=1.5)
    ax_d.set_xlabel("Time (h)"); ax_d.set_ylabel("km")
    ax_d.set_title("Relative Distance"); ax_d.grid(True, alpha=0.4)

    # ── 位置三分量（各自独立子图，单位 km）
    for i in range(3):
        ax = fig.add_subplot(gs[1, i + 1])
        ax.plot(t_h, rel[i], color=_C[i], lw=1.1)
        ax.axhline(0, color="k", lw=0.6, ls="--")
        ax.set_xlabel("Time (h)"); ax.set_ylabel("km")
        ax.set_title(f"Relative position {_POS[i]}")
        ax.grid(True, alpha=0.4)

    # ── 速度范数
    ax_vn = fig.add_subplot(gs[2, 0])
    ax_vn.plot(t_h, np.linalg.norm(relv, axis=0), "k-", lw=1.3)
    ax_vn.set_xlabel("Time (h)"); ax_vn.set_ylabel("m/s")
    ax_vn.set_title("Relative Speed ||Δv||")
    ax_vn.grid(True, alpha=0.4)

    # ── 速度三分量（各自独立子图，单位 m/s）
    for i in range(3):
        ax = fig.add_subplot(gs[2, i + 1])
        ax.plot(t_h, relv[i], color=_C[i], lw=1.1)
        ax.axhline(0, color="k", lw=0.6, ls="--")
        ax.set_xlabel("Time (h)"); ax.set_ylabel("m/s")
        ax.set_title(f"Relative velocity {_VEL[i]}")
        ax.grid(True, alpha=0.4)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    #plt.close()
    plt.show()
    print(f"图像已保存至 {out_path}")


# ── 图 2：控制推力 ────────────────────────────────────────────────────────────

def plot_control_history(result: EKFSDRESimResult, title: str, out_path: str) -> None:
    """
    3×2 布局：左列追踪星，右列逃逸星。
    行0-2: ux/uy/uz 分量（y 轴用 99 百分位裁剪）
    最后一行额外加推力范数对比。
    """
    t_h = result.t / 3600
    u_p = result.u_p_history * 1e6   # km/s²→μm/s²
    u_e = result.u_e_history * 1e6
    norm_p = np.linalg.norm(u_p, axis=0)
    norm_e = np.linalg.norm(u_e, axis=0)

    fig, axes = plt.subplots(4, 2, figsize=(14, 14), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for i in range(3):
        for col, (u, lbl, color) in enumerate([(u_p, "Pursuer", "tab:blue"),
                                                (u_e, "Evader",  "tab:red")]):
            ax = axes[i, col]
            ax.plot(t_h, u[i], color=color, lw=1.1)
            ax.set_ylabel("μm/s²")
            ax.set_title(f"{lbl} thrust {_CTRL[i]}")
            ax.grid(True, alpha=0.4)
            _sym_ylim(ax)

    # 推力范数
    axes[3, 0].plot(t_h, norm_p, "tab:blue", lw=1.3)
    axes[3, 0].set_xlabel("Time (h)"); axes[3, 0].set_ylabel("μm/s²")
    axes[3, 0].set_title("Pursuer ||u||"); axes[3, 0].grid(True, alpha=0.4)

    axes[3, 1].plot(t_h, norm_e, "tab:red", lw=1.3)
    axes[3, 1].set_xlabel("Time (h)"); axes[3, 1].set_ylabel("μm/s²")
    axes[3, 1].set_title("Evader ||u||"); axes[3, 1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    #plt.close()
    plt.show()
    print(f"图像已保存至 {out_path}")


# ── 图 3：EKF 性能 ────────────────────────────────────────────────────────────

def plot_ekf_performance(result: EKFSDRESimResult, title: str, out_path: str) -> None:
    """
    3×3 布局：
      行0: 位置误差 x/y/z + 3σ 包络（y 轴 99 百分位裁剪）
      行1: 速度误差 vx/vy/vz + 3σ 包络
      行2: 新息 δρ / δaz / δel
    """
    t_h = result.t / 3600
    err_pos = result.ekf_err_history[0:3, :] * 1000        # km→m
    err_vel = result.ekf_err_history[3:6, :] * 1000        # km/s→m/s
    std_pos = np.sqrt(np.maximum(result.P_diag_history[0:3, :], 0)) * 1000
    std_vel = np.sqrt(np.maximum(result.P_diag_history[3:6, :], 0)) * 1000

    innov = result.innov_history.copy()
    n_innov = innov.shape[0]  # 2 (angle_only) 或 3 (range_angle)

    if n_innov == 2:
        innov[0:2] *= 1000    # rad→mrad
        innov_labels = ["δaz", "δel"]
        innov_units  = ["mrad", "mrad"]
    else:
        innov[0]   *= 1000    # km→m
        innov[1:3] *= 1000    # rad→mrad
        innov_labels = ["δρ", "δaz", "δel"]
        innov_units  = ["m",  "mrad", "mrad"]

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    gs = GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.35)

    t_capture = t_h[-1] if result.captured else None

    for i in range(3):
        # 位置误差
        ax = fig.add_subplot(gs[0, i])
        lim_p = np.percentile(np.abs(err_pos[i]), 99) * 1.3 or 1.0
        ax.set_ylim(-lim_p, lim_p)
        ax.fill_between(t_h, -3 * std_pos[i], 3 * std_pos[i],
                        alpha=0.22, color=_C[i], label="3σ bound")
        ax.plot(t_h, err_pos[i], color=_C[i], lw=0.9, label="error")
        ax.axhline(0, color="k", lw=0.6, ls="--")
        if t_capture:
            ax.axvline(t_capture, color="gray", lw=0.8, ls=":", label="capture")
        ax.set_xlabel("Time (h)"); ax.set_ylabel("m")
        ax.set_title(f"Pos error {_POS[i]}")
        ax.grid(True, alpha=0.4); ax.legend(fontsize=7, loc="upper right")

        # 速度误差
        ax = fig.add_subplot(gs[1, i])
        lim_v = np.percentile(np.abs(err_vel[i]), 99) * 1.3 or 1.0
        ax.set_ylim(-lim_v, lim_v)
        ax.fill_between(t_h, -3 * std_vel[i], 3 * std_vel[i],
                        alpha=0.22, color=_C[i], label="3σ bound")
        ax.plot(t_h, err_vel[i], color=_C[i], lw=0.9, label="error")
        ax.axhline(0, color="k", lw=0.6, ls="--")
        if t_capture:
            ax.axvline(t_capture, color="gray", lw=0.8, ls=":")
        ax.set_xlabel("Time (h)"); ax.set_ylabel("m/s")
        ax.set_title(f"Vel error {_VEL[i]}")
        ax.grid(True, alpha=0.4); ax.legend(fontsize=7, loc="upper right")

        # 新息（仅绘制存在的通道）
        ax = fig.add_subplot(gs[2, i])
        if i < n_innov:
            lim_z = np.percentile(np.abs(innov[i]), 99) * 1.3 or 1.0
            ax.set_ylim(-lim_z, lim_z)
            ax.plot(t_h, innov[i], color=_C[i], lw=0.8, alpha=0.85)
            ax.axhline(0, color="k", lw=0.6, ls="--")
            if t_capture:
                ax.axvline(t_capture, color="gray", lw=0.8, ls=":", label="capture")
                ax.legend(fontsize=7, loc="upper right")
            ax.set_xlabel("Time (h)"); ax.set_ylabel(innov_units[i])
            ax.set_title(f"Innovation {innov_labels[i]}")
        else:
            ax.set_visible(False)
        ax.grid(True, alpha=0.4)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    #plt.close()
    plt.show()
    print(f"图像已保存至 {out_path}")


# ── 公共接口 ──────────────────────────────────────────────────────────────────

def plot_single_simulation(result: EKFSDRESimResult, orb: OrbitalDynamics,
                           title: str, out_path: str) -> None:
    """生成三张图：相对运动、控制推力、EKF 性能。"""
    base = out_path.replace(".png", "")
    plot_relative_motion(result, title + " — Relative Motion", base + "_rel.png")
    plot_control_history(result, title + " — Control",         base + "_ctrl.png")
    plot_ekf_performance(result, title + " — EKF Performance", base + "_ekf.png")


def plot_comparison(result_noisy: EKFSDRESimResult, result_clean: EKFSDRESimResult,
                    orb: OrbitalDynamics, out_path: str) -> None:
    """有噪声 vs 无噪声：相对距离 + 位置估计误差范数对比（2×2）。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("NERM+EKF+SDRE: Noisy vs. Ideal", fontsize=13, fontweight="bold")

    for col, (res, lbl) in enumerate([(result_noisy, "With Noise (EKF)"),
                                       (result_clean, "Ideal (No Noise)")]):
        t_h = res.t / 3600

        ax = axes[0, col]
        ax.semilogy(t_h, res.dist_history, "k-", lw=1.5)
        ax.set_xlabel("Time (h)"); ax.set_ylabel("km")
        ax.set_title(f"Relative Distance — {lbl}"); ax.grid(True, alpha=0.4)

        ax = axes[1, col]
        pos_err = np.linalg.norm(res.ekf_err_history[0:3, :], axis=0) * 1000
        ax.plot(t_h, pos_err, color="tab:green", lw=1.0)
        ax.set_xlabel("Time (h)"); ax.set_ylabel("m")
        ax.set_title(f"Position Estimation Error ||Δr|| — {lbl}")
        ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    #plt.close()
    plt.show()
    print(f"对比图已保存至 {out_path}")

