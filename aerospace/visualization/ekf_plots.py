"""
NERM + EKF + SDRE 仿真结果可视化

提供统一的绘图接口，支持单次仿真和对比仿真。
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aerospace.dynamics.nerm import OrbitalDynamics
from aerospace.simulation.nerm_ekf_sdre import EKFSDRESimResult


def lvlh_to_inertial(x_lvlh: np.ndarray, nu: float, r_c: float) -> np.ndarray:
    """将 LVLH 坐标转换到惯性系（以地心为原点）。"""
    r_ref_inertial = np.array([r_c * np.cos(nu), r_c * np.sin(nu), 0.0])
    cos_nu, sin_nu = np.cos(nu), np.sin(nu)
    R = np.array([
        [cos_nu, -sin_nu, 0],
        [sin_nu,  cos_nu, 0],
        [0,       0,      1]
    ])
    return r_ref_inertial + R @ x_lvlh


def plot_single_simulation(result: EKFSDRESimResult, orb: OrbitalDynamics,
                           title: str, out_path: str) -> None:
    """绘制单次仿真结果（6 子图布局）。"""
    hist_t = result.t
    hist_pos_p = result.states[0:3, :].T
    hist_pos_e = result.states[6:9, :].T
    hist_nu = result.states[12, :]
    hist_dist = result.dist_history
    hist_ekf_err = result.ekf_err_history

    # LVLH → 惯性系转换
    N = len(hist_t)
    pos_p_inertial = np.zeros_like(hist_pos_p)
    pos_e_inertial = np.zeros_like(hist_pos_e)
    for i in range(N):
        r_c, _, _ = orb.get_orbital_params(hist_nu[i])
        pos_p_inertial[i] = lvlh_to_inertial(hist_pos_p[i], hist_nu[i], r_c)
        pos_e_inertial[i] = lvlh_to_inertial(hist_pos_e[i], hist_nu[i], r_c)

    # 参考轨道
    nu_ref = np.linspace(0, 2 * np.pi, 360)
    ref_x = np.array([orb.get_orbital_params(n)[0] * np.cos(n) for n in nu_ref])
    ref_y = np.array([orb.get_orbital_params(n)[0] * np.sin(n) for n in nu_ref])

    R_earth = 6371.0
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # 1. 惯性系 3D
    ax0 = fig.add_subplot(2, 3, 1, projection="3d")
    u_sph = np.linspace(0, 2 * np.pi, 40)
    v_sph = np.linspace(0, np.pi, 20)
    xe = R_earth * np.outer(np.cos(u_sph), np.sin(v_sph))
    ye = R_earth * np.outer(np.sin(u_sph), np.sin(v_sph))
    ze = R_earth * np.outer(np.ones_like(u_sph), np.cos(v_sph))
    ax0.plot_surface(xe, ye, ze, color="deepskyblue", alpha=0.35, linewidth=0)
    ax0.plot(ref_x, ref_y, np.zeros(360), "gray", lw=0.8, ls="--", alpha=0.6, label="Ref")
    ax0.plot(pos_p_inertial[:, 0], pos_p_inertial[:, 1], pos_p_inertial[:, 2],
             "b-", lw=1.2, label="Pursuer")
    ax0.plot(pos_e_inertial[:, 0], pos_e_inertial[:, 1], pos_e_inertial[:, 2],
             "r-", lw=1.2, label="Evader")
    ax0.scatter(*pos_p_inertial[0], c="b", s=40, marker="o")
    ax0.scatter(*pos_e_inertial[0], c="r", s=40, marker="o")
    ax0.scatter(*pos_p_inertial[-1], c="b", s=60, marker="*")
    ax0.scatter(*pos_e_inertial[-1], c="r", s=60, marker="*")
    ax0.set_xlabel("X (km)"); ax0.set_ylabel("Y (km)"); ax0.set_zlabel("Z (km)")
    ax0.set_title("Inertial Frame 3D"); ax0.legend(fontsize=7)

    # 2. LVLH 3D
    ax1 = fig.add_subplot(2, 3, 2, projection="3d")
    ax1.plot(hist_pos_p[:, 0], hist_pos_p[:, 1], hist_pos_p[:, 2],
             "b-", lw=1.2, label="Pursuer")
    ax1.plot(hist_pos_e[:, 0], hist_pos_e[:, 1], hist_pos_e[:, 2],
             "r-", lw=1.2, label="Evader")
    ax1.scatter(*hist_pos_p[0], c="b", s=40, marker="o")
    ax1.scatter(*hist_pos_e[0], c="r", s=40, marker="o")
    ax1.scatter(*hist_pos_p[-1], c="b", s=60, marker="*")
    ax1.scatter(*hist_pos_e[-1], c="r", s=60, marker="*")
    ax1.set_xlabel("x (km)"); ax1.set_ylabel("y (km)"); ax1.set_zlabel("z (km)")
    ax1.set_title("3D LVLH Trajectory"); ax1.legend(fontsize=7)

    # 3. 相对距离
    ax2 = fig.add_subplot(2, 3, 3)
    ax2.plot(hist_t / 3600, hist_dist, "k-", lw=1.5)
    ax2.set_xlabel("Time (h)"); ax2.set_ylabel("Relative distance (km)")
    ax2.set_title("Pursuer-Evader Distance")
    ax2.grid(True, alpha=0.4); ax2.set_yscale("log")

    # 4. 惯性系 2D
    ax5 = fig.add_subplot(2, 3, 4)
    theta_c = np.linspace(0, 2 * np.pi, 360)
    ax5.fill(R_earth * np.cos(theta_c), R_earth * np.sin(theta_c),
             color="deepskyblue", alpha=0.4, label="Earth")
    ax5.plot(ref_x, ref_y, "gray", lw=0.8, ls="--", alpha=0.6, label="Ref")
    ax5.plot(pos_p_inertial[:, 0], pos_p_inertial[:, 1], "b-", lw=1.2, label="Pursuer")
    ax5.plot(pos_e_inertial[:, 0], pos_e_inertial[:, 1], "r-", lw=1.2, label="Evader")
    ax5.scatter(pos_p_inertial[0, 0], pos_p_inertial[0, 1], c="b", s=40, marker="o")
    ax5.scatter(pos_e_inertial[0, 0], pos_e_inertial[0, 1], c="r", s=40, marker="o")
    ax5.scatter(pos_p_inertial[-1, 0], pos_p_inertial[-1, 1], c="b", s=60, marker="*")
    ax5.scatter(pos_e_inertial[-1, 0], pos_e_inertial[-1, 1], c="r", s=60, marker="*")
    ax5.set_xlabel("X (km)"); ax5.set_ylabel("Y (km)")
    ax5.set_title("Orbital Plane (Inertial)")
    ax5.legend(fontsize=7); ax5.grid(True, alpha=0.3); ax5.set_aspect("equal")

    # 5. EKF 误差
    ax3 = fig.add_subplot(2, 3, 5)
    ax3.plot(hist_t / 3600, hist_ekf_err * 1000, "g-", lw=1.2)
    ax3.set_xlabel("Time (h)"); ax3.set_ylabel("EKF error (m)")
    ax3.set_title("EKF Estimation Error"); ax3.grid(True, alpha=0.4)

    # 6. LVLH x-y
    ax4 = fig.add_subplot(2, 3, 6)
    ax4.plot(hist_pos_p[:, 0], hist_pos_p[:, 1], "b-", lw=1.2, label="Pursuer")
    ax4.plot(hist_pos_e[:, 0], hist_pos_e[:, 1], "r-", lw=1.2, label="Evader")
    ax4.scatter(hist_pos_p[0, 0], hist_pos_p[0, 1], c="b", s=40, marker="o")
    ax4.scatter(hist_pos_e[0, 0], hist_pos_e[0, 1], c="r", s=40, marker="o")
    ax4.scatter(hist_pos_p[-1, 0], hist_pos_p[-1, 1], c="b", s=60, marker="*")
    ax4.scatter(hist_pos_e[-1, 0], hist_pos_e[-1, 1], c="r", s=60, marker="*")
    ax4.set_xlabel("x (km)"); ax4.set_ylabel("y (km)")
    ax4.set_title("x-y Projection (LVLH)")
    ax4.legend(fontsize=7); ax4.grid(True, alpha=0.4); ax4.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"图像已保存至 {out_path}")


def plot_comparison(result_noisy: EKFSDRESimResult, result_clean: EKFSDRESimResult,
                   orb: OrbitalDynamics, out_path: str) -> None:
    """并排对比：左列有噪声，右列无噪声（3 行 × 2 列）。"""
    # 参考轨道
    nu_ref = np.linspace(0, 2 * np.pi, 360)
    ref_x = np.array([orb.get_orbital_params(n)[0] * np.cos(n) for n in nu_ref])
    ref_y = np.array([orb.get_orbital_params(n)[0] * np.sin(n) for n in nu_ref])

    R_earth = 6371.0
    theta_c = np.linspace(0, 2 * np.pi, 360)

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle("NERM+EKF+SDRE: Noisy vs. Ideal Comparison",
                 fontsize=14, fontweight="bold")
    titles_col = ["With Measurement Noise (EKF)", "Ideal (No Noise)"]

    for col, result in enumerate([result_noisy, result_clean]):
        hist_t = result.t
        hist_pos_p = result.states[0:3, :].T
        hist_pos_e = result.states[6:9, :].T
        hist_nu = result.states[12, :]
        hist_dist = result.dist_history
        hist_ekf_err = result.ekf_err_history

        # LVLH → 惯性系
        N = len(hist_t)
        pos_p_in = np.zeros_like(hist_pos_p)
        pos_e_in = np.zeros_like(hist_pos_e)
        for i in range(N):
            r_c, _, _ = orb.get_orbital_params(hist_nu[i])
            pos_p_in[i] = lvlh_to_inertial(hist_pos_p[i], hist_nu[i], r_c)
            pos_e_in[i] = lvlh_to_inertial(hist_pos_e[i], hist_nu[i], r_c)

        # 行 0: 惯性系 2D
        ax = axes[0, col]
        ax.fill(R_earth * np.cos(theta_c), R_earth * np.sin(theta_c),
                color="deepskyblue", alpha=0.4, label="Earth")
        ax.plot(ref_x, ref_y, "gray", lw=0.8, ls="--", alpha=0.5, label="Ref orbit")
        ax.plot(pos_p_in[:, 0], pos_p_in[:, 1], "b-", lw=1.2, label="Pursuer")
        ax.plot(pos_e_in[:, 0], pos_e_in[:, 1], "r-", lw=1.2, label="Evader")
        ax.scatter(pos_p_in[0, 0], pos_p_in[0, 1], c="b", s=40, marker="o")
        ax.scatter(pos_e_in[0, 0], pos_e_in[0, 1], c="r", s=40, marker="o")
        ax.scatter(pos_p_in[-1, 0], pos_p_in[-1, 1], c="b", s=60, marker="*")
        ax.scatter(pos_e_in[-1, 0], pos_e_in[-1, 1], c="r", s=60, marker="*")
        ax.set_xlabel("X (km)"); ax.set_ylabel("Y (km)")
        ax.set_title(f"Orbital Plane — {titles_col[col]}")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_aspect("equal")

        # 行 1: 相对距离
        ax = axes[1, col]
        ax.plot(hist_t / 3600, hist_dist, "k-", lw=1.5)
        ax.set_xlabel("Time (h)"); ax.set_ylabel("Relative distance (km)")
        ax.set_title(f"Distance — {titles_col[col]}")
        ax.grid(True, alpha=0.4); ax.set_yscale("log")

        # 行 2: EKF 误差
        ax = axes[2, col]
        label = "EKF pos error" if col == 0 else "True rel pos error (≈0)"
        ax.plot(hist_t / 3600, hist_ekf_err * 1000, "g-", lw=1.2, label=label)
        ax.set_xlabel("Time (h)"); ax.set_ylabel("Error (m)")
        ax.set_title(f"Estimation Error — {titles_col[col]}")
        ax.grid(True, alpha=0.4); ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"对比图已保存至 {out_path}")
