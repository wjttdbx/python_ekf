"""Generate simulation figures for the patent measurement-mode embodiments.

The script intentionally keeps the embodiment parameters separate from main.py
so the figures and their source data remain reproducible and auditable.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aerospace.control.sdre import SDREGameController
from aerospace.dynamics.nerm import OrbitalDynamics
from aerospace.estimation.ekf import RelativeStateEKF
from aerospace.simulation.nerm_ekf_sdre import EKFSDRESimulation


DEG2RAD = np.pi / 180.0
OUT_DIR = ROOT / "outputs" / "EKF-SDRE-相对导航" / "仿真附图_原始Pekf_固定扰动_20260727"
MEASUREMENT_MODE = "angle_only"
FIGURE_START = 3
CAPTURE_SOURCE = "estimated"


def _configure_matplotlib() -> None:
    mpl.use("Agg")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Arial",
                "DejaVu Sans",
                "sans-serif",
            ],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8.5,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "lines.linewidth": 1.1,
        }
    )


def _save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix, kwargs in (
        (".png", {"dpi": 400}),
        (".pdf", {}),
        (".svg", {}),
    ):
        fig.savefig(OUT_DIR / f"{stem}{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def run_case() -> tuple:
    mu = 3.986e5
    a_c = 15000.0
    e_c = 0.5
    dt = 10.0

    orb = OrbitalDynamics(mu=mu, a_c=a_c, e_c=e_c)
    x_p0 = np.array([500.0, 500.0, 500.0, 0.01, 0.01, 0.01])
    x_e0 = np.zeros(6)
    x_true0 = x_p0 - x_e0

    sigma_ang = 0.008 * DEG2RAD
    initial_dist = float(np.linalg.norm(x_true0[:3]))
    sigma_pos = initial_dist * sigma_ang
    sigma_vel = 1.0 * sigma_ang
    p0 = np.diag([sigma_pos**2] * 3 + [sigma_vel**2] * 3)
    # Follow the repository's noisy-scenario convention: draw the initial
    # state error from the covariance represented by P0 with a fixed seed.
    initial_error = (
        np.random.default_rng(42).standard_normal(6) * np.sqrt(np.diag(p0))
    )
    x_est0 = x_true0 + initial_error
    q_proc = np.diag([5e-4, 5e-4, 5e-4, 5e-8, 5e-8, 5e-8])
    sigma_range = 0.01
    if MEASUREMENT_MODE == "angle_only":
        r_meas = np.diag([sigma_ang**2, sigma_ang**2])
    else:
        r_meas = np.diag([sigma_range**2, sigma_ang**2, sigma_ang**2])
    ekf = RelativeStateEKF(
        x0=x_est0,
        P0=p0,
        Q=q_proc,
        R=r_meas,
        angles_only=MEASUREMENT_MODE == "angle_only",
    )

    controller = SDREGameController(
        Q=np.eye(6),
        R=1e13 * np.eye(3),
        gamma=np.sqrt(2.0),
    )
    simulation = EKFSDRESimulation(
        dynamics=orb,
        controller=controller,
        ekf=ekf,
        X_p0=x_p0,
        X_e0=x_e0,
        nu0=0.0,
        dt=dt,
        are_interval=1,
        capture_dist=0.1,
        rng=np.random.default_rng(43),
        passive_evader=True,
        early_stop=True,
        capture_source=CAPTURE_SOURCE,
    )
    # Allow repeated elliptic-orbit passages; the simulation still terminates
    # immediately at the first true-distance crossing of the 100 m boundary.
    result = simulation.run(t_end=10.0 * orb.T_orbit)
    return result, orb, controller, initial_error


def compute_metrics(result, controller) -> dict:
    rel_pos = (result.states[0:3] - result.states[6:9]).T
    rel_vel = (result.states[3:6] - result.states[9:12]).T
    distance = np.linalg.norm(rel_pos, axis=1)
    estimated_distance = np.linalg.norm(result.x_est_history[0:3].T, axis=1)
    speed = np.linalg.norm(rel_vel, axis=1)
    control_norm = np.linalg.norm(result.u_p_history.T, axis=1)
    pos_err = np.linalg.norm(result.ekf_err_history[0:3].T, axis=1)
    vel_err = np.linalg.norm(result.ekf_err_history[3:6].T, axis=1)
    decision_distance = (
        estimated_distance if CAPTURE_SOURCE == "estimated" else distance
    )
    first = np.flatnonzero(decision_distance <= 0.1)
    idx = int(first[0]) if first.size else None
    true_first = np.flatnonzero(distance <= 0.1)
    true_idx = int(true_first[0]) if true_first.size else None

    return {
        "captured": bool(idx is not None),
        "capture_source": CAPTURE_SOURCE,
        "capture_time_s": float(result.t[idx]) if idx is not None else None,
        "capture_time_h": float(result.t[idx] / 3600.0) if idx is not None else None,
        "entry_speed_m_s": float(speed[idx] * 1000.0) if idx is not None else None,
        "true_distance_at_capture_m": (
            float(distance[idx] * 1000.0) if idx is not None else None
        ),
        "estimated_distance_at_capture_m": (
            float(estimated_distance[idx] * 1000.0) if idx is not None else None
        ),
        "true_entry_time_h": (
            float(result.t[true_idx] / 3600.0) if true_idx is not None else None
        ),
        "minimum_distance_m": float(np.min(distance) * 1000.0),
        "final_distance_m": float(distance[-1] * 1000.0),
        "peak_control_m_s2": float(np.max(control_norm) * 1000.0),
        "mean_control_m_s2": float(np.mean(control_norm) * 1000.0),
        "final_position_error_m": float(pos_err[-1] * 1000.0),
        "final_velocity_error_m_s": float(vel_err[-1] * 1000.0),
        "are_solve_steps": int(len(controller.step_times)),
        "are_fallback_count": int(controller._are_fallback_count),
    }


def plot_trajectory(result, metrics: dict) -> None:
    t_h = result.t / 3600.0
    rel_true = (result.states[0:3] - result.states[6:9]).T
    rel_est = result.x_est_history[0:3].T
    dist_true = np.linalg.norm(rel_true, axis=1)
    dist_est = np.linalg.norm(rel_est, axis=1)

    fig = plt.figure(figsize=(8.2, 3.45))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35])
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax3d.plot(*rel_true.T, color="black", label="真实轨迹")
    ax3d.plot(*rel_est.T, color="0.45", ls="--", label="EKF估计轨迹")
    ax3d.scatter(*rel_true[0], c="white", edgecolors="black", s=28, label="起点")
    ax3d.scatter(0, 0, 0, c="black", marker="x", s=30, label="目标点")
    ax3d.set_xlabel("$x$/km")
    ax3d.set_ylabel("$y$/km")
    ax3d.set_zlabel("$z$/km", labelpad=-7)
    ax3d.tick_params(pad=-1)
    ax3d.view_init(elev=22, azim=-55)
    ax3d.legend(fontsize=7, loc="best")
    ax3d.text2D(-0.08, 1.02, "(a)", transform=ax3d.transAxes, fontweight="bold")

    ax = fig.add_subplot(gs[0, 1])
    ax.semilogy(t_h, dist_true, color="black", label="真实相对距离")
    ax.semilogy(t_h, dist_est, color="0.45", ls="--", label="估计相对距离")
    threshold_label = (
        "估计距离判据 0.1 km"
        if CAPTURE_SOURCE == "estimated"
        else "真实距离判据 0.1 km"
    )
    ax.axhline(0.1, color="black", ls="-.", lw=0.9, label=threshold_label)
    if metrics["captured"]:
        tc = metrics["capture_time_h"]
        ax.axvline(tc, color="0.25", ls=":", lw=0.9)
        decision_curve = dist_est if CAPTURE_SOURCE == "estimated" else dist_true
        idx = int(np.flatnonzero(decision_curve <= 0.1)[0])
        ax.plot(tc, decision_curve[idx], marker="^", ms=5, color="black")
        ax.annotate(
            (
                f"$T_{{dec}}$={tc:.3f} h"
                if CAPTURE_SOURCE == "estimated"
                else f"$T_{{FP}}$={tc:.3f} h"
            ),
            xy=(tc, 0.1),
            xytext=(-68, 28),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "lw": 0.8},
        )
    ax.set_xlabel("时间 $t$/h")
    ax.set_ylabel("相对距离 $\\rho$/km")
    ax.grid(True, which="both", color="0.88", lw=0.5)
    ax.legend(fontsize=7, loc="best")
    ax.text(-0.12, 1.02, "(b)", transform=ax.transAxes, fontweight="bold")
    fig.subplots_adjust(left=0.02, right=0.985, bottom=0.17, top=0.98, wspace=0.34)
    _save_figure(fig, f"图{FIGURE_START}_相对交会轨迹与距离")


def plot_ekf(result) -> None:
    t_h = result.t / 3600.0
    err = result.ekf_err_history
    p_diag = np.maximum(result.P_diag_history, 0.0)
    pos_error_norm = np.linalg.norm(err[0:3], axis=0) * 1000.0
    vel_error_norm = np.linalg.norm(err[3:6], axis=0) * 1000.0
    pos_3sigma = 3.0 * np.sqrt(np.sum(p_diag[0:3], axis=0)) * 1000.0
    vel_3sigma = 3.0 * np.sqrt(np.sum(p_diag[3:6], axis=0)) * 1000.0

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    axes[0].semilogy(t_h, np.maximum(pos_error_norm, 1e-6), color="black", label="位置误差范数")
    axes[0].semilogy(t_h, np.maximum(pos_3sigma, 1e-6), color="0.5", ls="--",
                     label=r"$3\sqrt{\mathrm{tr}(P_{rr})}$")
    axes[0].set_xlabel("时间 $t$/h")
    axes[0].set_ylabel("位置估计误差范数/m")
    axes[0].grid(True, which="both", color="0.88", lw=0.5)
    axes[0].legend(fontsize=7)
    axes[0].text(0.01, 1.02, "(a)", transform=axes[0].transAxes, fontweight="bold")

    axes[1].semilogy(t_h, np.maximum(vel_error_norm, 1e-9), color="black", label="速度误差范数")
    axes[1].semilogy(t_h, np.maximum(vel_3sigma, 1e-9), color="0.5", ls="--",
                     label=r"$3\sqrt{\mathrm{tr}(P_{vv})}$")
    axes[1].set_xlabel("时间 $t$/h")
    axes[1].set_ylabel("速度估计误差范数/(m/s)")
    axes[1].grid(True, which="both", color="0.88", lw=0.5)
    axes[1].legend(fontsize=7)
    axes[1].text(0.01, 1.02, "(b)", transform=axes[1].transAxes, fontweight="bold")
    _save_figure(fig, f"图{FIGURE_START + 1}_EKF状态估计误差")


def plot_control(result, metrics: dict) -> None:
    t_h = result.t / 3600.0
    control = result.u_p_history * 1000.0
    norm = np.linalg.norm(control, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    for i, (name, ls) in enumerate(zip(("$u_x$", "$u_y$", "$u_z$"), ("-", "--", "-."))):
        axes[0].plot(t_h, control[i], color=str(0.12 + 0.25 * i), ls=ls, label=name)
    axes[0].set_xlabel("时间 $t$/h")
    axes[0].set_ylabel("控制加速度/(m/s$^2$)")
    axes[0].grid(True, color="0.88", lw=0.5)
    axes[0].legend(fontsize=7, ncol=3)
    axes[0].text(0.01, 1.02, "(a)", transform=axes[0].transAxes, fontweight="bold")

    axes[1].plot(t_h, norm, color="black")
    peak_i = int(np.argmax(norm))
    axes[1].plot(t_h[peak_i], norm[peak_i], marker="^", color="black", ms=5)
    axes[1].annotate(
        f"峰值 {metrics['peak_control_m_s2']:.4g} m/s$^2$",
        xy=(t_h[peak_i], norm[peak_i]),
        xytext=(28, -28),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "lw": 0.8},
    )
    axes[1].set_xlabel("时间 $t$/h")
    axes[1].set_ylabel("控制加速度范数/(m/s$^2$)")
    axes[1].grid(True, color="0.88", lw=0.5)
    axes[1].text(0.01, 1.02, "(b)", transform=axes[1].transAxes, fontweight="bold")
    _save_figure(fig, f"图{FIGURE_START + 2}_SDRE控制输入")


def export_source_data(result, metrics: dict, initial_error: np.ndarray) -> None:
    rel_true = (result.states[0:6] - result.states[6:12]).T
    rel_est = result.x_est_history.T
    control = result.u_p_history.T
    columns = [
        "time_s", "true_x_km", "true_y_km", "true_z_km",
        "true_vx_km_s", "true_vy_km_s", "true_vz_km_s",
        "est_x_km", "est_y_km", "est_z_km",
        "est_vx_km_s", "est_vy_km_s", "est_vz_km_s",
        "u_x_km_s2", "u_y_km_s2", "u_z_km_s2",
        "P_x", "P_y", "P_z", "P_vx", "P_vy", "P_vz",
    ]
    rows = np.column_stack(
        [result.t, rel_true, rel_est, control, result.P_diag_history.T]
    )
    with (OUT_DIR / "仿真源数据.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)

    metadata = {
        "parameters": {
            "mu_km3_s2": 3.986e5,
            "a_c_km": 15000.0,
            "e_c": 0.5,
            "dt_s": 10.0,
            "capture_distance_km": 0.1,
            "sigma_angle_deg": 0.008,
            "sigma_range_km": 0.01 if MEASUREMENT_MODE == "range_angle" else None,
            "measurement_mode": MEASUREMENT_MODE,
            "P_EKF_0_diag": [
                float((np.linalg.norm([500.0, 500.0, 500.0]) * 0.008 * DEG2RAD) ** 2)
            ] * 3 + [float((0.008 * DEG2RAD) ** 2)] * 3,
            "Q_control_diag": [1.0] * 6,
            "R_control_diag": [1e13] * 3,
            "Q_process_diag": [5e-4, 5e-4, 5e-4, 5e-8, 5e-8, 5e-8],
            "initial_estimation_error": initial_error.tolist(),
            "initial_error_random_seed": 42,
            "measurement_noise_random_seed": 43,
            "capture_source": CAPTURE_SOURCE,
            "are_interval": 1,
            "passive_target": True,
            "unconstrained_control": True,
        },
        "metrics": metrics,
    }
    (OUT_DIR / "仿真参数与结果.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    global OUT_DIR, MEASUREMENT_MODE, FIGURE_START, CAPTURE_SOURCE
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("angle_only", "range_angle"),
        default="angle_only",
    )
    parser.add_argument("--figure-start", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--capture-source",
        choices=("true", "estimated"),
        default="estimated",
    )
    args = parser.parse_args()
    MEASUREMENT_MODE = args.mode
    CAPTURE_SOURCE = args.capture_source
    FIGURE_START = args.figure_start or (3 if args.mode == "angle_only" else 6)
    if args.output_dir is not None:
        OUT_DIR = args.output_dir.resolve()
    else:
        mode_label = "仅测角" if args.mode == "angle_only" else "距离角度"
        OUT_DIR = (
            ROOT
            / "outputs"
            / "EKF-SDRE-相对导航"
            / f"仿真附图_{mode_label}_实施例参数_估计距离判据_20260727"
        )
    _configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result, _, controller, initial_error = run_case()
    metrics = compute_metrics(result, controller)
    export_source_data(result, metrics, initial_error)
    plot_trajectory(result, metrics)
    plot_ekf(result)
    plot_control(result, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
