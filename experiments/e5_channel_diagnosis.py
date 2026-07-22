"""
E5: 通道级反事实诊断

低/高两档噪声 × 固定 P0 × N=60 配对种子。
6 种控制状态配置 (A0-A5)，EKF 仍沿原估计路径运行（仅控制器输入不同）。

配置:
  A0: r_hat, v_hat, A_hat  (默认)
  A1: r_true, v_hat, A_hat  (替换位置)
  A2: r_hat, v_true, A_hat  (替换速度)
  A3: r_true, v_true, A_hat  (替换位置+速度)
  A4: r_hat, v_hat, A_true  (替换A矩阵)
  A5: r_true, v_true, A_true  (全真值)

输出: 森林图数据 (各配置对高低噪声差异的效应量)
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from aerospace.dynamics.nerm import OrbitalDynamics
from aerospace.control.sdre import SDREGameController
from aerospace.simulation.nerm_ekf_sdre import EKFSDRESimulation
from aerospace.simulation.metrics import compute_metrics
from aerospace.estimation.navigators import create_navigator
from aerospace.paths import DATA_DIR

DEG2RAD = np.pi / 180.0

# ── 配置 ────────────────────────────────────────────────────────────────────
NAV_TYPE = "jacobian"       # E0c-selected primary navigator
CHANNELS = {
    "A0": None,  # 默认: 全部 estimated
    "A1": {"pos": "true", "vel": "estimated", "A": "estimated"},
    "A2": {"pos": "estimated", "vel": "true", "A": "estimated"},
    "A3": {"pos": "true", "vel": "true", "A": "estimated"},
    "A4": {"pos": "estimated", "vel": "estimated", "A": "true"},
    "A5": {"pos": "true", "vel": "true", "A": "true"},
}

SIGMA_LOW_DEG = 0.001   # 低噪声
SIGMA_HIGH_DEG = 0.1    # 高噪声
GEOMETRY_LABELS = ["radial", "alongtrack", "crosstrack"]
N_SEEDS = 60
MASTER_SEED = 456
T_ORBIT_FACTOR = 3.0
CAPTURE_DIST = 0.1
V_THRESH = 1e-4
SUSTAIN_DUR = 600.0
DT = 10.0

MU = 3.986e5; A_C = 15000.0; E_C = 0.5


# ── 工具 ────────────────────────────────────────────────────────────────────

def _make_initial_state(geometry: str, dist_km: float = 30.0):
    if geometry == "radial": pos = np.array([dist_km, 0.0, 0.0])
    elif geometry == "alongtrack": pos = np.array([0.0, dist_km, 0.0])
    elif geometry == "crosstrack": pos = np.array([0.0, 0.0, dist_km])
    else: raise ValueError(geometry)
    return np.concatenate([pos, np.zeros(3)]), np.zeros(6)


def _los_p0_fixed(x_rel0: np.ndarray) -> np.ndarray:
    r = x_rel0[:3]; rn = np.linalg.norm(r)
    if rn < 1e-12: return np.eye(6) * 0.1
    e_los = r / rn
    e2 = np.array([-e_los[1], e_los[0], 0.0])
    if np.linalg.norm(e2) < 1e-12: e2 = np.array([1.0, 0.0, 0.0])
    e2 /= np.linalg.norm(e2); e3 = np.cross(e_los, e2)
    R_mat = np.array([e_los, e2, e3])
    P_pos = np.diag([1.0**2, (30.0*0.01*DEG2RAD)**2, (30.0*0.01*DEG2RAD)**2])
    P = np.zeros((6,6)); P[:3,:3] = R_mat.T @ P_pos @ R_mat
    P[3:,3:] = np.eye(3) * (1e-3)**2
    return P


class StepRNG:
    def __init__(self, noise): self._n, self._i = noise, 0
    def multivariate_normal(self, m, c):
        if self._i >= len(self._n): return np.zeros(len(m))
        z = self._n[self._i]; self._i += 1; return z


# ── 主实验 ─────────────────────────────────────────────────────────────────

def run_e5(output_dir: Path | None = None):
    if output_dir is None:
        output_dir = DATA_DIR / "e5_channels"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    orb = OrbitalDynamics(mu=MU, a_c=A_C, e_c=E_C)
    t_end = T_ORBIT_FACTOR * orb.T_orbit
    n_steps = int(t_end / DT) + 2

    # 预生成配对噪声（低/高噪声共用同一标准正态序列）
    master_rng = np.random.default_rng(MASTER_SEED)
    noise_std = master_rng.standard_normal((N_SEEDS, n_steps, 2))

    csv_path = output_dir / "e5_results.csv"
    fields = [
        "seed", "geometry", "sigma_theta_deg", "channel",
        "T_FP", "v_FP", "T_soft", "T_sustain", "sustained",
        "exit_count", "total_delta_v", "peak_accel", "control_energy",
        "rmse_dist", "final_pos_err", "est_dist_bias",
    ]

    total = len(GEOMETRY_LABELS) * 2 * len(CHANNELS) * N_SEEDS  # 2 noise levels
    count = 0
    print(f"E5: {total} 次仿真")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for geom in GEOMETRY_LABELS:
            X_p0, X_e0 = _make_initial_state(geom)
            x_rel0 = X_p0 - X_e0
            P0 = _los_p0_fixed(x_rel0)
            Q_proc = np.zeros((6, 6))

            for sigma_deg in [SIGMA_LOW_DEG, SIGMA_HIGH_DEG]:
                sigma_rad = max(sigma_deg * DEG2RAD, 1e-12)
                R_meas = np.diag([sigma_rad**2, sigma_rad**2])

                for channel_name, override in CHANNELS.items():
                    for seed in range(N_SEEDS):
                        ctrl = SDREGameController(
                            Q=np.eye(6), R=np.eye(3) * 1e13,
                            gamma=np.sqrt(2),
                        )
                        noise_scaled = noise_std[seed] * sigma_rad
                        rng = StepRNG(noise_scaled)

                        nav = create_navigator(NAV_TYPE, x0=x_rel0, P0=P0, Q=Q_proc, R=R_meas, angles_only=True)

                        sim = EKFSDRESimulation(
                            dynamics=orb, controller=ctrl, navigator=nav,
                            X_p0=X_p0, X_e0=X_e0, nu0=0.0, dt=DT,
                            are_interval=1, rng=rng,
                            passive_evader=True, early_stop=False,
                            prediction_map="flow",
                            ctrl_state_override=override,
                        )
                        result = sim.run(t_end=t_end)
                        m = compute_metrics(result, dt=DT,
                                            capture_dist=CAPTURE_DIST,
                                            v_thresh=V_THRESH,
                                            sustain_dur=SUSTAIN_DUR)
                        writer.writerow({
                            "seed": seed, "geometry": geom,
                            "sigma_theta_deg": sigma_deg,
                            "channel": channel_name,
                            "T_FP": m.T_FP or "", "v_FP": m.v_FP or "",
                            "T_soft": m.T_soft or "",
                            "T_sustain": m.T_sustain or "",
                            "sustained": m.sustained,
                            "exit_count": m.exit_count,
                            "total_delta_v": m.total_delta_v,
                            "peak_accel": m.peak_accel,
                            "control_energy": m.control_energy,
                            "rmse_dist": m.rmse_dist,
                            "final_pos_err": m.final_pos_err,
                            "est_dist_bias": m.est_dist_bias,
                        })
                        count += 1
                        if count % 100 == 0:
                            print(f"  [{count}/{total}]")

    print(f"\nE5 结果已保存至 {csv_path}")
    return csv_path


if __name__ == "__main__":
    run_e5()



