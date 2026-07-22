"""
E4: 信息基线与稳健性

对比: AON / AON+Range / 全状态反馈 / SDC-EKF 近似基线
敏感性: dt={10,5,2}s, R_ctrl={1e13, 1e14}
不施加推力限幅，只记录峰值加速度和 ΔV。
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
SENSOR_MODES = ["aon", "aon_range", "full_state"]  # 仅测角 / 测角+测距 / 全状态
DT_VALUES = [10, 5, 2]
R_CTRL_VALUES = [1e13, 1e14]
GEOMETRY_LABELS = ["radial", "alongtrack", "crosstrack"]
SIGMA_THETA_DEG = 0.008
SIGMA_RANGE_KM = 0.01  # 10 m 测距噪声
N_SEEDS = 30
MASTER_SEED = 123
T_ORBIT_FACTOR = 3.0
CAPTURE_DIST = 0.1
V_THRESH = 1e-4
SUSTAIN_DUR = 600.0

MU = 3.986e5; A_C = 15000.0; E_C = 0.5


# ── 工具 ────────────────────────────────────────────────────────────────────

def _make_initial_state(geometry: str, dist_km: float = 30.0):
    if geometry == "radial":
        pos = np.array([dist_km, 0.0, 0.0])
    elif geometry == "alongtrack":
        pos = np.array([0.0, dist_km, 0.0])
    elif geometry == "crosstrack":
        pos = np.array([0.0, 0.0, dist_km])
    else:
        raise ValueError(geometry)
    return np.concatenate([pos, np.zeros(3)]), np.zeros(6)


def _los_p0_fixed(x_rel0: np.ndarray) -> np.ndarray:
    r = x_rel0[:3]; rn = np.linalg.norm(r)
    if rn < 1e-12: return np.eye(6) * 0.1
    e_los = r / rn
    e2 = np.array([-e_los[1], e_los[0], 0.0])
    if np.linalg.norm(e2) < 1e-12: e2 = np.array([1.0, 0.0, 0.0])
    e2 /= np.linalg.norm(e2); e3 = np.cross(e_los, e2)
    R_mat = np.array([e_los, e2, e3])
    P_pos_los = np.diag([1.0**2, (30.0*0.01*DEG2RAD)**2, (30.0*0.01*DEG2RAD)**2])
    P = np.zeros((6,6)); P[:3,:3] = R_mat.T @ P_pos_los @ R_mat
    P[3:,3:] = np.eye(3) * (1e-3)**2
    return P


class StepRNG:
    def __init__(self, noise): self._n, self._i = noise, 0
    def multivariate_normal(self, m, c):
        if self._i >= len(self._n): return np.zeros(len(m))
        z = self._n[self._i]; self._i += 1; return z


# ── 主实验 ─────────────────────────────────────────────────────────────────

def run_e4(output_dir: Path | None = None):
    if output_dir is None:
        output_dir = DATA_DIR / "e4_baseline"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    orb = OrbitalDynamics(mu=MU, a_c=A_C, e_c=E_C)
    t_end_base = T_ORBIT_FACTOR * orb.T_orbit

    master_rng = np.random.default_rng(MASTER_SEED)
    max_steps = int(t_end_base / min(DT_VALUES)) + 2
    all_noise_2d = master_rng.standard_normal((N_SEEDS, max_steps, 2))
    all_noise_3d = master_rng.standard_normal((N_SEEDS, max_steps, 3))

    sigma_ang = SIGMA_THETA_DEG * DEG2RAD

    csv_path = output_dir / "e4_results.csv"
    fields = [
        "seed", "geometry", "sensor_mode", "filter_type", "dt", "R_ctrl",
        "T_FP", "v_FP", "T_soft", "T_sustain", "sustained",
        "exit_count", "total_delta_v", "peak_accel", "control_energy",
        "rmse_dist", "final_pos_err",
    ]

    total_combos = (len(GEOMETRY_LABELS) * len(SENSOR_MODES) * len(DT_VALUES)
                    * len(R_CTRL_VALUES) * N_SEEDS)
    count = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for geom in GEOMETRY_LABELS:
            X_p0, X_e0 = _make_initial_state(geom)
            x_rel0 = X_p0 - X_e0
            P0 = _los_p0_fixed(x_rel0)

            for dt in DT_VALUES:
                t_end = T_ORBIT_FACTOR * orb.T_orbit
                n_steps = int(t_end / dt) + 2
                Q_proc = np.zeros((6, 6))

                for R_ctrl_val in R_CTRL_VALUES:

                    for sensor_mode in SENSOR_MODES:
                        is_oracle = (sensor_mode == "full_state")
                        if is_oracle:
                            angles_only = False
                            # 全状态反馈：rng=None → 直接用真值，不做估计
                            for seed in range(N_SEEDS):
                                ctrl = SDREGameController(
                                    Q=np.eye(6), R=np.eye(3) * R_ctrl_val,
                                    gamma=np.sqrt(2),
                                )
                                nav = create_navigator(NAV_TYPE, x0=x_rel0, P0=P0, Q=Q_proc,
                                                       R=np.diag([1e-12, 1e-12, 1e-12]),
                                                       angles_only=False)
                                sim = EKFSDRESimulation(
                                    dynamics=orb, controller=ctrl, navigator=nav,
                                    X_p0=X_p0, X_e0=X_e0, nu0=0.0, dt=dt,
                                    are_interval=1, rng=None,
                                    passive_evader=True, early_stop=False,
                                    prediction_map="flow",
                                )
                                result = sim.run(t_end=t_end)
                                m = compute_metrics(result, dt=dt,
                                                    capture_dist=CAPTURE_DIST,
                                                    v_thresh=V_THRESH,
                                                    sustain_dur=SUSTAIN_DUR)
                                writer.writerow({
                                    "seed": seed, "geometry": geom,
                                    "sensor_mode": sensor_mode,
                                    "filter_type": "oracle",
                                    "dt": dt, "R_ctrl": R_ctrl_val,
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
                                })
                                count += 1
                                if count % 50 == 0:
                                    print(f"  [{count}] {geom} dt={dt} R={R_ctrl_val:.0e} {sensor_mode}")
                            continue

                        # ── AON / AON+Range（非全状态）────────────────
                        if sensor_mode == "aon":
                            angles_only = True
                            R_meas = np.diag([sigma_ang**2, sigma_ang**2])
                        elif sensor_mode == "aon_range":
                            angles_only = False
                            R_meas = np.diag([SIGMA_RANGE_KM**2, sigma_ang**2, sigma_ang**2])

                        noise_dim = R_meas.shape[0]
                        noise_pool = all_noise_3d if noise_dim == 3 else all_noise_2d

                        for seed in range(N_SEEDS):
                            ctrl = SDREGameController(
                                Q=np.eye(6), R=np.eye(3) * R_ctrl_val,
                                gamma=np.sqrt(2),
                            )
                            noise_scaled = noise_pool[seed, :n_steps, :noise_dim]
                            if sensor_mode == "full_state":
                                noise_scaled_eff = noise_scaled * 1e-8  # 极小噪声
                            else:
                                noise_scaled_eff = noise_scaled.copy()
                                if noise_dim >= 1:
                                    noise_scaled_eff[:, 0] *= (sigma_ang if angles_only else SIGMA_RANGE_KM)
                                if noise_dim >= 2:
                                    noise_scaled_eff[:, 1] *= sigma_ang
                                if noise_dim >= 3:
                                    noise_scaled_eff[:, 2] *= sigma_ang
                            # 简化：直接缩放
                            rng = StepRNG(noise_scaled_eff)

                            nav = create_navigator(NAV_TYPE, x0=x_rel0, P0=P0, Q=Q_proc, R=R_meas, angles_only=angles_only)

                            sim = EKFSDRESimulation(
                                dynamics=orb, controller=ctrl, navigator=nav,
                                X_p0=X_p0, X_e0=X_e0, nu0=0.0, dt=dt,
                                are_interval=1, rng=rng,
                                passive_evader=True, early_stop=False,
                                prediction_map="flow",
                            )
                            result = sim.run(t_end=t_end)
                            m = compute_metrics(result, dt=dt,
                                                capture_dist=CAPTURE_DIST,
                                                v_thresh=V_THRESH,
                                                sustain_dur=SUSTAIN_DUR)
                            writer.writerow({
                                "seed": seed, "geometry": geom,
                                "sensor_mode": sensor_mode,
                                "filter_type": NAV_TYPE,
                                "dt": dt, "R_ctrl": R_ctrl_val,
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
                            })
                            count += 1
                            if count % 50 == 0:
                                print(f"  [{count}] {geom} dt={dt} R={R_ctrl_val:.0e} {sensor_mode}")

    print(f"\nE4 结果已保存至 {csv_path}")
    return csv_path


if __name__ == "__main__":
    run_e4()






