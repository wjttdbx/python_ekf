"""
E3: 初始化去混杂

完全因子实验: σθ × P0{固定, 随σ缩放} × e0{零误差, 固定先验抽样}
LOS 坐标独立定义距离和角度先验不确定度。
每组 N=60 配对种子。
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
SIGMA_THETA_VALUES = [0.001, 0.008, 0.1]  # deg
GEOMETRY_LABELS = ["radial", "alongtrack", "crosstrack"]
P0_MODES = ["fixed", "scaled"]     # 固定先验 vs 随 σθ 缩放
E0_MODES = ["zero", "sampled"]     # 零初始误差 vs 从先验抽样
N_SEEDS = 60
MASTER_SEED = 99
T_ORBIT_FACTOR = 3.0
CAPTURE_DIST = 0.1
SUSTAIN_DUR = 600.0
V_THRESH = 1e-4
DT = 10.0

MU = 3.986e5; A_C = 15000.0; E_C = 0.5

# ── 工具 ────────────────────────────────────────────────────────────────────

def _seed_from_key(*args) -> int:
    """将任意可哈希对象映射到确定性 31 位种子。"""
    import zlib
    return zlib.adler32(repr(args).encode()) & 0x7fffffff


def _los_p0_fixed(x_rel0: np.ndarray) -> np.ndarray:
    """固定 LOS 先验（与 σθ 无关）。"""
    r = x_rel0[:3]
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-12:
        return np.eye(6) * 0.1

    e_los = r / r_norm
    e2 = np.array([-e_los[1], e_los[0], 0.0])
    if np.linalg.norm(e2) < 1e-12:
        e2 = np.array([1.0, 0.0, 0.0])
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e_los, e2)
    R_mat = np.array([e_los, e2, e3])

    # 沿 LOS: 1 km²，横向: (30km × 0.01°)²
    P_pos_los = np.diag([1.0**2, (30.0 * 0.01 * DEG2RAD)**2, (30.0 * 0.01 * DEG2RAD)**2])
    P_pos_lvlh = R_mat.T @ P_pos_los @ R_mat
    P = np.zeros((6, 6))
    P[:3, :3] = P_pos_lvlh
    P[3:, 3:] = np.eye(3) * (1e-3)**2
    return P


def _los_p0_scaled(x_rel0: np.ndarray, sigma_ang_rad: float) -> np.ndarray:
    """随 σθ 缩放的 LOS 先验（模拟旧 main.py 行为）。"""
    r_norm = np.linalg.norm(x_rel0[:3])
    sigma_pos = r_norm * sigma_ang_rad
    sigma_vel = 1.0 * sigma_ang_rad
    return np.diag([sigma_pos**2]*3 + [sigma_vel**2]*3)


def _make_initial_state(geometry: str, dist_km: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    if geometry == "radial":
        pos = np.array([dist_km, 0.0, 0.0])
    elif geometry == "alongtrack":
        pos = np.array([0.0, dist_km, 0.0])
    elif geometry == "crosstrack":
        pos = np.array([0.0, 0.0, dist_km])
    else:
        raise ValueError(geometry)
    vel = np.zeros(3)  # 零初始相对速度
    return np.concatenate([pos, vel]), np.zeros(6)


class StepRNG:
    """逐步读取预生成噪声的随机生成器。"""
    def __init__(self, noise: np.ndarray):
        self._noise = noise
        self._i = 0

    def multivariate_normal(self, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        if self._i >= len(self._noise):
            return np.zeros(len(mean))
        z = self._noise[self._i]
        self._i += 1
        return z


# ── 主实验 ─────────────────────────────────────────────────────────────────

def run_e3(output_dir: Path | None = None):
    if output_dir is None:
        output_dir = DATA_DIR / "e3_init"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    orb = OrbitalDynamics(mu=MU, a_c=A_C, e_c=E_C)
    t_end = T_ORBIT_FACTOR * orb.T_orbit
    n_steps = int(t_end / DT) + 2

    # 预生成噪声序列
    master_rng = np.random.default_rng(MASTER_SEED)
    all_noise = master_rng.standard_normal((N_SEEDS, n_steps, 2))

    csv_path = output_dir / "e3_results.csv"
    fields = [
        "seed", "geometry", "sigma_theta_deg", "P0_mode", "e0_mode",
        "T_FP", "v_FP", "T_soft", "T_sustain", "sustained",
        "exit_count", "total_delta_v", "peak_accel", "control_energy",
        "rmse_dist", "final_pos_err", "est_dist_bias",
    ]

    total = len(GEOMETRY_LABELS) * len(SIGMA_THETA_VALUES) * len(P0_MODES) * len(E0_MODES) * N_SEEDS
    count = 0
    print(f"E3: {total} 次仿真")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for geom in GEOMETRY_LABELS:
            X_p0, X_e0 = _make_initial_state(geom)
            x_rel0 = X_p0 - X_e0
            P0_fixed = _los_p0_fixed(x_rel0)

            for sigma_deg in SIGMA_THETA_VALUES:
                sigma_rad = max(sigma_deg * DEG2RAD, 1e-12)
                R_meas = np.diag([sigma_rad**2, sigma_rad**2])
                Q_proc = np.zeros((6, 6))

                for p0_mode in P0_MODES:
                    if p0_mode == "fixed":
                        P0 = P0_fixed.copy()
                    else:
                        P0 = _los_p0_scaled(x_rel0, sigma_rad)

                    for e0_mode in E0_MODES:
                        # 预生成初始误差样本
                        err_rng = np.random.default_rng(
                            _seed_from_key(geom, sigma_deg, p0_mode, e0_mode))
                        # Cholesky 抽样
                        try:
                            L = np.linalg.cholesky(P0_fixed if e0_mode == "sampled" else P0)
                        except np.linalg.LinAlgError:
                            L = np.linalg.cholesky(
                                (P0_fixed if e0_mode == "sampled" else P0) + np.eye(6)*1e-14)
                        err_samples = np.zeros((N_SEEDS, 6))
                        for s in range(N_SEEDS):
                            if e0_mode == "zero":
                                err_samples[s] = 0.0
                            else:
                                err_samples[s] = L @ err_rng.standard_normal(6)

                        for seed in range(N_SEEDS):
                            ctrl = SDREGameController(
                                Q=np.eye(6), R=np.eye(3) * 1e13,
                                gamma=np.sqrt(2),
                            )
                            noise_scaled = all_noise[seed] * sigma_rad
                            rng = StepRNG(noise_scaled)

                            x0_err = err_samples[seed] if e0_mode == "sampled" else None
                            x0_est = x_rel0 if x0_err is None else x_rel0 + x0_err

                            nav = create_navigator(NAV_TYPE, x0=x0_est, P0=P0, Q=Q_proc, R=R_meas, angles_only=True)

                            sim = EKFSDRESimulation(
                                dynamics=orb, controller=ctrl, navigator=nav,
                                X_p0=X_p0, X_e0=X_e0, nu0=0.0, dt=DT,
                                are_interval=1, rng=rng,
                                passive_evader=True, early_stop=False,
                                prediction_map="flow",
                            )
                            result = sim.run(t_end=t_end)
                            m = compute_metrics(result, dt=DT,
                                                capture_dist=CAPTURE_DIST,
                                                v_thresh=V_THRESH,
                                                sustain_dur=SUSTAIN_DUR)
                            writer.writerow({
                                "seed": seed, "geometry": geom,
                                "sigma_theta_deg": sigma_deg,
                                "P0_mode": p0_mode, "e0_mode": e0_mode,
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

    print(f"\nE3 结果已保存至 {csv_path}")
    return csv_path


if __name__ == "__main__":
    run_e3()



