"""
E1+E2: 场景资格审查 + Go/No-Go 复验

E1: 全状态反馈 + 无噪声 → 验证三类几何可达性（通过条件：全部达到软到达）
E2: 3几何 × 3σθ × 30配对种子 × 全时域 → speed-softness 是否存在

配对噪声：每个种子预先生成标准正态序列，不同 σθ 乘以同一序列。
固定 LOS 先验 P0（与 σθ 无关）。
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aerospace.dynamics.nerm import OrbitalDynamics
from aerospace.control.sdre import SDREGameController
from aerospace.simulation.nerm_ekf_sdre import EKFSDRESimulation
from aerospace.simulation.metrics import compute_metrics, CaptureMetrics
from aerospace.estimation.navigators import create_navigator
from aerospace.paths import DATA_DIR, ensure_figures_dir

warnings.filterwarnings("ignore")

DEG2RAD = np.pi / 180.0


def seed_from_str(s: str) -> int:
    """将字符串映射到确定性的 31 位正整数种子。"""
    import zlib
    return zlib.adler32(s.encode()) & 0x7fffffff

# ── 实验配置 ────────────────────────────────────────────────────────────────
NAV_TYPE = "jacobian"       # E0c-selected primary navigator
SIGMA_THETA_VALUES = [0.001, 0.008, 0.1]      # deg
GEOMETRY_LABELS = ["radial", "alongtrack", "crosstrack"]
N_SEEDS_E1 = 3        # E1: 少量种子验证可达性
N_SEEDS_E2 = 30       # E2: 正式复验
N_SEEDS_E3_PREP = 60  # E3-E5 预分配种子数（本脚本不跑但预生成噪声）
MASTER_SEED = 42
T_ORBIT_FACTOR = 3.0  # 公共时域 = 3 × T_orbit
CAPTURE_DIST = 0.1    # km
SUSTAIN_DUR = 600.0   # s
V_THRESH = 1e-4       # 0.1 m/s = 1e-4 km/s

# 轨道参数
MU = 3.986e5
A_C = 15000.0
E_C = 0.5

# LOS 先验：位置 1km LOS不确定度，角度 0.01° 横向不确定度，速度 1m/s
P0_POS_LOS = np.array([1.0**2, (30.0 * 0.01 * DEG2RAD)**2, (30.0 * 0.01 * DEG2RAD)**2])
P0_VEL = (1e-3) ** 2  # (km/s)²


# ── 几何构造 ────────────────────────────────────────────────────────────────

def make_initial_state(geometry: str, dist_km: float = 30.0,
                       vel_scale: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """构造三类初始几何的相对状态（追踪星相对逃逸星）。

    - radial:      位置在 x 方向（径向）
    - alongtrack:  位置在 y 方向（沿迹）
    - crosstrack:  位置在 z 方向（法向）

    Returns (X_p0, X_e0)
    """
    if geometry == "radial":
        pos = np.array([dist_km, 0.0, 0.0])
    elif geometry == "alongtrack":
        pos = np.array([0.0, dist_km, 0.0])
    elif geometry == "crosstrack":
        pos = np.array([0.0, 0.0, dist_km])
    else:
        raise ValueError(f"Unknown geometry: {geometry}")

    vel = np.random.default_rng(seed_from_str(geometry)).normal(0, vel_scale, 3)
    X_p0 = np.concatenate([pos, vel])
    X_e0 = np.zeros(6)
    return X_p0, X_e0


def los_p0(x_rel0: np.ndarray, sigma_ang_rad: float) -> np.ndarray:
    """在 LOS 坐标系中定义初始协方差，再旋转到 LVLH。

    LOS 坐标系：x_los = 沿视线方向，y_los, z_los 在垂直平面内。
    """
    r = x_rel0[:3]
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-12:
        return np.diag([P0_POS_LOS[0]]*3 + [P0_VEL]*3)

    e_los = r / r_norm
    # 构造正交基
    e2 = np.array([-e_los[1], e_los[0], 0.0])
    if np.linalg.norm(e2) < 1e-12:
        e2 = np.array([1.0, 0.0, 0.0])
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e_los, e2)

    R_lvlh_to_los = np.array([e_los, e2, e3])  # (3,3)
    P_los_pos = np.diag([
        P0_POS_LOS[0],
        (r_norm * sigma_ang_rad * 5)**2,  # 横向 = 5× 角不确定度 × 距离（保守）
        (r_norm * sigma_ang_rad * 5)**2,
    ])
    P_lvlh_pos = R_lvlh_to_los.T @ P_los_pos @ R_lvlh_to_los
    P_lvlh_vel = np.eye(3) * P0_VEL
    P = np.zeros((6, 6))
    P[:3, :3] = P_lvlh_pos
    P[3:, 3:] = P_lvlh_vel
    return P


# ── 配对噪声生成 ────────────────────────────────────────────────────────────

def generate_paired_noise_sequences(n_seeds: int, n_steps: int,
                                     m_dim: int, master_seed: int = MASTER_SEED
                                     ) -> np.ndarray:
    """为所有种子预先生成标准正态噪声序列 (n_seeds, n_steps, m_dim)。"""
    rng = np.random.default_rng(master_seed)
    return rng.standard_normal((n_seeds, n_steps, m_dim))


# ── 单次 E2 运行 ────────────────────────────────────────────────────────────

def run_e2_single(orb: OrbitalDynamics, ctrl: SDREGameController,
                  X_p0: np.ndarray, X_e0: np.ndarray,
                  nu0: float, dt: float, t_end: float,
                  sigma_ang_rad: float,
                  noise_seq: np.ndarray,
                  P0: np.ndarray,
                  x0_err: np.ndarray | None = None,
                  ) -> CaptureMetrics:
    """运行单次全时域仿真并返回指标。"""
    sigma_ang_eff = max(sigma_ang_rad, 1e-12)
    R_meas = np.diag([sigma_ang_eff**2, sigma_ang_eff**2])
    Q_proc = np.zeros((6, 6))

    x_true_rel0 = X_p0 - X_e0
    if x0_err is not None:
        x_est0 = x_true_rel0 + x0_err
    else:
        x_est0 = x_true_rel0.copy()

    nav = create_navigator(NAV_TYPE, x0=x_est0, P0=P0, Q=Q_proc, R=R_meas, angles_only=True)

    # 预生成带 σ 缩放的噪声序列，存入固定数组供仿真逐步读取
    n_steps_expected = int(t_end / dt) + 2
    noise_scaled = np.zeros((n_steps_expected, R_meas.shape[0]))
    for i in range(min(len(noise_seq), n_steps_expected)):
        noise_scaled[i] = noise_seq[i] * sigma_ang_eff

    # 用固定噪声的随机生成器
    rng = FixedNoiseRNG(noise_scaled, sigma_ang_eff, R_meas.shape[0])

    sim = EKFSDRESimulation(
        dynamics=orb, controller=ctrl, navigator=nav,
        X_p0=X_p0, X_e0=X_e0, nu0=nu0, dt=dt,
        are_interval=1, rng=rng,
        passive_evader=True, early_stop=False,
        prediction_map="flow",
    )
    result = sim.run(t_end=t_end)
    metrics = compute_metrics(result, dt=dt, capture_dist=CAPTURE_DIST,
                              v_thresh=V_THRESH, sustain_dur=SUSTAIN_DUR)
    metrics.are_fallback_count = ctrl._are_fallback_count
    if ctrl.step_times:
        metrics.controller_mean_step_ms = float(np.mean(ctrl.step_times) * 1000.0)
    return metrics


class FixedNoiseRNG:
    """固定噪声序列读取器，实现 multivariate_normal 接口。"""

    def __init__(self, noise_array: np.ndarray, sigma: float, m_dim: int):
        self._noise = noise_array
        self._sigma = sigma
        self._m_dim = m_dim
        self._idx = 0

    def multivariate_normal(self, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        if self._idx >= len(self._noise):
            return np.zeros(self._m_dim)
        z = self._noise[self._idx]
        self._idx += 1
        return z


# ── E1: 场景资格审查 ────────────────────────────────────────────────────────

def run_e1(output_dir: Path):
    """全状态反馈 + 无噪声，验证三类几何可达性。"""
    print("=" * 60)
    print("E1: 场景资格审查（全状态反馈 + 无噪声）")
    print("=" * 60)

    orb = OrbitalDynamics(mu=MU, a_c=A_C, e_c=E_C)
    t_end = T_ORBIT_FACTOR * orb.T_orbit

    results = {}
    for geom in GEOMETRY_LABELS:
        ctrl = SDREGameController(Q=np.eye(6), R=np.eye(3) * 1e13, gamma=np.sqrt(2))
        print(f"\n  Geometry: {geom}")
        X_p0, X_e0 = make_initial_state(geom, dist_km=30.0)
        nu0 = 0.0

        # 全状态反馈仿真（rng=None → 直接用真值）
        sim = EKFSDRESimulation(
            dynamics=orb, controller=ctrl,
            navigator=create_navigator(
                "sdc", x0=X_p0 - X_e0, P0=np.eye(6),
                Q=np.zeros((6, 6)), R=np.diag([1e-8, 1e-8]),
                angles_only=True),
            X_p0=X_p0, X_e0=X_e0, nu0=nu0, dt=10.0,
            are_interval=1, rng=None,
            passive_evader=True, early_stop=False,
            prediction_map="flow",
        )
        result = sim.run(t_end=t_end)
        m = compute_metrics(result, dt=10.0, capture_dist=CAPTURE_DIST,
                            v_thresh=V_THRESH, sustain_dur=SUSTAIN_DUR)
        results[geom] = {
            "T_FP": m.T_FP,
            "v_FP": m.v_FP,
            "T_soft": m.T_soft,
            "T_sustain": m.T_sustain,
            "sustained": m.sustained,
            "total_delta_v": m.total_delta_v,
            "peak_accel": m.peak_accel,
            "min_dist": m.min_dist_after_FP,
        }
        status = "PASS" if m.sustained else "FAIL"
        print(f"    T_FP={m.T_FP}, T_soft={m.T_soft}, sustained={m.sustained} → {status}")

    passed = all(r["sustained"] for r in results.values())
    print(f"\n  E1 通过: {passed}")
    with open(output_dir / "e1_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return passed


# ── E2: Go/No-Go ────────────────────────────────────────────────────────────

def run_e2(output_dir: Path):
    """3几何 × 3σθ × 30种子 = 270次全时域仿真。"""
    print("\n" + "=" * 60)
    print("E2: Go/No-Go 复验")
    print("=" * 60)

    orb = OrbitalDynamics(mu=MU, a_c=A_C, e_c=E_C)
    t_end = T_ORBIT_FACTOR * orb.T_orbit
    dt = 10.0
    n_steps = int(t_end / dt) + 2
    m_dim = 2  # angles only

    # 预生成配对噪声（跨 σθ 的配对由同一标准正态 × 不同 σ 实现）
    noise_seqs = generate_paired_noise_sequences(N_SEEDS_E2, n_steps, m_dim)

    # LOS 先验 P0（固定，不随 σθ 变化）
    sigma_ref = 0.008 * DEG2RAD  # 参考角度用于定义 P0 横向不确定度

    csv_path = output_dir / "e2_results.csv"
    fieldnames = [
        "seed", "geometry", "sigma_theta_deg",
        "T_FP", "v_FP", "radial_closure_rate_FP",
        "T_soft", "v_at_soft", "T_sustain", "sustained",
        "max_dist_after_FP", "min_dist_after_FP", "exit_count",
        "time_in_sphere_frac",
        "total_delta_v", "peak_accel", "control_energy",
        "rmse_pos_x", "rmse_pos_y", "rmse_pos_z",
        "rmse_dist", "final_pos_err", "final_vel_err",
        "est_dist_bias", "est_radial_vel_bias",
        "nis_mean", "nees_mean", "are_fallback_count", "controller_mean_step_ms",
    ]

    total = len(GEOMETRY_LABELS) * len(SIGMA_THETA_VALUES) * N_SEEDS_E2
    count = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for geom in GEOMETRY_LABELS:
            X_p0, X_e0 = make_initial_state(geom, dist_km=30.0)
            nu0 = 0.0
            P0 = los_p0(X_p0 - X_e0, sigma_ref)

            for sigma_deg in SIGMA_THETA_VALUES:
                sigma_rad = sigma_deg * DEG2RAD

                for seed in range(N_SEEDS_E2):
                    ctrl = SDREGameController(
                        Q=np.eye(6), R=np.eye(3) * 1e13, gamma=np.sqrt(2)
                    )
                    m = run_e2_single(
                        orb, ctrl, X_p0, X_e0, nu0, dt, t_end,
                        sigma_rad, noise_seqs[seed], P0,
                    )
                    writer.writerow({
                        "seed": seed,
                        "geometry": geom,
                        "sigma_theta_deg": sigma_deg,
                        "T_FP": "" if m.T_FP is None else m.T_FP,
                        "v_FP": "" if m.v_FP is None else m.v_FP,
                        "radial_closure_rate_FP": "" if m.radial_closure_rate_FP is None else m.radial_closure_rate_FP,
                        "T_soft": "" if m.T_soft is None else m.T_soft,
                        "v_at_soft": "" if m.v_at_soft is None else m.v_at_soft,
                        "T_sustain": "" if m.T_sustain is None else m.T_sustain,
                        "sustained": m.sustained,
                        "max_dist_after_FP": "" if m.max_dist_after_FP is None else m.max_dist_after_FP,
                        "min_dist_after_FP": "" if m.min_dist_after_FP is None else m.min_dist_after_FP,
                        "exit_count": m.exit_count,
                        "time_in_sphere_frac": m.time_in_sphere_frac,
                        "total_delta_v": m.total_delta_v,
                        "peak_accel": m.peak_accel,
                        "control_energy": m.control_energy,
                        "rmse_pos_x": m.rmse_pos[0],
                        "rmse_pos_y": m.rmse_pos[1],
                        "rmse_pos_z": m.rmse_pos[2],
                        "rmse_dist": m.rmse_dist,
                        "final_pos_err": m.final_pos_err,
                        "final_vel_err": m.final_vel_err,
                        "est_dist_bias": m.est_dist_bias,
                        "est_radial_vel_bias": m.est_radial_vel_bias,
                        "nis_mean": m.nis_mean,
                        "nees_mean": m.nees_mean,
                        "are_fallback_count": m.are_fallback_count,
                        "controller_mean_step_ms": m.controller_mean_step_ms,
                    })
                    count += 1
                    if count % 30 == 0:
                        print(f"  [{count}/{total}] {geom} σ={sigma_deg}° seed={seed}")

    print(f"\nE2 结果已保存至 {csv_path}")

    # ── Go/No-Go 判据 ───────────────────────────────────────────────────
    print("\n--- Go/No-Go 判据 ---")
    _gono_go_decision(csv_path)

    return csv_path


def _gono_go_decision(csv_path: Path):
    """根据 E2 结果自动判定 Go/No-Go（纯 csv 实现，不依赖 pandas）。

    判据：高噪声 (σθ=0.1°) 相对低噪声 (σθ=0.001°):
      (a) T_FP 更早（speed 效应）
      (b) sustain_rate 下降（softness 效应）
    两类判据各自独立投票，至少两类中的大部分几何方向支持 → Go。
    """
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    tfp_decisions = []
    sus_decisions = []
    inversion_decisions = []
    for geom in GEOMETRY_LABELS:
        low_tfps, high_tfps = [], []
        low_sus, high_sus = [], []

        for row in rows:
            if row["geometry"] != geom:
                continue
            sigma = float(row["sigma_theta_deg"])
            tfp_str = row["T_FP"]
            sus_str = row["sustained"]

            if sigma == 0.001:
                if tfp_str:
                    low_tfps.append(float(tfp_str))
                low_sus.append(sus_str.strip().lower() == "true")
            elif sigma == 0.1:
                if tfp_str:
                    high_tfps.append(float(tfp_str))
                high_sus.append(sus_str.strip().lower() == "true")

        tfp_low = float(np.median(low_tfps)) if low_tfps else None
        tfp_high = float(np.median(high_tfps)) if high_tfps else None
        sus_low = np.mean(low_sus) if low_sus else 0
        sus_high = np.mean(high_sus) if high_sus else 0

        tfp_earlier = (tfp_high is not None and tfp_low is not None
                       and tfp_high < tfp_low)
        sus_drop = sus_low - sus_high
        sus_degraded = sus_drop > 0.15  # sustain_rate 下降 ≥15%

        print(f"  {geom}: T_FP(low)={tfp_low:.0f}, T_FP(high)={tfp_high:.0f}, "
              f"earlier={tfp_earlier}, sustain_drop={sus_drop:.1%}, "
              f"degraded={sus_degraded}")

        tfp_decisions.append(tfp_earlier)
        sus_decisions.append(sus_degraded)
        inversion_decisions.append(tfp_earlier and sus_degraded)

    tfp_go = sum(tfp_decisions) >= 2
    sus_go = sum(sus_decisions) >= 2
    go = bool(sum(inversion_decisions) >= 2)
    print(f"\n  速度判据 (T_FP earlier): {sum(tfp_decisions)}/3 → {'支持' if tfp_go else '不支持'}")
    print(f"  软度判据 (sustain drop):  {sum(sus_decisions)}/3 → {'支持' if sus_go else '不支持'}")
    print(f"  同几何指标反转:            {sum(inversion_decisions)}/3")
    print(f"  Go/No-Go: {'GO (至少 2/3 几何同现)' if go else 'NO-GO'}")
    return go


# ── 主入口 ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    output_dir = DATA_DIR / "e2_experiment"
    output_dir.mkdir(parents=True, exist_ok=True)

    e1_pass = run_e1(output_dir)
    if not e1_pass:
        print("E1 未通过：场景本身不可达，终止实验。")
        sys.exit(1)

    run_e2(output_dir)

