"""
E2 n=10 隔离确认实验 — speed-softness 现象复验

隔离措施（与旧 E2 对比）：
  1. 同一初始估计误差跨噪声组复用（消除 x0_err 混杂）
  2. 同一标准正态量测序列 × 不同 σθ 缩放（消除噪声实现混杂）
  3. 每条轨迹新建 SDRE 控制器（消除跨轨迹有状态泄漏）
  4. 固定 LOS 坐标 P0（消除 P0 混杂）
  5. flow-map Jacobian EKF + Q=0（E0c 选定主导航器）
  6. 被动目标 u_e=0，固定全时域，不施加推力限制

Go/No-Go 判据（每几何内同时满足 → ≥2/3 几何）：
  (a) T_FP(high noise) < T_FP(low noise)  — speed 效应
  (b) sustained_rate(high) < sustained_rate(low) — softness 效应
  时间事件含失败样本时使用删失感知指标。
"""

from __future__ import annotations

import csv
import json
import os
import sys
import warnings
from pathlib import Path
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count

import numpy as np

from aerospace.dynamics.nerm import OrbitalDynamics
from aerospace.control.sdre import SDREGameController
from aerospace.simulation.nerm_ekf_sdre import EKFSDRESimulation
from aerospace.simulation.metrics import compute_metrics, CaptureMetrics
from aerospace.estimation.navigators import create_navigator
from aerospace.paths import DATA_DIR

warnings.filterwarnings("ignore")

DEG2RAD = np.pi / 180.0


def seed_from_str(s: str) -> int:
    import zlib
    return zlib.adler32(s.encode()) & 0x7fffffff


# ═══════════════════════════════════════════════════════════════════════════════
# 实验配置
# ═══════════════════════════════════════════════════════════════════════════════

NAV_TYPE = "jacobian"
SIGMA_THETA_VALUES = [0.001, 0.008, 0.1]  # deg
GEOMETRY_LABELS = ["radial", "alongtrack", "crosstrack"]
N_SEEDS = 10
MASTER_SEED = 42
T_ORBIT_FACTOR = 3.0
CAPTURE_DIST = 0.1   # km
SUSTAIN_DUR = 600.0   # s
V_THRESH = 1e-4       # km/s = 0.1 m/s

MU = 3.986e5
A_C = 15000.0
E_C = 0.5

DIST_NOM = 30.0  # km
P0_POS_LOS_DIAG = np.array([1.0**2, (DIST_NOM * 0.01 * DEG2RAD)**2,
                             (DIST_NOM * 0.01 * DEG2RAD)**2])
P0_VEL_VAR = (1e-3)**2  # (km/s)^2


# ═══════════════════════════════════════════════════════════════════════════════
# 几何构造
# ═══════════════════════════════════════════════════════════════════════════════

def make_initial_state(geometry: str, dist_km: float = 30.0,
                       vel_scale: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
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


def los_p0(x_rel0: np.ndarray) -> np.ndarray:
    """在 LOS 坐标系中定义初始协方差，旋转到 LVLH。"""
    r = x_rel0[:3]
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-12:
        return np.diag([P0_POS_LOS_DIAG[0]] * 3 + [P0_VEL_VAR] * 3)

    e_los = r / r_norm
    e2 = np.array([-e_los[1], e_los[0], 0.0])
    if np.linalg.norm(e2) < 1e-12:
        e2 = np.array([1.0, 0.0, 0.0])
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e_los, e2)

    R_lvlh_to_los = np.array([e_los, e2, e3])
    P_los_pos = np.diag([P0_POS_LOS_DIAG[0], P0_POS_LOS_DIAG[1], P0_POS_LOS_DIAG[2]])
    P_lvlh_pos = R_lvlh_to_los.T @ P_los_pos @ R_lvlh_to_los
    P_lvlh_vel = np.eye(3) * P0_VEL_VAR
    P = np.zeros((6, 6))
    P[:3, :3] = P_lvlh_pos
    P[3:, 3:] = P_lvlh_vel
    return P


# ═══════════════════════════════════════════════════════════════════════════════
# 配对噪声 & 初始误差预生成
# ═══════════════════════════════════════════════════════════════════════════════

def generate_paired_noise_sequences(n_seeds: int, n_steps: int,
                                     m_dim: int) -> np.ndarray:
    rng = np.random.default_rng(MASTER_SEED)
    return rng.standard_normal((n_seeds, n_steps, m_dim))


def generate_x0_errors(n_seeds: int, P0: np.ndarray,
                        seed_offset: int = 999) -> np.ndarray:
    """为每个种子预生成初始估计误差，从 P0 多元正态采样。"""
    rng = np.random.default_rng(MASTER_SEED + seed_offset)
    return rng.multivariate_normal(np.zeros(6), P0, size=n_seeds)


# ═══════════════════════════════════════════════════════════════════════════════
# 固定噪声 RNG
# ═══════════════════════════════════════════════════════════════════════════════

class FixedNoiseRNG:
    def __init__(self, noise_array: np.ndarray, m_dim: int):
        self._noise = noise_array
        self._m_dim = m_dim
        self._idx = 0

    def multivariate_normal(self, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        if self._idx >= len(self._noise):
            return np.zeros(self._m_dim)
        z = self._noise[self._idx]
        self._idx += 1
        return z


# ═══════════════════════════════════════════════════════════════════════════════
# Go/No-Go 判定（删失感知）
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GeometryVerdict:
    geom: str
    tfp_low_median: float | None
    tfp_high_median: float | None
    tfp_low_censored: int
    tfp_high_censored: int
    sus_low_rate: float
    sus_high_rate: float
    tfp_earlier: bool
    sus_degraded: bool
    passed: bool
    note: str = ""


def _gono_go_decision(csv_path: Path) -> bool:
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    verdicts = []
    for geom in GEOMETRY_LABELS:
        low_data = _extract_group(rows, geom, 0.001)
        high_data = _extract_group(rows, geom, 0.1)

        low_tfps = [d["T_FP"] for d in low_data if d["T_FP"] is not None]
        high_tfps = [d["T_FP"] for d in high_data if d["T_FP"] is not None]
        low_censored = len(low_data) - len(low_tfps)
        high_censored = len(high_data) - len(high_tfps)

        tfp_low_med = float(np.median(low_tfps)) if low_tfps else None
        tfp_high_med = float(np.median(high_tfps)) if high_tfps else None

        tfp_earlier = False
        note_parts = []
        if low_censored > N_SEEDS // 2:
            note_parts.append(f"low noise {low_censored}/{N_SEEDS} no FP")
        if high_censored > N_SEEDS // 2:
            note_parts.append(f"high noise {high_censored}/{N_SEEDS} no FP")

        if (tfp_low_med is not None and tfp_high_med is not None
                and low_censored <= N_SEEDS // 2 and high_censored <= N_SEEDS // 2):
            tfp_earlier = tfp_high_med < tfp_low_med
            if not tfp_earlier:
                note_parts.append(
                    f"T_FP high={tfp_high_med:.0f}s >= low={tfp_low_med:.0f}s")

        sus_low = np.mean([d["sustained"] for d in low_data])
        sus_high = np.mean([d["sustained"] for d in high_data])
        sus_drop = sus_low - sus_high
        sus_degraded = sus_drop > 0.15

        passed = tfp_earlier and sus_degraded
        note = "; ".join(note_parts) if note_parts else ""

        v = GeometryVerdict(
            geom=geom,
            tfp_low_median=tfp_low_med, tfp_high_median=tfp_high_med,
            tfp_low_censored=low_censored, tfp_high_censored=high_censored,
            sus_low_rate=sus_low, sus_high_rate=sus_high,
            tfp_earlier=tfp_earlier, sus_degraded=sus_degraded,
            passed=passed, note=note,
        )
        verdicts.append(v)

        censor_info = ""
        if low_censored > 0 or high_censored > 0:
            censor_info = f" [censored: low={low_censored}, high={high_censored}]"
        print(f"  {geom}: T_FP(low)={tfp_low_med}, T_FP(high)={tfp_high_med}, "
              f"earlier={tfp_earlier}, "
              f"sustain(low)={sus_low:.1%}, sustain(high)={sus_high:.1%}, "
              f"drop={sus_drop:.1%}, degraded={sus_degraded}"
              f"{censor_info}")
        if note:
            print(f"         WARNING: {note}")

    n_passed = sum(1 for v in verdicts if v.passed)
    go = n_passed >= 2

    print(f"\n  Both criteria per geometry: {n_passed}/3")
    for v in verdicts:
        status = "GO" if v.passed else "--"
        print(f"    {v.geom}: speed={v.tfp_earlier}, soft={v.sus_degraded} -> {status}")
    print(f"  Go/No-Go: {'GO' if go else 'NO-GO'}")

    return go


def _extract_group(rows: list[dict], geom: str, sigma_deg: float) -> list[dict]:
    result = []
    for row in rows:
        if row["geometry"] != geom:
            continue
        if abs(float(row["sigma_theta_deg"]) - sigma_deg) > 0.0005:
            continue
        tfp_str = row["T_FP"]
        sus_str = row["sustained"]
        result.append({
            "T_FP": float(tfp_str) if tfp_str and tfp_str.strip() else None,
            "sustained": sus_str.strip().lower() == "true",
        })
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 并行 worker 函数（模块级别，可被 pickle 序列化）
# ═══════════════════════════════════════════════════════════════════════════════

def _worker(job: dict) -> dict:
    """单个仿真 job 的执行函数（独立进程内运行）。"""
    geom = job["geom"]
    seed = job["seed"]
    sigma_deg = job["sigma_deg"]
    sigma_rad = job["sigma_rad"]
    noise_seq = np.array(job["noise_seq"])
    P0 = np.array(job["P0"])
    x0_err = np.array(job["x0_err"])
    X_p0 = np.array(job["X_p0"])
    X_e0 = np.array(job["X_e0"])
    nu0 = job["nu0"]
    dt = job["dt"]
    t_end = job["t_end"]

    orb = OrbitalDynamics(mu=MU, a_c=A_C, e_c=E_C)
    ctrl = SDREGameController(Q=np.eye(6), R=np.eye(3) * 1e13, gamma=np.sqrt(2))

    sigma_ang_eff = max(sigma_rad, 1e-12)
    R_meas = np.diag([sigma_ang_eff**2, sigma_ang_eff**2])
    Q_proc = np.zeros((6, 6))

    x_est0 = (X_p0 - X_e0) + x0_err
    nav = create_navigator(NAV_TYPE, x0=x_est0, P0=P0, Q=Q_proc, R=R_meas,
                           angles_only=True)

    n_steps_expected = int(t_end / dt) + 2
    noise_scaled = np.zeros((n_steps_expected, R_meas.shape[0]))
    for i in range(min(len(noise_seq), n_steps_expected)):
        noise_scaled[i] = noise_seq[i] * sigma_ang_eff

    rng = FixedNoiseRNG(noise_scaled, R_meas.shape[0])

    sim = EKFSDRESimulation(
        dynamics=orb, controller=ctrl, navigator=nav,
        X_p0=X_p0, X_e0=X_e0, nu0=nu0, dt=dt,
        are_interval=1, rng=rng,
        passive_evader=True, early_stop=False,
        prediction_map="flow",
    )
    result = sim.run(t_end=t_end)
    m = compute_metrics(result, dt=dt, capture_dist=CAPTURE_DIST,
                        v_thresh=V_THRESH, sustain_dur=SUSTAIN_DUR)
    m.are_fallback_count = ctrl._are_fallback_count
    if ctrl.step_times:
        m.controller_mean_step_ms = float(np.mean(ctrl.step_times) * 1000.0)

    return {
        "seed": seed, "geometry": geom, "sigma_theta_deg": sigma_deg,
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
        "rmse_pos_x": m.rmse_pos[0], "rmse_pos_y": m.rmse_pos[1], "rmse_pos_z": m.rmse_pos[2],
        "rmse_dist": m.rmse_dist,
        "final_pos_err": m.final_pos_err, "final_vel_err": m.final_vel_err,
        "est_dist_bias": m.est_dist_bias, "est_radial_vel_bias": m.est_radial_vel_bias,
        "nis_mean": "" if m.nis_mean is None else m.nis_mean,
        "nees_mean": "" if m.nees_mean is None else m.nees_mean,
        "are_fallback_count": m.are_fallback_count,
        "controller_mean_step_ms": m.controller_mean_step_ms,
    }


def _pool_init() -> None:
    """Worker 进程初始化：禁止 numpy 内部多线程竞争。"""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


# ═══════════════════════════════════════════════════════════════════════════════
# 主实验（并行版，支持断点恢复与增量写盘）
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(output_dir: Path, n_workers: int | None = None):
    if n_workers is None:
        n_workers = min(cpu_count(), 10)

    print("=" * 60)
    print("E2 n=10 isolation experiment (parallel)")
    print(f"Navigator: {NAV_TYPE} (flow-map Jacobian EKF, Q=0)")
    print(f"Noise levels: {SIGMA_THETA_VALUES} deg")
    print(f"Geometries: {GEOMETRY_LABELS}")
    print(f"Seeds: {N_SEEDS}, parallel workers: {n_workers}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    orb = OrbitalDynamics(mu=MU, a_c=A_C, e_c=E_C)
    t_end = T_ORBIT_FACTOR * orb.T_orbit
    dt = 10.0
    n_steps = int(t_end / dt) + 2
    m_dim = 2

    noise_seqs = generate_paired_noise_sequences(N_SEEDS, n_steps, m_dim)

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

    # ── 断点恢复 ──────────────────────────────────────────────────
    completed: set[tuple[str, int, float]] = set()
    if csv_path.exists() and csv_path.stat().st_size > 0:
        try:
            with open(csv_path, newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("seed") and row.get("geometry") and row.get("sigma_theta_deg"):
                        completed.add((
                            row["geometry"],
                            int(row["seed"]),
                            float(row["sigma_theta_deg"]),
                        ))
            print(f"Resuming: {len(completed)} existing records")
        except Exception:
            completed.clear()

    # ── 构建 job 列表 ──────────────────────────────────────────────
    jobs: list[dict] = []
    for geom in GEOMETRY_LABELS:
        X_p0, X_e0 = make_initial_state(geom, dist_km=30.0)
        P0 = los_p0(X_p0 - X_e0)
        x0_errs = generate_x0_errors(N_SEEDS, P0, seed_offset=seed_from_str(geom))

        for seed in range(N_SEEDS):
            for sigma_deg in SIGMA_THETA_VALUES:
                if (geom, seed, sigma_deg) in completed:
                    continue
                sigma_rad = sigma_deg * DEG2RAD
                jobs.append({
                    "geom": geom, "seed": seed,
                    "sigma_deg": sigma_deg, "sigma_rad": sigma_rad,
                    "noise_seq": noise_seqs[seed].tolist(),
                    "P0": P0.tolist(), "x0_err": x0_errs[seed].tolist(),
                    "X_p0": X_p0.tolist(), "X_e0": X_e0.tolist(),
                    "nu0": 0.0, "dt": dt, "t_end": t_end,
                })

    total = len(GEOMETRY_LABELS) * len(SIGMA_THETA_VALUES) * N_SEEDS
    print(f"Queued: {len(jobs)}/{total} trajectories pending")

    if jobs:
        write_mode = "a" if completed else "w"
        count = len(completed)

        with open(csv_path, write_mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            if write_mode == "w":
                writer.writeheader()
                f.flush()

            with Pool(processes=n_workers, initializer=_pool_init) as pool:
                for result in pool.imap_unordered(_worker, jobs):
                    writer.writerow(result)
                    f.flush()
                    count += 1
                    if count % 5 == 0 or count == total:
                        print(f"  [{count}/{total}] {result['geometry']} "
                              f"sigma={result['sigma_theta_deg']} "
                              f"seed={result['seed']} "
                              f"T_FP={result['T_FP']} sus={result['sustained']}",
                              flush=True)

    print(f"\nResults saved to {csv_path} ({total} records)")

    # ── Go/No-Go ───────────────────────────────────────────────────
    print("\n" + "-" * 40)
    print("Go/No-Go Decision")
    print("-" * 40)
    go = _gono_go_decision(csv_path)

    return csv_path, go


# ═══════════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    output_dir = DATA_DIR / "e2_flow_isolated_n10"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path, go = run_experiment(output_dir)

    summary = {
        "experiment": "E2 n=10 isolated confirmation",
        "navigator": NAV_TYPE,
        "prediction_map": "flow",
        "Q": "zero",
        "n_seeds": N_SEEDS,
        "sigma_theta_deg": SIGMA_THETA_VALUES,
        "geometries": GEOMETRY_LABELS,
        "gono_go": go,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {output_dir / 'summary.json'}")

    sys.exit(0 if go else 1)
