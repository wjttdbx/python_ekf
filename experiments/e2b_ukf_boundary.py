"""
E2b: Flow-UKF 几何一致性边界实验

与 E2 Flow-Jacobian 基线严格配对：完全相同的初始状态、LOS P0、x0_err、
测量噪声序列、dt、时域、控制器参数。

输出策略：每个 case 原子写入独立 JSON → 合并为 ukf_results.csv。
"""

from __future__ import annotations
import csv, json, os, sys, warnings, time
from pathlib import Path
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

# ── 复用 E2 的全部实验定义 ──────────────────────────────────────────────
from experiments.e2_flow_isolated_n10 import (
    make_initial_state, los_p0, generate_x0_errors, FixedNoiseRNG,
    generate_paired_noise_sequences, seed_from_str,
    MU, A_C, E_C, DEG2RAD as _UNUSED,
    SIGMA_THETA_VALUES, GEOMETRY_LABELS, N_SEEDS, MASTER_SEED,
    T_ORBIT_FACTOR, CAPTURE_DIST, V_THRESH, SUSTAIN_DUR,
)

NAV_TYPE = "ukf"


# ═══════════════════════════════════════════════════════════════════════════════
# 原子级 worker
# ═══════════════════════════════════════════════════════════════════════════════

def _ukf_worker(job: dict) -> dict:
    """单个 UKF 仿真 case，返回完整指标字典。"""
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
    R_meas = np.diag([sigma_ang_eff ** 2, sigma_ang_eff ** 2])
    Q_proc = np.zeros((6, 6))

    x_est0 = (X_p0 - X_e0) + x0_err
    nav = create_navigator(NAV_TYPE, x0=x_est0, P0=P0, Q=Q_proc, R=R_meas,
                           angles_only=True)

    n_steps_expected = int(t_end / dt) + 2
    noise_scaled = np.zeros((n_steps_expected, R_meas.shape[0]))
    for i in range(min(len(noise_seq), n_steps_expected)):
        noise_scaled[i] = noise_seq[i] * sigma_ang_eff

    rng = FixedNoiseRNG(noise_scaled, R_meas.shape[0])

    t0 = time.perf_counter()
    sim = EKFSDRESimulation(
        dynamics=orb, controller=ctrl, navigator=nav,
        X_p0=X_p0, X_e0=X_e0, nu0=nu0, dt=dt,
        are_interval=1, rng=rng,
        passive_evader=True, early_stop=False,
        prediction_map="flow",
    )
    result = sim.run(t_end=t_end)
    wall_s = time.perf_counter() - t0

    m = compute_metrics(result, dt=dt, capture_dist=CAPTURE_DIST,
                        v_thresh=V_THRESH, sustain_dur=SUSTAIN_DUR)
    m.are_fallback_count = ctrl._are_fallback_count

    # ── 边界指标：P 条件数与最小特征值 ──────────────────────────────
    P_final = nav.P
    try:
        P_eigvals = np.linalg.eigvalsh(P_final)
        P_eig_min = float(np.min(P_eigvals))
        P_cond = float(np.max(P_eigvals) / max(np.min(P_eigvals), 1e-300))
    except Exception:
        P_eig_min = float("nan")
        P_cond = float("nan")

    # ── NIS/NEES 超界比例 ──────────────────────────────────────────
    from scipy.stats import chi2
    nis_hist = getattr(result, 'nis_history', None)
    if nis_hist is not None and len(nis_hist) > 0:
        valid_nis = nis_hist[~np.isnan(nis_hist)]
        m_dim = result.innov_history.shape[0] if hasattr(result, 'innov_history') else 2
        threshold_nis = chi2.ppf(0.997, m_dim)
        nis_inlier_frac = float(np.mean(valid_nis < threshold_nis)) if len(valid_nis) > 0 else float("nan")
    else:
        nis_inlier_frac = float("nan")

    nees_hist = getattr(result, 'nees_history', None)
    if nees_hist is not None and len(nees_hist) > 0:
        valid_nees = nees_hist[~np.isnan(nees_hist)]
        threshold_nees = chi2.ppf(0.997, 6)
        nees_inlier_frac = float(np.mean(valid_nees < threshold_nees)) if len(valid_nees) > 0 else float("nan")
    else:
        nees_inlier_frac = float("nan")

    return {
        "seed": seed, "geometry": geom, "sigma_theta_deg": sigma_deg,
        "nav_type": NAV_TYPE,
        "T_FP": None if m.T_FP is None else float(m.T_FP),
        "v_FP": None if m.v_FP is None else float(m.v_FP),
        "T_soft": None if m.T_soft is None else float(m.T_soft),
        "v_at_soft": None if m.v_at_soft is None else float(m.v_at_soft),
        "T_sustain": None if m.T_sustain is None else float(m.T_sustain),
        "sustained": m.sustained,
        "exit_count": m.exit_count,
        "time_in_sphere_frac": m.time_in_sphere_frac,
        "total_delta_v": m.total_delta_v,
        "peak_accel": m.peak_accel,
        "control_energy": m.control_energy,
        "rmse_pos_x": m.rmse_pos[0], "rmse_pos_y": m.rmse_pos[1], "rmse_pos_z": m.rmse_pos[2],
        "rmse_dist": m.rmse_dist,
        "final_pos_err": m.final_pos_err, "final_vel_err": m.final_vel_err,
        "est_dist_bias": m.est_dist_bias, "est_radial_vel_bias": m.est_radial_vel_bias,
        "nis_mean": None if m.nis_mean is None else float(m.nis_mean),
        "nees_mean": None if m.nees_mean is None else float(m.nees_mean),
        "nis_inlier_frac": nis_inlier_frac,
        "nees_inlier_frac": nees_inlier_frac,
        "are_fallback_count": m.are_fallback_count,
        "P_eig_min": P_eig_min, "P_cond": P_cond,
        "wall_time_s": wall_s,
    }


def _pool_init() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


# ═══════════════════════════════════════════════════════════════════════════════
# 主运行器
# ═══════════════════════════════════════════════════════════════════════════════

def run_ukf_experiment(output_dir: Path, n_workers: int | None = None,
                        seeds_override: int | None = None):
    if n_workers is None:
        n_workers = min(cpu_count(), 10)

    n_seeds_eff = seeds_override if seeds_override is not None else N_SEEDS

    print("=" * 60)
    print("E2b: Flow-UKF boundary experiment")
    print(f"Navigator: {NAV_TYPE} (flow-map UKF, Q=0)")
    print(f"Noise levels: {SIGMA_THETA_VALUES} deg")
    print(f"Geometries: {GEOMETRY_LABELS}")
    print(f"Seeds: {n_seeds_eff}, parallel workers: {n_workers}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    orb = OrbitalDynamics(mu=MU, a_c=A_C, e_c=E_C)
    t_end = T_ORBIT_FACTOR * orb.T_orbit
    dt = 10.0
    n_steps = int(t_end / dt) + 2
    m_dim = 2

    noise_seqs = generate_paired_noise_sequences(n_seeds_eff, n_steps, m_dim)

    # ── 原子输出目录 ──────────────────────────────────────────────
    atoms_dir = output_dir / "_atoms"
    atoms_dir.mkdir(parents=True, exist_ok=True)

    # ── 构建 job 列表 ────────────────────────────────────────────
    jobs: list[dict] = []
    for geom in GEOMETRY_LABELS:
        X_p0, X_e0 = make_initial_state(geom, dist_km=30.0)
        P0 = los_p0(X_p0 - X_e0)
        x0_errs = generate_x0_errors(n_seeds_eff, P0, seed_offset=seed_from_str(geom))

        for seed in range(n_seeds_eff):
            for sigma_deg in SIGMA_THETA_VALUES:
                # 跳过已存在的原子文件
                atom_path = atoms_dir / f"{geom}_seed{seed}_sigma{sigma_deg}.json"
                if atom_path.exists():
                    continue
                sigma_rad = sigma_deg * DEG2RAD
                jobs.append({
                    "geom": geom, "seed": seed,
                    "sigma_deg": sigma_deg, "sigma_rad": sigma_rad,
                    "noise_seq": noise_seqs[seed].tolist(),
                    "P0": P0.tolist(), "x0_err": x0_errs[seed].tolist(),
                    "X_p0": X_p0.tolist(), "X_e0": X_e0.tolist(),
                    "nu0": 0.0, "dt": dt, "t_end": t_end,
                    "_atom_path": str(atom_path),
                })

    total_jobs = len(GEOMETRY_LABELS) * len(SIGMA_THETA_VALUES) * n_seeds_eff
    # Count already completed
    already_done = total_jobs - len(jobs)
    if already_done > 0:
        print(f"Resuming: {already_done}/{total_jobs} already complete")

    print(f"Queued: {len(jobs)} jobs pending")

    if jobs:
        with Pool(processes=n_workers, initializer=_pool_init) as pool:
            for i, result in enumerate(pool.imap_unordered(_ukf_worker, jobs)):
                # 原子写入
                atom_path_str = jobs[i]["_atom_path"] if i < len(jobs) else None
                # imap_unordered doesn't guarantee order, so find the atom path from result
                atom_path = atoms_dir / f"{result['geometry']}_seed{result['seed']}_sigma{result['sigma_theta_deg']}.json"
                with open(atom_path, "w") as f_atom:
                    json.dump(result, f_atom)
                count_done = already_done + i + 1
                if count_done % 10 == 0 or count_done == total_jobs:
                    print(f"  [{count_done}/{total_jobs}] {result['geometry']} "
                          f"sigma={result['sigma_theta_deg']} seed={result['seed']} "
                          f"NEES={result['nees_mean']} sus={result['sustained']}",
                          flush=True)

    # ── 合并原子文件为 CSV ─────────────────────────────────────────
    print("\nMerging atom files into ukf_results.csv ...")
    fieldnames = [
        "seed", "geometry", "sigma_theta_deg", "nav_type",
        "T_FP", "v_FP", "T_soft", "v_at_soft", "T_sustain", "sustained",
        "exit_count", "time_in_sphere_frac",
        "total_delta_v", "peak_accel", "control_energy",
        "rmse_pos_x", "rmse_pos_y", "rmse_pos_z", "rmse_dist",
        "final_pos_err", "final_vel_err",
        "est_dist_bias", "est_radial_vel_bias",
        "nis_mean", "nees_mean", "nis_inlier_frac", "nees_inlier_frac",
        "are_fallback_count", "P_eig_min", "P_cond", "wall_time_s",
    ]

    csv_path = output_dir / "ukf_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for atom_file in sorted(atoms_dir.glob("*.json")):
            with open(atom_file) as af:
                row = json.load(af)
            writer.writerow(row)

    # 也保存为 JSON
    summary = {
        "experiment": "E2b Flow-UKF boundary",
        "navigator": NAV_TYPE,
        "prediction_map": "flow",
        "Q": "zero",
        "n_seeds": n_seeds_eff,
        "sigma_theta_deg": SIGMA_THETA_VALUES,
        "geometries": GEOMETRY_LABELS,
        "total_runs": total_jobs,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"UKF results saved to {csv_path} ({total_jobs} rows)")
    return csv_path


# ═══════════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Run smoke test with 1 seed only")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    output_dir = DATA_DIR / "e2b_geometry_filter_boundary"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_seeds_eff = 1 if args.smoke else N_SEEDS
    csv_path = run_ukf_experiment(output_dir, n_workers=args.workers,
                                   seeds_override=n_seeds_eff)
