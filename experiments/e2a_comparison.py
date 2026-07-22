"""
E2a Phase 3: 配对对比实验
crosstrack: sigma={0.001,0.008,0.1} × nav={jacobian,ukf} × dt={10,5} × seeds={1,3,9}
"""
from __future__ import annotations
import csv, json, os, sys, warnings, time
from pathlib import Path
import numpy as np

from aerospace.dynamics.nerm import OrbitalDynamics
from aerospace.control.sdre import SDREGameController
from aerospace.simulation.nerm_ekf_sdre import EKFSDRESimulation
from aerospace.simulation.metrics import compute_metrics
from aerospace.estimation.navigators import create_navigator
from aerospace.paths import DATA_DIR

warnings.filterwarnings("ignore")
DEG2RAD = np.pi / 180.0

from experiments.e2_flow_isolated_n10 import (
    make_initial_state, los_p0, generate_x0_errors, FixedNoiseRNG,
    generate_paired_noise_sequences, seed_from_str,
    MU, A_C, E_C, CAPTURE_DIST, V_THRESH, SUSTAIN_DUR,
    T_ORBIT_FACTOR, N_SEEDS, MASTER_SEED,
)

SIGMAS = [0.001, 0.008, 0.1]
NAV_TYPES = ["jacobian", "ukf"]
DT_VALUES = [10.0, 5.0]
SEEDS = [1, 3, 9]  # worst, median, best


def run_one(geom: str, seed: int, sigma_deg: float, dt: float,
            nav_type: str) -> dict:
    orb = OrbitalDynamics(mu=MU, a_c=A_C, e_c=E_C)
    t_end = T_ORBIT_FACTOR * orb.T_orbit
    n_steps = int(t_end / dt) + 2
    noise_seqs = generate_paired_noise_sequences(N_SEEDS, n_steps, 2)

    X_p0, X_e0 = make_initial_state(geom, dist_km=30.0)
    P0 = los_p0(X_p0 - X_e0)
    x0_errs = generate_x0_errors(N_SEEDS, P0, seed_offset=seed_from_str(geom))
    x0_err = x0_errs[seed]

    sigma_rad = max(sigma_deg * DEG2RAD, 1e-12)
    R_meas = np.diag([sigma_rad**2, sigma_rad**2])
    Q_proc = np.zeros((6, 6))

    x_true_rel0 = X_p0 - X_e0
    x_est0 = x_true_rel0 + x0_err

    nav = create_navigator(nav_type, x0=x_est0, P0=P0, Q=Q_proc, R=R_meas,
                           angles_only=True)

    ctrl = SDREGameController(Q=np.eye(6), R=np.eye(3)*1e13, gamma=np.sqrt(2))

    noise_seq = noise_seqs[seed]
    noise_scaled = np.zeros((n_steps, R_meas.shape[0]))
    for i in range(min(len(noise_seq), n_steps)):
        noise_scaled[i] = noise_seq[i] * sigma_rad
    rng = FixedNoiseRNG(noise_scaled, R_meas.shape[0])

    t0 = time.perf_counter()
    sim = EKFSDRESimulation(
        dynamics=orb, controller=ctrl, navigator=nav,
        X_p0=X_p0, X_e0=X_e0, nu0=0.0, dt=dt,
        are_interval=1, rng=rng,
        passive_evader=True, early_stop=False,
        prediction_map="flow",
    )
    result = sim.run(t_end=t_end)
    elapsed = time.perf_counter() - t0
    m = compute_metrics(result, dt=dt, capture_dist=CAPTURE_DIST,
                        v_thresh=V_THRESH, sustain_dur=SUSTAIN_DUR)
    m.are_fallback_count = ctrl._are_fallback_count

    return {
        "geom": geom, "seed": seed, "sigma_deg": sigma_deg,
        "dt": dt, "nav_type": nav_type,
        "T_FP": "" if m.T_FP is None else m.T_FP,
        "sustained": m.sustained,
        "nis_mean": "" if m.nis_mean is None else m.nis_mean,
        "nees_mean": "" if m.nees_mean is None else m.nees_mean,
        "nis_inlier_frac": "" if m.nis_inlier_frac is None else m.nis_inlier_frac,
        "nees_inlier_frac": "" if m.nees_inlier_frac is None else m.nees_inlier_frac,
        "rmse_dist": m.rmse_dist,
        "final_pos_err": m.final_pos_err,
        "est_dist_bias": m.est_dist_bias,
        "total_delta_v": m.total_delta_v,
        "peak_accel": m.peak_accel,
        "are_fallback_count": m.are_fallback_count,
        "wall_time_s": elapsed,
        "n_steps": n_steps,
    }


if __name__ == "__main__":
    output_dir = DATA_DIR / "e2a_crosstrack_consistency"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "e2a_comparison.csv"
    fieldnames = [
        "geom", "seed", "sigma_deg", "dt", "nav_type",
        "T_FP", "sustained", "nis_mean", "nees_mean",
        "nis_inlier_frac", "nees_inlier_frac",
        "rmse_dist", "final_pos_err", "est_dist_bias",
        "total_delta_v", "peak_accel", "are_fallback_count",
        "wall_time_s", "n_steps",
    ]

    geom = "crosstrack"
    total = len(SIGMAS) * len(NAV_TYPES) * len(DT_VALUES) * len(SEEDS)
    count = 0
    print(f"E2a Phase 3: {total} comparison runs")
    print(f"  Sigmas: {SIGMAS}")
    print(f"  Navigators: {NAV_TYPES}")
    print(f"  dt: {DT_VALUES}")
    print(f"  Seeds: {SEEDS}")
    print()

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()

        for sigma_deg in SIGMAS:
            for dt in DT_VALUES:
                for nav_type in NAV_TYPES:
                    for seed in SEEDS:
                        count += 1
                        print(f"  [{count}/{total}] sigma={sigma_deg} dt={dt} "
                              f"nav={nav_type} seed={seed} ...", end=" ", flush=True)
                        row = run_one(geom, seed, sigma_deg, dt, nav_type)
                        writer.writerow(row)
                        f.flush()
                        print(f"NEES={row['nees_mean']}, NIS={row['nis_mean']}, "
                              f"rmse={row['rmse_dist']:.0f}m, "
                              f"wall={row['wall_time_s']:.0f}s", flush=True)

    print(f"\nResults saved to {csv_path}")

    # ── Summary ──────────────────────────────────────────────────
    print("\n=== SUMMARY ===")
    print(f"{'sigma':>7s} {'dt':>5s} {'nav':>9s} {'seed':>5s} {'NEES':>10s} {'NIS':>8s} {'rmse_dist':>10s}")
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    for r in rows:
        print(f"{r['sigma_deg']:>7s} {r['dt']:>5s} {r['nav_type']:>9s} {r['seed']:>5s} "
              f"{r['nees_mean']:>10s} {r['nis_mean']:>8s} {r['rmse_dist']:>10s}")
