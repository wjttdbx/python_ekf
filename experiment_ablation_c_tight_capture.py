"""
Ablation C — 加紧捕获判据 (Tight Capture Criterion)
====================================================

Purpose:
    Test Factor 3 (first-passage bias at inflated terminal velocity) of the
    §5.2 three-factor noise-acceleration mechanism.

Setup:
    Same as baseline noise sweep (main.py + noise_sweep_results.csv),
    but the capture criterion is tightened from
        rho < 100 m
    to
        rho < 100 m  AND  |v_rel| < 0.1 m/s.

Prediction (from mechanism whitepaper §5):
    Under the classical loose criterion, higher sigma_theta accelerates
    approach because the chaser transits the 100m ball at inflated terminal
    velocity. Under the tight criterion the chaser must slow down before
    entering the ball, so:
    - Low sigma_theta seeds: still succeed (soft arrival, low terminal v_rel)
    - High sigma_theta seeds: expected to FAIL or take much longer (must
      brake, which requires re-approach cycles)

Data written to data/ablation_c_tight_capture_results.csv.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from aerospace.dynamics.nerm import OrbitalDynamics
from aerospace.control.sdre import SDREGameController
from aerospace.estimation.ekf import RelativeStateEKF
from aerospace.simulation.nerm_ekf_sdre import EKFSDRESimulation


DEG2RAD = np.pi / 180.0
SIGMAS_DEG = [0.001, 0.004, 0.008, 0.02, 0.05, 0.1]
SEEDS = [42, 43, 44]
V_REL_THRESHOLD = 0.1e-3  # 0.1 m/s in km/s
CAPTURE_DIST = 0.1        # 100 m in km


def make_ekf(sigma_ang_rad: float, x0: np.ndarray, initial_dist: float) -> RelativeStateEKF:
    R_meas = np.diag([max(sigma_ang_rad**2, 1e-30)] * 2)
    Q_proc = np.diag([5e-4, 5e-4, 5e-4, 5e-8, 5e-8, 5e-8])
    sigma_pos = initial_dist * sigma_ang_rad
    sigma_vel = 1.0 * sigma_ang_rad
    P0 = np.diag([sigma_pos**2] * 3 + [sigma_vel**2] * 3)
    return RelativeStateEKF(x0=x0, P0=P0, Q=Q_proc, R=R_meas, angles_only=True)


def run_one(sigma_deg: float, seed: int) -> dict:
    sigma_ang_rad = sigma_deg * DEG2RAD
    orb = OrbitalDynamics(mu=3.986e5, a_c=15000.0, e_c=0.5)
    X_p0 = np.array([500.0, 500.0, 500.0, 0.01, 0.01, 0.01])
    X_e0 = np.zeros(6)
    x0_est = X_p0 - X_e0
    initial_dist = float(np.linalg.norm(x0_est[:3]))

    ekf = make_ekf(sigma_ang_rad, x0_est, initial_dist)
    ctrl = SDREGameController(
        Q=np.eye(6),
        R=np.eye(3) * 1e13,
        gamma=np.sqrt(2),
    )
    sim = EKFSDRESimulation(
        dynamics=orb,
        controller=ctrl,
        ekf=ekf,
        X_p0=X_p0,
        X_e0=X_e0,
        nu0=0.0,
        dt=10.0,
        are_interval=1,
        capture_dist=CAPTURE_DIST,  # loose sim-side threshold; we post-filter with v_rel
        rng=np.random.default_rng(seed),
    )
    t_start = time.time()
    res = sim.run(t_end=10.0 * orb.T_orbit)
    wall = time.time() - t_start

    # Post-filter with tightened criterion: find first index where BOTH
    #   dist < CAPTURE_DIST AND |v_rel| < V_REL_THRESHOLD
    states = res.states           # (13, N)
    dist   = res.dist_history     # (N,)
    x_p    = states[3:6, :]       # chaser velocity (rows 3-5 of X_p)
    x_e    = states[9:12, :]      # evader velocity (rows 3-5 of X_e)
    v_rel_mag = np.linalg.norm(x_p - x_e, axis=0)  # km/s

    tight_hit = np.where((dist < CAPTURE_DIST) & (v_rel_mag < V_REL_THRESHOLD))[0]
    tight_capture = bool(len(tight_hit) > 0)
    if tight_capture:
        idx = int(tight_hit[0])
        t_capture_tight = float(res.t[idx])
        final_dist_tight = float(dist[idx])
        final_vrel_tight = float(v_rel_mag[idx])
    else:
        idx = int(np.argmin(dist))
        t_capture_tight = float("nan")
        final_dist_tight = float(dist[-1])
        final_vrel_tight = float(v_rel_mag[-1])
    # Also record loose-criterion metrics for direct comparison
    loose_hit = np.where(dist < CAPTURE_DIST)[0]
    loose_capture = bool(len(loose_hit) > 0)
    t_capture_loose = float(res.t[int(loose_hit[0])]) if loose_capture else float("nan")
    final_vrel_loose = (
        float(v_rel_mag[int(loose_hit[0])]) if loose_capture else float(v_rel_mag[-1])
    )
    return dict(
        sigma_ang_deg=sigma_deg,
        seed=seed,
        tight_captured=tight_capture,
        t_capture_tight_s=t_capture_tight,
        final_dist_tight_km=final_dist_tight,
        final_vrel_tight_ms=final_vrel_tight * 1000.0,
        loose_captured=loose_capture,
        t_capture_loose_s=t_capture_loose,
        final_vrel_loose_ms=final_vrel_loose * 1000.0,
        wall_time_s=wall,
    )


def main() -> None:
    out = Path(__file__).parent / "data" / "ablation_c_tight_capture_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(run_one(0.008, 42).keys())  # burn-in / probe
    rows = []
    for sigma in SIGMAS_DEG:
        for seed in SEEDS:
            print(f"[Ablation C] sigma={sigma} deg, seed={seed}", flush=True)
            row = run_one(sigma, seed)
            print(
                f"  loose: cap={row['loose_captured']} "
                f"t_loose={row['t_capture_loose_s']:.0f}s "
                f"v_loose={row['final_vrel_loose_ms']:.3f} m/s"
            )
            print(
                f"  tight: cap={row['tight_captured']} "
                f"t_tight={row['t_capture_tight_s'] if not np.isnan(row['t_capture_tight_s']) else 'FAIL':<6} "
                f"final_vrel={row['final_vrel_tight_ms']:.3f} m/s"
            )
            rows.append(row)
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
