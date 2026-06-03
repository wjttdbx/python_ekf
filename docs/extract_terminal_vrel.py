"""
Extract terminal relative velocity from noise-sweep and other experiments.
Re-runs simulations to record v_rel at capture for each trial.

Usage:
    uv run python docs/extract_terminal_vrel.py
"""

import sys
from pathlib import Path
import csv
import io
import contextlib
import time as time_mod
import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from aerospace.dynamics.nerm import OrbitalDynamics
from aerospace.control.sdre import SDREGameController
from aerospace.estimation.ekf import RelativeStateEKF
from aerospace.simulation.nerm_ekf_sdre import EKFSDRESimulation
from aerospace.paths import DATA_DIR

DEG2RAD = np.pi / 180.0

MU = 3.986e5; A_C = 15000.0; E_C = 0.5; DT = 10.0
X_P0 = np.array([500.0, 500.0, 500.0, 0.01, 0.01, 0.01])
X_E0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
NU0 = 0.0
Q_CTRL = np.eye(6); R_CTRL = np.eye(3) * 1e13
PROC_NOISE_POS = 5e-4; PROC_NOISE_VEL = 5e-8
GAMMA = np.sqrt(2)


def run_and_get_vrel(sigma_ang_deg, seed, angles_only=True):
    """Run one sim, return terminal relative velocity (m/s)."""
    sigma_ang_rad = sigma_ang_deg * DEG2RAD
    orb = OrbitalDynamics(mu=MU, a_c=A_C, e_c=E_C)
    ctrl = SDREGameController(Q=Q_CTRL, R=R_CTRL, gamma=GAMMA)

    x0_est = X_P0 - X_E0
    rho0 = float(np.linalg.norm(x0_est[:3]))
    R_meas = np.diag([sigma_ang_rad**2] * (2 if angles_only else 3))
    Q_proc = np.diag([PROC_NOISE_POS]*3 + [PROC_NOISE_VEL]*3)
    sigma_pos = rho0 * sigma_ang_rad
    sigma_vel = 1.0 * sigma_ang_rad
    P0 = np.diag([sigma_pos**2]*3 + [sigma_vel**2]*3)
    ekf = RelativeStateEKF(x0=x0_est, P0=P0, Q=Q_proc, R=R_meas, angles_only=angles_only)

    rng = np.random.default_rng(seed) if sigma_ang_deg > 0 else None
    sim = EKFSDRESimulation(dynamics=orb, controller=ctrl, ekf=ekf,
                            X_p0=X_P0, X_e0=X_E0, nu0=NU0, dt=DT,
                            are_interval=1, rng=rng)
    with contextlib.redirect_stdout(io.StringIO()):
        result = sim.run(t_end=10.0 * orb.T_orbit)

    # terminal relative velocity (km/s) → m/s
    s = result.states
    vrel = np.sqrt(np.sum((s[3:6, -1] - s[9:12, -1])**2))
    return float(vrel * 1000.0)  # m/s


def main():
    # --- 1. Noise sweep terminal velocities ---
    sigma_levels = [0.001, 0.004, 0.008, 0.02, 0.05, 0.1]
    seeds = [42, 43, 44, 45, 46]

    print("="*60)
    print("Noise sweep: extracting terminal v_rel")
    print("="*60)

    noise_vrel = {}
    for sa in sigma_levels:
        vals = []
        for seed in seeds:
            v = run_and_get_vrel(sa, seed, angles_only=True)
            vals.append(v)
            print(f"  σ={sa:.3f}° seed={seed}: v_rel={v:.3f} m/s")
        noise_vrel[sa] = {"mean": np.mean(vals), "std": np.std(vals, ddof=1),
                          "min": np.min(vals), "max": np.max(vals), "all": vals}
        print(f"  → mean={np.mean(vals):.3f} ± {np.std(vals, ddof=1):.3f} m/s")

    # --- 2. Full-info baseline terminal velocity ---
    print()
    print("Full-info SDRE baseline: extracting terminal v_rel")
    v_fi = run_and_get_vrel(0.008, 42, angles_only=False)
    print(f"  Full-info: v_rel={v_fi:.4f} m/s")
    noise_vrel["full_info"] = {"mean": v_fi, "std": 0, "min": v_fi, "max": v_fi, "all": [v_fi]}

    # --- 3. Distance-scale terminal velocities ---
    print()
    print("Distance-scale check (ang 0.008°, seed=42)")
    scales = [
        ("866 km", np.array([500., 500., 500., 0.01, 0.01, 0.01]), 100.0),
        ("86.6 km", np.array([50., 50., 50., 0.01, 0.01, 0.01]), 100.0),
        ("8.66 km", np.array([5., 5., 5., 0.001, 0.001, 0.001]), 10.0),
    ]
    scale_results = {}
    for name, x0, cap_dist in scales:
        # Re-create with custom X_P0
        try:
            sigma_ang_rad = 0.008 * DEG2RAD
            orb = OrbitalDynamics(mu=MU, a_c=A_C, e_c=E_C)
            ctrl = SDREGameController(Q=Q_CTRL, R=R_CTRL, gamma=GAMMA)
            x0_est = x0 - X_E0
            rho0 = float(np.linalg.norm(x0_est[:3]))
            R_meas = np.diag([sigma_ang_rad**2, sigma_ang_rad**2])
            Q_proc = np.diag([PROC_NOISE_POS]*3 + [PROC_NOISE_VEL]*3)
            P0 = np.diag([(rho0*sigma_ang_rad)**2]*3 + [(sigma_ang_rad)**2]*3)
            ekf = RelativeStateEKF(x0=x0_est, P0=P0, Q=Q_proc, R=R_meas, angles_only=True)
            rng = np.random.default_rng(42)
            sim = EKFSDRESimulation(dynamics=orb, controller=ctrl, ekf=ekf,
                                    X_p0=x0, X_e0=X_E0, nu0=NU0, dt=DT,
                                    are_interval=1, rng=rng, capture_dist=cap_dist/1000.0)
            with contextlib.redirect_stdout(io.StringIO()):
                result = sim.run(t_end=10.0 * orb.T_orbit)
            s = result.states
            vrel = float(np.sqrt(np.sum((s[3:6, -1] - s[9:12, -1])**2)) * 1000)
            # Full-info for same scale
            ekf_fi = RelativeStateEKF(x0=x0_est, P0=P0, Q=Q_proc*0, R=R_meas, angles_only=False)
            rng2 = None
            sim_fi = EKFSDRESimulation(dynamics=orb, controller=ctrl, ekf=ekf_fi,
                                       X_p0=x0, X_e0=X_E0, nu0=NU0, dt=DT,
                                       are_interval=1, rng=rng2, capture_dist=cap_dist/1000.0)
            with contextlib.redirect_stdout(io.StringIO()):
                result_fi = sim_fi.run(t_end=10.0 * orb.T_orbit)
            sf = result_fi.states
            vrel_fi = float(np.sqrt(np.sum((sf[3:6, -1] - sf[9:12, -1])**2)) * 1000)
            scale_results[name] = {"ao": vrel, "fi": vrel_fi,
                                   "t_cap_ao": result.t[-1], "t_cap_fi": result_fi.t[-1]}
            print(f"  {name}: AO v_rel={vrel:.3f} m/s (T={result.t[-1]/3600:.1f}h), "
                  f"FI v_rel={vrel_fi:.4f} m/s (T={result_fi.t[-1]/3600:.1f}h)")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
            scale_results[name] = {"ao": np.nan, "fi": np.nan,
                                   "t_cap_ao": np.nan, "t_cap_fi": np.nan}

    # --- Save results ---
    out_path = ROOT / "data" / "terminal_vrel_results.csv"
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sigma_ang", "seed", "terminal_vrel_ms"])
        for sa in sigma_levels:
            for seed, v in zip(seeds, noise_vrel[sa]["all"]):
                writer.writerow([sa, seed, v])
        writer.writerow(["full_info", "n/a", v_fi])
        # Distance scales
        for name, d in scale_results.items():
            writer.writerow([f"scale_{name}", "AO", d["ao"]])
            writer.writerow([f"scale_{name}", "FI", d["fi"]])

    print(f"\nResults saved to {out_path}")

    # --- Print summary table for manuscript ---
    print()
    print("="*70)
    print("SUMMARY: Terminal v_rel for manuscript")
    print("-"*70)
    print(f"{'σ (deg)':>10s}  {'v_rel mean':>10s}  {'v_rel std':>10s}  {'range':>16s}")
    for sa in sigma_levels:
        d = noise_vrel[sa]
        print(f"{sa:10.3f}  {d['mean']:10.3f}  {d['std']:10.3f}  "
              f"{d['min']:.3f}–{d['max']:.3f}")
    print(f"{'Full-info':>10s}  {v_fi:10.4f}")
    print()
    print("Distance scales:")
    for name, d in scale_results.items():
        print(f"  {name}: AO={d['ao']:.3f} m/s, FI={d['fi']:.4f} m/s, "
              f"ratio={d['ao']/d['fi']:.1f}x")
    print("="*70)


if __name__ == "__main__":
    main()
