"""
E2a 法向几何一致性逐步诊断

目标：定位 crosstrack 低噪声 NIS/NEES 异常的根因
  (a) 指标/bug（P0 transform, S/P 计算, wrap）
  (b) FD Jacobian 步长精度（state_scales 缺失）
  (c) Q=0 未建模误差地板
  (d) 几何可观测性退化
"""

from __future__ import annotations
import csv, json, os, sys, warnings
from pathlib import Path
import numpy as np

from aerospace.dynamics.nerm import OrbitalDynamics
from aerospace.control.sdre import SDREGameController
from aerospace.simulation.nerm_ekf_sdre import EKFSDRESimulation
from aerospace.estimation.navigators import create_navigator, JacobianEKFNavigator
from aerospace.paths import DATA_DIR

warnings.filterwarnings("ignore")
DEG2RAD = np.pi / 180.0

# Reuse E2 config
from experiments.e2_flow_isolated_n10 import (
    make_initial_state, los_p0, generate_x0_errors, FixedNoiseRNG,
    generate_paired_noise_sequences, seed_from_str,
    MU, A_C, E_C, DEG2RAD, CAPTURE_DIST, V_THRESH, SUSTAIN_DUR,
    T_ORBIT_FACTOR, NAV_TYPE, N_SEEDS, MASTER_SEED, SIGMA_THETA_VALUES,
)


def run_diagnostic(geom: str, seeds: list[int], sigma_deg: float,
                   state_scales: np.ndarray | None = None,
                   dt: float = 10.0,
                   nav_type: str = "jacobian",
                   output_dir: Path | None = None,
                   max_steps: int | None = None,
                   ) -> dict:
    """Run simulation with per-step NEES/NIS/condition logging.

    Returns a dict with per-step arrays.
    """
    orb = OrbitalDynamics(mu=MU, a_c=A_C, e_c=E_C)
    t_end = T_ORBIT_FACTOR * orb.T_orbit
    if max_steps is not None:
        t_end = min(t_end, max_steps * dt)

    n_steps = int(t_end / dt) + 2
    noise_seqs = generate_paired_noise_sequences(N_SEEDS, n_steps, 2)

    X_p0, X_e0 = make_initial_state(geom, dist_km=30.0)
    P0 = los_p0(X_p0 - X_e0)
    x0_errs = generate_x0_errors(N_SEEDS, P0, seed_offset=seed_from_str(geom))
    nu0 = 0.0

    sigma_rad = max(sigma_deg * DEG2RAD, 1e-12)
    R_meas = np.diag([sigma_rad**2, sigma_rad**2])
    Q_proc = np.zeros((6, 6))

    all_results = {}

    for seed in seeds:
        x0_err = x0_errs[seed]
        x_true_rel0 = X_p0 - X_e0
        x_est0 = x_true_rel0 + x0_err

        # Build navigator with optional state_scales
        if nav_type == "jacobian":
            nav = JacobianEKFNavigator(
                x0=x_est0, P0=P0, Q=Q_proc, R=R_meas,
                angles_only=True, eps=1e-4, state_scales=state_scales,
            )
        elif nav_type == "ukf":
            nav = create_navigator("ukf", x0=x_est0, P0=P0, Q=Q_proc, R=R_meas,
                                   angles_only=True)
        else:
            nav = create_navigator(nav_type, x0=x_est0, P0=P0, Q=Q_proc, R=R_meas,
                                   angles_only=True)

        ctrl = SDREGameController(Q=np.eye(6), R=np.eye(3)*1e13, gamma=np.sqrt(2))

        noise_seq = noise_seqs[seed]
        noise_scaled = np.zeros((n_steps, R_meas.shape[0]))
        for i in range(min(len(noise_seq), n_steps)):
            noise_scaled[i] = noise_seq[i] * sigma_rad
        rng = FixedNoiseRNG(noise_scaled, R_meas.shape[0])

        # ── Manual simulation loop with per-step logging ───────────
        state = np.zeros(13)
        state[0:6] = X_p0
        state[6:12] = X_e0
        state[12] = nu0
        t = 0.0
        N = int(t_end / dt)

        log = {
            "t": [],
            "nis": [], "nees": [],
            "P_trace": [], "P_cond": [], "P_eig_min": [], "P_eig_max": [],
            "F_cond": [], "F_norm": [],
            "fd_h": [],  # FD step sizes
            "innov_norm": [],
            "err_pos_km": [], "err_vel_kms": [],
            "dist_km": [],
            "x_est_pos": [],
        }

        for k in range(N):
            nu = float(state[12])
            r_c, nu_dot, nu_ddot = orb.get_orbital_params(nu)

            # Control
            x_ctrl = nav.x
            X_p_true = state[0:6]
            X_e_est = X_p_true - nav.x
            A_SDC_ctrl = orb.get_SDC_matrix(X_p_true, X_e_est, r_c, nu_dot, nu_ddot)
            u_p, u_e = ctrl.compute_control(A_SDC_ctrl, x_ctrl, t=t,
                                            solve_are=(k % 1 == 0))
            u_e = np.zeros(3)  # passive evader

            # Navigator predict
            du = u_p - u_e
            B_ctrl = np.zeros((6, 3))
            B_ctrl[3:, :] = np.eye(3)

            def flow_discrete(x_rel):
                est_state = np.concatenate([X_p_true, X_p_true - x_rel, [nu]])
                dt_local = dt
                k1 = orb.dynamics_13d(0.0, est_state, u_p, u_e)
                k2 = orb.dynamics_13d(0.5*dt_local, est_state + 0.5*dt_local*k1, u_p, u_e)
                k3 = orb.dynamics_13d(0.5*dt_local, est_state + 0.5*dt_local*k2, u_p, u_e)
                k4 = orb.dynamics_13d(dt_local, est_state + dt_local*k3, u_p, u_e)
                prop = est_state + (dt_local/6.0)*(k1 + 2*k2 + 2*k3 + k4)
                return prop[0:6] - prop[6:12]

            def A_SDC_fn(x_rel):
                X_e_est_local = X_p_true - x_rel
                return orb.get_SDC_matrix(X_p_true, X_e_est_local, r_c, nu_dot, nu_ddot)

            A_SDC_nominal = A_SDC_fn(nav.x)
            x_priori, P_priori = nav.predict(A_SDC_nominal, B_ctrl, u_p, u_e, dt,
                                             f_discrete=flow_discrete,
                                             A_SDC_fn=A_SDC_fn)

            # Log FD step sizes (for jacobian navigator)
            fd_h = []
            if hasattr(nav, 'eps'):
                x_curr = nav._x  # pre-predict state
                eps_val = getattr(nav, 'eps', 1e-4)
                scales = getattr(nav, '_scales', None)
                for i in range(6):
                    if scales is not None:
                        s = scales[i]
                    else:
                        s = max(abs(x_curr[i]), 1e-8)
                    h = eps_val * s
                    if h < 1e-15:
                        h = eps_val * 1e-6
                    fd_h.append(h)
            log["fd_h"].append(fd_h)

            # True state propagation
            from scipy.integrate import solve_ivp
            sol = solve_ivp(orb.dynamics_13d, [t, t+dt], state,
                            args=(u_p, u_e), method="RK45",
                            rtol=1e-8, atol=1e-10)
            state = sol.y[:, -1]
            t += dt

            # Measurement + update
            z_true = np.array([
                np.arctan2(state[1]-state[7], state[0]-state[6]),
                np.arcsin(np.clip((state[2]-state[8]) /
                         (np.linalg.norm(state[0:3]-state[6:9]) + 1e-12), -1, 1)),
            ])
            z_meas = z_true + rng.multivariate_normal(np.zeros(2), R_meas)

            innov = nav.update(x_priori, P_priori, z_meas)
            S = getattr(nav, '_last_S', None)
            nis = np.nan
            nees = np.nan
            x_true = state[0:6] - state[6:12]
            err = nav.x - x_true

            if S is not None:
                try:
                    nis = float(innov @ np.linalg.solve(S, innov))
                except np.linalg.LinAlgError:
                    pass
            try:
                nees = float(err @ np.linalg.solve(nav.P, err))
            except np.linalg.LinAlgError:
                pass

            # Log
            log["t"].append(t)
            log["nis"].append(nis)
            log["nees"].append(nees)
            log["P_trace"].append(float(np.trace(nav.P)))
            log["P_cond"].append(float(np.linalg.cond(nav.P)))
            eigvals = np.linalg.eigvalsh(nav.P)
            log["P_eig_min"].append(float(np.min(eigvals)))
            log["P_eig_max"].append(float(np.max(eigvals)))
            log["F_cond"].append(float(np.linalg.cond(getattr(nav, '_last_F', np.eye(6)))))
            log["F_norm"].append(float(np.linalg.norm(getattr(nav, '_last_F', np.eye(6)), 'fro')))
            log["innov_norm"].append(float(np.linalg.norm(innov)))
            log["err_pos_km"].append(float(np.linalg.norm(err[:3])))
            log["err_vel_kms"].append(float(np.linalg.norm(err[3:])))
            log["dist_km"].append(float(np.linalg.norm(state[0:3] - state[6:9])))
            log["x_est_pos"].append(nav.x[:3].copy())

        all_results[seed] = log
        print(f"  seed={seed}: NEES mean={np.nanmean(log['nees']):.1f}, "
              f"final pos err={log['err_pos_km'][-1]*1000:.1f} m, "
              f"final P_trace={log['P_trace'][-1]:.2e}",
              flush=True)

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# Main diagnostic
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    output_dir = DATA_DIR / "e2a_crosstrack_consistency"
    output_dir.mkdir(parents=True, exist_ok=True)

    geom = "crosstrack"
    seeds_to_test = [1, 9, 3]  # worst, best, median
    dt_test = 10.0

    print("=" * 60)
    print("E2a Phase 1: Per-step diagnostic (crosstrack, sigma=0.001)")
    print(f"Seeds: {seeds_to_test}")
    print("=" * 60)

    # ── Test 1: Baseline (default Jacobian, no state_scales) ──────
    print("\n--- Baseline: default Jacobian (no state_scales) ---")
    results_default = run_diagnostic(
        geom, seeds_to_test, 0.001, dt=dt_test,
        nav_type="jacobian", state_scales=None,
        output_dir=output_dir,
    )

    # ── Test 2: Jacobian with fixed state_scales ──────────────────
    print("\n--- Fixed state_scales: [1,1,1,0.01,0.01,0.01] ---")
    fixed_scales = np.array([1.0, 1.0, 1.0, 0.01, 0.01, 0.01])
    results_fixed = run_diagnostic(
        geom, seeds_to_test, 0.001, dt=dt_test,
        nav_type="jacobian", state_scales=fixed_scales,
        output_dir=output_dir,
    )

    # ── Test 3: UKF ──────────────────────────────────────────────
    print("\n--- UKF (flow-map sigma points) ---")
    results_ukf = run_diagnostic(
        geom, seeds_to_test, 0.001, dt=dt_test,
        nav_type="ukf", state_scales=None,
        output_dir=output_dir,
    )

    # ── Save per-step logs ──────────────────────────────────────
    for name, results in [("default", results_default),
                           ("fixed_scales", results_fixed),
                           ("ukf", results_ukf)]:
        for seed, log in results.items():
            np.savez_compressed(
                output_dir / f"diag_{name}_seed{seed}.npz",
                **{k: np.array(v, dtype=object) if k == 'fd_h' else np.array(v)
                   for k, v in log.items()}
            )

    # ── Summary table ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Summary: NEES mean / final pos err (m) / final P_trace")
    print("=" * 60)
    for name, results in [("default", results_default),
                           ("fixed_scales", results_fixed),
                           ("ukf", results_ukf)]:
        print(f"\n{name}:")
        for seed, log in results.items():
            nees_m = np.nanmean(log['nees'])
            nis_m = np.nanmean(log['nis'])
            fpe = log['err_pos_km'][-1] * 1000
            pt = log['P_trace'][-1]
            print(f"  seed={seed}: NEES={nees_m:.1f}, NIS={nis_m:.2f}, "
                  f"final_pos_err={fpe:.1f}m, P_trace={pt:.2e}")

    print(f"\nResults saved to {output_dir}")
