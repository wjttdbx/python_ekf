"""E0c: validate Euler and nonlinear-flow navigation predictors.

Four paired filters replay the same passive-target truth, controls, initial
error, and angle measurements:

N0: Euler mean + SDC transition
N1: Euler mean + finite-difference discrete Jacobian
N2: NERM RK4 flow + finite-difference flow Jacobian
N3: NERM RK4 flow + square-root UKF

All cases use common physical horizons so results across ``dt`` are directly
comparable. Nominal truth has no process noise, therefore Q=0 is used and the
initial estimation error is sampled from P0.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.integrate import solve_ivp

from aerospace.dynamics.nerm import OrbitalDynamics
from aerospace.estimation.ekf import RelativeStateEKF
from aerospace.estimation.navigators import (
    JacobianEKFNavigator,
    SDCEKFNavigator,
    SquareRootUKFNavigator,
)
from aerospace.paths import DATA_DIR


MASTER_SEED = 20260720
DT_VALUES = (2.0, 5.0, 10.0)
CHECKPOINT_TIMES = (1000.0, 5000.0)
SIGMA_ANG = np.deg2rad(0.008)
R_MEAS = np.diag([SIGMA_ANG**2, SIGMA_ANG**2])
P0_POS = (30.0 * SIGMA_ANG) ** 2
P0_VEL = (0.02 * SIGMA_ANG) ** 2
P0 = np.diag([P0_POS, P0_POS, P0_POS, P0_VEL, P0_VEL, P0_VEL])
Q_ZERO = np.zeros((6, 6))
STATE_SCALES = np.array([30.0, 30.0, 30.0, 0.01, 0.01, 0.01])
B_CTRL = np.vstack([np.zeros((3, 3)), np.eye(3)])
FILTER_NAMES = ("n0_sdc_euler", "n1_jac_euler", "n2_jac_flow", "n3_ukf_flow")


def _initial_case(rng: np.random.Generator):
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    distance = rng.uniform(20.0, 40.0)
    velocity = rng.normal(0.0, 0.01, size=3)
    X_p = np.concatenate([direction * distance, velocity])
    X_e = np.zeros(6)
    nu = float(rng.uniform(0.0, 2.0 * np.pi))
    initial_error = np.linalg.cholesky(P0) @ rng.standard_normal(6)
    return X_p, X_e, nu, initial_error


def _control_at_time(t: float):
    u_p = 1e-6 * np.array([
        np.sin(0.007 * t),
        np.cos(0.005 * t),
        0.5 * np.sin(0.003 * t),
    ])
    return u_p, np.zeros(3)


def _rk4_step(dynamics, state, u_p, u_e, t, dt):
    derivative = dynamics.dynamics_13d
    k1 = derivative(t, state, u_p, u_e)
    k2 = derivative(t + 0.5 * dt, state + 0.5 * dt * k1, u_p, u_e)
    k3 = derivative(t + 0.5 * dt, state + 0.5 * dt * k2, u_p, u_e)
    k4 = derivative(t + dt, state + dt * k3, u_p, u_e)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _build_maps(dynamics, state_k, u_p, u_e, dt):
    X_p = state_k[:6].copy()
    nu = float(state_k[12])
    r_c, nu_dot, nu_ddot = dynamics.get_orbital_params(nu)
    du = u_p - u_e

    def A_fn(x_rel):
        X_e_est = X_p - x_rel
        return dynamics.get_SDC_matrix(X_p, X_e_est, r_c, nu_dot, nu_ddot)

    def euler_map(x_rel):
        return x_rel + dt * (A_fn(x_rel) @ x_rel + B_CTRL @ du)

    def flow_map(x_rel):
        estimated_state = np.concatenate([X_p, X_p - x_rel, [nu]])
        propagated = _rk4_step(
            dynamics, estimated_state, u_p, u_e, 0.0, dt,
        )
        return propagated[:6] - propagated[6:12]

    return A_fn, euler_map, flow_map


def _make_filters(x0):
    return {
        "n0_sdc_euler": SDCEKFNavigator(x0, P0, Q_ZERO, R_MEAS, True),
        "n1_jac_euler": JacobianEKFNavigator(
            x0, P0, Q_ZERO, R_MEAS, True,
            eps=1e-4, state_scales=STATE_SCALES,
        ),
        "n2_jac_flow": JacobianEKFNavigator(
            x0, P0, Q_ZERO, R_MEAS, True,
            eps=1e-4, state_scales=STATE_SCALES,
        ),
        "n3_ukf_flow": SquareRootUKFNavigator(
            x0, P0, Q_ZERO, R_MEAS, True,
            alpha=0.3, beta=2.0, kappa=0.0,
        ),
    }


def _propagate_truth(dynamics, state, u_p, u_e, t, dt):
    solution = solve_ivp(
        dynamics.dynamics_13d,
        [t, t + dt],
        state,
        args=(u_p, u_e),
        method="RK45",
        rtol=1e-9,
        atol=1e-11,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y[:, -1]


def _checkpoint_record(name, navigator, errors, nis_values, nees_values, timings):
    error_array = np.asarray(errors)
    return {
        "filter": name,
        "rmse_pos_m": np.sqrt(np.mean(np.sum(error_array[:, :3] ** 2, axis=1))) * 1000.0,
        "rmse_vel_m_s": np.sqrt(np.mean(np.sum(error_array[:, 3:] ** 2, axis=1))) * 1000.0,
        "final_pos_err_m": np.linalg.norm(error_array[-1, :3]) * 1000.0,
        "final_vel_err_m_s": np.linalg.norm(error_array[-1, 3:]) * 1000.0,
        "nis_mean": np.mean(nis_values),
        "nees_mean": np.mean(nees_values),
        "min_P_eig": np.linalg.eigvalsh(navigator.P).min(),
        "mean_step_time_ms": np.mean(timings) * 1000.0,
        "p95_step_time_ms": np.quantile(timings, 0.95) * 1000.0,
    }


def run_case(dynamics, seed, dt, checkpoint_times):
    rng = np.random.default_rng(MASTER_SEED + seed)
    X_p, X_e, nu, initial_error = _initial_case(rng)
    state = np.concatenate([X_p, X_e, [nu]])
    x_true0 = X_p - X_e
    filters = _make_filters(x_true0 + initial_error)
    max_time = max(checkpoint_times)
    n_steps = int(round(max_time / dt))
    checkpoint_steps = {int(round(time / dt)): time for time in checkpoint_times}
    measurement_noise = rng.multivariate_normal(np.zeros(2), R_MEAS, size=n_steps)

    errors = {name: [] for name in FILTER_NAMES}
    nis_values = {name: [] for name in FILTER_NAMES}
    nees_values = {name: [] for name in FILTER_NAMES}
    timings = {name: [] for name in FILTER_NAMES}
    active = {name: True for name in FILTER_NAMES}
    failure = {name: "" for name in FILTER_NAMES}
    rows = []

    for step in range(1, n_steps + 1):
        t = (step - 1) * dt
        u_p, u_e = _control_at_time(t)
        state_k = state
        A_fn, euler_map, flow_map = _build_maps(
            dynamics, state_k, u_p, u_e, dt,
        )
        state = _propagate_truth(dynamics, state_k, u_p, u_e, t, dt)
        z_true = RelativeStateEKF.measure(state[:6], state[6:12], angle_only=True)
        measurement = z_true + measurement_noise[step - 1]
        x_true = state[:6] - state[6:12]

        for name, navigator in filters.items():
            if not active[name]:
                continue
            discrete_map = euler_map if name in ("n0_sdc_euler", "n1_jac_euler") else flow_map
            A_nominal = A_fn(navigator.x)
            start = perf_counter()
            try:
                x_prior, P_prior = navigator.predict(
                    A_nominal, B_CTRL, u_p, u_e, dt,
                    A_SDC_fn=A_fn,
                    f_discrete=discrete_map,
                )
                innovation = navigator.update(x_prior, P_prior, measurement.copy())
                timings[name].append(perf_counter() - start)
                error = navigator.x - x_true
                errors[name].append(error)
                nis_values[name].append(
                    float(innovation @ np.linalg.solve(navigator._last_S, innovation))
                )
                nees_values[name].append(
                    float(error @ np.linalg.solve(navigator.P, error))
                )
                if not np.all(np.isfinite(navigator.P)):
                    raise FloatingPointError("non-finite covariance")
            except Exception as exc:
                active[name] = False
                failure[name] = f"{type(exc).__name__}: {exc}"

        if step in checkpoint_steps:
            checkpoint_time = checkpoint_steps[step]
            reference = filters["n2_jac_flow"] if active["n2_jac_flow"] else None
            for name, navigator in filters.items():
                base = {
                    "seed": seed,
                    "dt": dt,
                    "time_s": checkpoint_time,
                    "steps": step,
                    "filter": name,
                    "status": "ok" if active[name] else "error",
                    "error": failure[name],
                }
                if active[name]:
                    base.update(_checkpoint_record(
                        name, navigator, errors[name], nis_values[name],
                        nees_values[name], timings[name],
                    ))
                    if reference is not None:
                        delta = navigator.x - reference.x
                        base["vs_flow_jac_pos_m"] = np.linalg.norm(delta[:3]) * 1000.0
                        base["vs_flow_jac_vel_m_s"] = np.linalg.norm(delta[3:]) * 1000.0
                        base["vs_flow_jac_P_rel"] = (
                            np.linalg.norm(navigator.P - reference.P, "fro")
                            / max(np.linalg.norm(reference.P, "fro"), np.finfo(float).tiny)
                        )
                rows.append(base)
    return rows


def run_experiment(output_dir, n_seeds, dt_values, checkpoint_times):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "e0c_flow_map_results.csv"
    fields = [
        "seed", "dt", "time_s", "steps", "filter", "status", "error",
        "rmse_pos_m", "rmse_vel_m_s", "final_pos_err_m", "final_vel_err_m_s",
        "nis_mean", "nees_mean", "min_P_eig",
        "mean_step_time_ms", "p95_step_time_ms",
        "vs_flow_jac_pos_m", "vs_flow_jac_vel_m_s", "vs_flow_jac_P_rel",
    ]
    dynamics = OrbitalDynamics(mu=3.986e5, a_c=15000.0, e_c=0.5)
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for dt in dt_values:
            for seed in range(n_seeds):
                for row in run_case(dynamics, seed, dt, checkpoint_times):
                    writer.writerow(row)
                print(f"  dt={dt:g}s seed {seed + 1}/{n_seeds} complete")
    return output_path


def summarize(results_path, output_dir):
    with results_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    summary = {
        "rows": len(rows),
        "errors": sum(row["status"] != "ok" for row in rows),
        "groups": [],
    }
    groups = defaultdict(list)
    for row in rows:
        if row["status"] == "ok":
            groups[(float(row["dt"]), float(row["time_s"]), row["filter"])].append(row)
    fields = (
        "rmse_pos_m", "rmse_vel_m_s", "final_pos_err_m", "final_vel_err_m_s",
        "nis_mean", "nees_mean", "mean_step_time_ms",
        "vs_flow_jac_pos_m", "vs_flow_jac_vel_m_s", "vs_flow_jac_P_rel",
    )
    for (dt, time_s, filter_name), group_rows in sorted(groups.items()):
        item = {"dt": dt, "time_s": time_s, "filter": filter_name, "n": len(group_rows)}
        for field in fields:
            values = np.array([float(row[field]) for row in group_rows if row[field] != ""])
            if values.size:
                item[f"{field}_median"] = float(np.median(values))
                item[f"{field}_p95"] = float(np.quantile(values, 0.95))
                item[f"{field}_max"] = float(np.max(values))
        summary["groups"].append(item)
    summary_path = output_dir / "e0c_flow_map_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--dt", type=float, nargs="+", default=list(DT_VALUES))
    parser.add_argument(
        "--checkpoints", type=float, nargs="+", default=list(CHECKPOINT_TIMES),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=DATA_DIR / "e0c_flow_map_validation",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_path = run_experiment(
        args.output_dir, args.seeds, tuple(args.dt), tuple(args.checkpoints),
    )
    summary_path = summarize(results_path, args.output_dir)
    print(f"results: {results_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
