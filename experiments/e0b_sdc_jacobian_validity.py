"""E0b: isolate SDC-EKF covariance approximation from process-noise masking.

Part A compares the SDC transition ``I + A(x) dt`` with the finite-difference
Jacobian of the same discrete map for several process-noise scales. Part B
replays identical truth, controls, and angle measurements through paired
SDC-EKF and Jacobian-EKF instances for 100 and 500 steps.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from aerospace.dynamics.nerm import OrbitalDynamics
from aerospace.estimation.ekf import RelativeStateEKF
from aerospace.estimation.navigators import JacobianEKFNavigator, SDCEKFNavigator
from aerospace.paths import DATA_DIR


MASTER_SEED = 20260714
DT_VALUES = (2.0, 5.0, 10.0)
Q_SCALES = (0.0, 0.01, 0.1, 1.0)
REPLAY_DT = 10.0
REPLAY_CHECKPOINTS = (100, 500)
SIGMA_ANG = np.deg2rad(0.008)
R_MEAS = np.diag([SIGMA_ANG**2, SIGMA_ANG**2])
Q_BASE = np.diag([5e-4, 5e-4, 5e-4, 5e-8, 5e-8, 5e-8])
P0_POS = (30.0 * SIGMA_ANG) ** 2
P0_VEL = (0.02 * SIGMA_ANG) ** 2
P0 = np.diag([P0_POS, P0_POS, P0_POS, P0_VEL, P0_VEL, P0_VEL])
STATE_SCALES = np.array([30.0, 30.0, 30.0, 0.01, 0.01, 0.01])
B_CTRL = np.vstack([np.zeros((3, 3)), np.eye(3)])


def _initial_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    distance = rng.uniform(20.0, 40.0)
    velocity = rng.normal(0.0, 0.01, size=3)
    x_true = np.concatenate([direction * distance, velocity])
    initial_error = np.linalg.cholesky(P0) @ rng.standard_normal(6)
    return x_true.copy(), np.zeros(6), float(rng.uniform(0.0, 2.0 * np.pi)), initial_error


def _control_at_step(step: int) -> tuple[np.ndarray, np.ndarray]:
    phase = float(step)
    u_p = 1e-6 * np.array([
        np.sin(0.07 * phase),
        np.cos(0.05 * phase),
        0.5 * np.sin(0.03 * phase),
    ])
    return u_p, np.zeros(3)


def _context(
    dynamics: OrbitalDynamics,
    X_p_true: np.ndarray,
    r_c: float,
    nu_dot: float,
    nu_ddot: float,
    u_p: np.ndarray,
    u_e: np.ndarray,
    dt: float,
):
    du = u_p - u_e

    def A_SDC_fn(x_rel: np.ndarray) -> np.ndarray:
        X_e_est = X_p_true - x_rel
        return dynamics.get_SDC_matrix(X_p_true, X_e_est, r_c, nu_dot, nu_ddot)

    def f_discrete(x_rel: np.ndarray) -> np.ndarray:
        A = A_SDC_fn(x_rel)
        return x_rel + dt * (A @ x_rel + B_CTRL @ du)

    return A_SDC_fn, f_discrete


def run_matrix_isolation(
    output_dir: Path,
    n_seeds: int,
) -> Path:
    dynamics = OrbitalDynamics(mu=3.986e5, a_c=15000.0, e_c=0.5)
    rng = np.random.default_rng(MASTER_SEED)
    output_path = output_dir / "e0b_matrix_isolation.csv"
    fields = [
        "seed", "dt", "q_scale",
        "F_rel_fro", "F_max_abs",
        "P_rel_fro", "P_max_abs", "P_trace_rel",
        "mean_pos_diff_m", "mean_vel_diff_m_s",
    ]

    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for seed in range(n_seeds):
            X_p, X_e, nu, initial_error = _initial_case(rng)
            x_est = X_p - X_e + initial_error
            u_p, u_e = _control_at_step(seed)
            r_c, nu_dot, nu_ddot = dynamics.get_orbital_params(nu)
            A_fn, f_discrete = _context(
                dynamics, X_p, r_c, nu_dot, nu_ddot, u_p, u_e, 1.0,
            )

            for dt in DT_VALUES:
                A_fn, f_discrete = _context(
                    dynamics, X_p, r_c, nu_dot, nu_ddot, u_p, u_e, dt,
                )
                A_nominal = A_fn(x_est)
                F_sdc = np.eye(6) + A_nominal * dt
                for q_scale in Q_SCALES:
                    Q = Q_BASE * q_scale
                    nav_sdc = SDCEKFNavigator(x_est, P0, Q, R_MEAS, True)
                    nav_jac = JacobianEKFNavigator(
                        x_est, P0, Q, R_MEAS, True,
                        eps=1e-4, state_scales=STATE_SCALES,
                    )
                    x_sdc, P_sdc = nav_sdc.predict(
                        A_nominal, B_CTRL, u_p, u_e, dt,
                        A_SDC_fn=A_fn, f_discrete=f_discrete,
                    )
                    x_jac, P_jac = nav_jac.predict(
                        A_nominal, B_CTRL, u_p, u_e, dt,
                        A_SDC_fn=A_fn, f_discrete=f_discrete,
                    )
                    F_jac = nav_jac._last_F
                    F_delta = F_sdc - F_jac
                    P_delta = P_sdc - P_jac
                    writer.writerow({
                        "seed": seed,
                        "dt": dt,
                        "q_scale": q_scale,
                        "F_rel_fro": np.linalg.norm(F_delta, "fro") / max(np.linalg.norm(F_jac, "fro"), np.finfo(float).tiny),
                        "F_max_abs": np.max(np.abs(F_delta)),
                        "P_rel_fro": np.linalg.norm(P_delta, "fro") / max(np.linalg.norm(P_jac, "fro"), np.finfo(float).tiny),
                        "P_max_abs": np.max(np.abs(P_delta)),
                        "P_trace_rel": abs(np.trace(P_sdc) - np.trace(P_jac)) / max(abs(np.trace(P_jac)), np.finfo(float).tiny),
                        "mean_pos_diff_m": np.linalg.norm(x_sdc[:3] - x_jac[:3]) * 1000.0,
                        "mean_vel_diff_m_s": np.linalg.norm(x_sdc[3:] - x_jac[3:]) * 1000.0,
                    })
    return output_path


def _truth_replay(
    dynamics: OrbitalDynamics,
    X_p0: np.ndarray,
    X_e0: np.ndarray,
    nu0: float,
    measurement_noise: np.ndarray,
    max_steps: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    states = [np.concatenate([X_p0, X_e0, [nu0]])]
    controls = []
    measurements = []
    for step in range(max_steps):
        u_p, u_e = _control_at_step(step)
        controls.append(np.concatenate([u_p, u_e]))
        state = states[-1]
        solution = solve_ivp(
            dynamics.dynamics_13d,
            [step * REPLAY_DT, (step + 1) * REPLAY_DT],
            state,
            args=(u_p, u_e),
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
        )
        next_state = solution.y[:, -1]
        states.append(next_state)
        z_true = RelativeStateEKF.measure(
            next_state[:6], next_state[6:12], angle_only=True,
        )
        measurements.append(z_true + measurement_noise[step])
    return states, controls, measurements


def _paired_replay(
    dynamics: OrbitalDynamics,
    states: list[np.ndarray],
    controls: list[np.ndarray],
    measurements: list[np.ndarray],
    initial_error: np.ndarray,
    q_scale: float,
    checkpoints: tuple[int, ...],
) -> list[dict]:
    x_true0 = states[0][:6] - states[0][6:12]
    x_est0 = x_true0 + initial_error
    Q = Q_BASE * q_scale
    navigators = {
        "sdc": SDCEKFNavigator(x_est0, P0, Q, R_MEAS, True),
        "jac": JacobianEKFNavigator(
            x_est0, P0, Q, R_MEAS, True,
            eps=1e-4, state_scales=STATE_SCALES,
        ),
    }
    errors = {name: [] for name in navigators}
    nis = {name: [] for name in navigators}
    records = []

    for step, (control, measurement) in enumerate(zip(controls, measurements), start=1):
        state_k = states[step - 1]
        state_next = states[step]
        u_p, u_e = control[:3], control[3:]
        r_c, nu_dot, nu_ddot = dynamics.get_orbital_params(state_k[12])

        for name, navigator in navigators.items():
            A_fn, f_discrete = _context(
                dynamics, state_k[:6], r_c, nu_dot, nu_ddot,
                u_p, u_e, REPLAY_DT,
            )
            A_nominal = A_fn(navigator.x)
            x_prior, P_prior = navigator.predict(
                A_nominal, B_CTRL, u_p, u_e, REPLAY_DT,
                A_SDC_fn=A_fn, f_discrete=f_discrete,
            )
            innovation = navigator.update(x_prior, P_prior, measurement.copy())
            innovation_cov = navigator._last_S
            nis[name].append(float(innovation @ np.linalg.solve(innovation_cov, innovation)))
            x_true = state_next[:6] - state_next[6:12]
            errors[name].append(navigator.x - x_true)

        if step in checkpoints:
            error_sdc = np.asarray(errors["sdc"])
            error_jac = np.asarray(errors["jac"])
            state_delta = navigators["sdc"].x - navigators["jac"].x
            covariance_delta = navigators["sdc"].P - navigators["jac"].P
            records.append({
                "steps": step,
                "sdc_rmse_pos_m": float(np.sqrt(np.mean(np.sum(error_sdc[:, :3]**2, axis=1))) * 1000.0),
                "jac_rmse_pos_m": float(np.sqrt(np.mean(np.sum(error_jac[:, :3]**2, axis=1))) * 1000.0),
                "sdc_rmse_vel_m_s": float(np.sqrt(np.mean(np.sum(error_sdc[:, 3:]**2, axis=1))) * 1000.0),
                "jac_rmse_vel_m_s": float(np.sqrt(np.mean(np.sum(error_jac[:, 3:]**2, axis=1))) * 1000.0),
                "state_pos_diff_m": float(np.linalg.norm(state_delta[:3]) * 1000.0),
                "state_vel_diff_m_s": float(np.linalg.norm(state_delta[3:]) * 1000.0),
                "P_rel_fro": float(np.linalg.norm(covariance_delta, "fro") / max(np.linalg.norm(navigators["jac"].P, "fro"), np.finfo(float).tiny)),
                "nis_mean_abs_diff": float(abs(np.mean(nis["sdc"]) - np.mean(nis["jac"]))),
                "sdc_min_P_eig": float(np.linalg.eigvalsh(navigators["sdc"].P).min()),
                "jac_min_P_eig": float(np.linalg.eigvalsh(navigators["jac"].P).min()),
            })
    return records


def run_replay(
    output_dir: Path,
    n_seeds: int,
    max_steps: int,
) -> Path:
    dynamics = OrbitalDynamics(mu=3.986e5, a_c=15000.0, e_c=0.5)
    output_path = output_dir / "e0b_fixed_replay.csv"
    fields = [
        "seed", "q_scale", "steps", "status", "error",
        "sdc_rmse_pos_m", "jac_rmse_pos_m",
        "sdc_rmse_vel_m_s", "jac_rmse_vel_m_s",
        "state_pos_diff_m", "state_vel_diff_m_s",
        "P_rel_fro", "nis_mean_abs_diff",
        "sdc_min_P_eig", "jac_min_P_eig",
    ]
    checkpoints = tuple(step for step in REPLAY_CHECKPOINTS if step <= max_steps)
    if max_steps not in checkpoints:
        checkpoints = tuple(sorted(set(checkpoints + (max_steps,))))

    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for seed in range(n_seeds):
            rng = np.random.default_rng(MASTER_SEED + 10000 + seed)
            X_p, X_e, nu, initial_error = _initial_case(rng)
            measurement_noise = rng.multivariate_normal(
                np.zeros(2), R_MEAS, size=max_steps,
            )
            states, controls, measurements = _truth_replay(
                dynamics, X_p, X_e, nu, measurement_noise, max_steps,
            )
            for q_scale in Q_SCALES:
                try:
                    records = _paired_replay(
                        dynamics, states, controls, measurements,
                        initial_error, q_scale, checkpoints,
                    )
                    for record in records:
                        writer.writerow({
                            "seed": seed,
                            "q_scale": q_scale,
                            "status": "ok",
                            "error": "",
                            **record,
                        })
                except Exception as exc:
                    writer.writerow({
                        "seed": seed,
                        "q_scale": q_scale,
                        "steps": max_steps,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
            print(f"  replay seed {seed + 1}/{n_seeds} complete")
    return output_path


def _read_numeric(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def summarize(matrix_path: Path, replay_path: Path, output_dir: Path) -> Path:
    matrix_rows = _read_numeric(matrix_path)
    replay_rows = _read_numeric(replay_path)
    summary: dict[str, object] = {
        "matrix_rows": len(matrix_rows),
        "replay_rows": len(replay_rows),
        "replay_errors": sum(row["status"] != "ok" for row in replay_rows),
        "matrix": [],
        "replay": [],
    }

    matrix_groups: dict[tuple[float, float], list[dict]] = defaultdict(list)
    for row in matrix_rows:
        matrix_groups[(float(row["dt"]), float(row["q_scale"]))].append(row)
    for (dt, q_scale), rows in sorted(matrix_groups.items()):
        item = {"dt": dt, "q_scale": q_scale}
        for field in ("F_rel_fro", "P_rel_fro", "P_trace_rel"):
            values = np.array([float(row[field]) for row in rows])
            item[f"{field}_median"] = float(np.median(values))
            item[f"{field}_p95"] = float(np.quantile(values, 0.95))
            item[f"{field}_max"] = float(np.max(values))
        summary["matrix"].append(item)

    replay_groups: dict[tuple[float, int], list[dict]] = defaultdict(list)
    for row in replay_rows:
        if row["status"] == "ok":
            replay_groups[(float(row["q_scale"]), int(float(row["steps"])))].append(row)
    replay_fields = (
        "state_pos_diff_m", "state_vel_diff_m_s", "P_rel_fro",
        "nis_mean_abs_diff", "sdc_rmse_pos_m", "jac_rmse_pos_m",
    )
    for (q_scale, steps), rows in sorted(replay_groups.items()):
        item = {"q_scale": q_scale, "steps": steps, "n": len(rows)}
        for field in replay_fields:
            values = np.array([float(row[field]) for row in rows])
            item[f"{field}_median"] = float(np.median(values))
            item[f"{field}_p95"] = float(np.quantile(values, 0.95))
            item[f"{field}_max"] = float(np.max(values))
        summary["replay"].append(item)

    output_path = output_dir / "e0b_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-seeds", type=int, default=100)
    parser.add_argument("--replay-seeds", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--output-dir", type=Path,
        default=DATA_DIR / "e0b_sdc_jacobian_validity",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("E0b matrix isolation")
    matrix_path = run_matrix_isolation(args.output_dir, args.matrix_seeds)
    print("E0b fixed-measurement replay")
    replay_path = run_replay(args.output_dir, args.replay_seeds, args.max_steps)
    summary_path = summarize(matrix_path, replay_path, args.output_dir)
    print(f"matrix:  {matrix_path}")
    print(f"replay:  {replay_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()