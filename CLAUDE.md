# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python-based spacecraft pursuit-evasion game simulation combining nonlinear orbital dynamics (NERM), Extended Kalman Filter (EKF) state estimation, and SDRE optimal control. The pursuer tracks the evader using noisy range-azimuth-elevation measurements, estimates relative state via EKF, then computes optimal thrust commands via SDRE game-theoretic control.

Three orbital dynamics models for spacecraft relative motion in LVLH coordinates:

1. **CW (Clohessy-Wiltshire)**: Linear dynamics for near-circular orbits
2. **LERM (Linearized Elliptical Relative Motion)**: Time-varying linear dynamics for elliptical orbits
3. **NERM (Nonlinear Elliptical Relative Motion)**: Full nonlinear dynamics for elliptical orbits using SDRE approach

Recent research focus: using CasADi+IPOPT direct collocation to solve the Two-Point Boundary Value Problem (TPBVP) as ground-truth optimal control, then comparing SDRE suboptimality and performing inverse optimal fitting to recover state-dependent penalty functions R(x).

## Development Commands

```bash
# Run main simulation (NERM + EKF + SDRE, generates 3 PNG files)
uv run python main.py

# Run ideal simulation (NERM + SDRE, no EKF)
uv run python main_nerm_sdre.py

# Batch multi-scenario simulation (5 scenarios, ECI→LVLH)
uv run python run_scenarios.py

# Root-level experiments
uv run python experiment_followup.py        # Angle-only vs range+angle comparison
uv run python experiment_followup_2d.py     # 2D version
uv run python experiment_q_tuning.py        # Process noise tuning
uv run python experiment_angle_vs_range.py  # Measurement mode comparison
uv run python experiment_inverse_optimal.py # Inverse optimal SDRE fitting
uv run python experiment_tpbvp_compare.py   # TPBVP vs SDRE comparison
uv run python experiment_gain_factor.py     # Kalman gain analysis

# Package-level experiments
uv run python -m aerospace.experiments.monte_carlo
uv run python -m aerospace.experiments.altitude_sweep
uv run python -m aerospace.experiments.eccentricity_sweep
uv run python -m aerospace.experiments.sensor_noise_sweep
uv run python -m aerospace.experiments.compare_tpbvp_vs_sdre
uv run python -m aerospace.experiments.prediction_error_diagnostic
uv run python -m aerospace.experiments.geo_diagnostic
uv run python -m aerospace.experiments.leo_diagnostic
uv run python -m aerospace.experiments.plot_all  # Generate summary plots from CSV data
```

Python 3.12+ required. Use `uv run python` — the system `python` may be 3.8.

Dependencies in `pyproject.toml`: `numpy`, `scipy`, `matplotlib`, `casadi`, `python-docx`. `zhplot` (Chinese matplotlib support) is also used and installed in the venv.

## Architecture

### Package Structure (`aerospace/`)

```
aerospace/
├── dynamics/       # Orbital dynamics models (CW, LERM, NERM, NERM-2D)
├── control/        # Control strategies (SDRE, LQDG, Neural surrogates, TPBVP, inverse optimal)
├── estimation/     # State estimation (EKF 3D and 2D)
├── simulation/     # Closed-loop simulation engines
├── visualization/  # Plotting utilities
├── shared/         # Coordinate transforms (LVLH↔ECI), general visualization
├── experiments/    # Research experiments (sweeps, Monte Carlo, diagnostics)
└── paths.py        # FIGURES_DIR, DATA_DIR, CHECKPOINTS_DIR
```

### Coordinate System
All models use LVLH: x=radial, y=along-track, z=cross-track (orbit normal).

### Dynamics (`aerospace/dynamics/`)

**`nerm.py` — `OrbitalDynamics`** (primary model)
- `get_orbital_params(nu)` → `(r_c, nu_dot, nu_ddot)`: Keplerian parameters at true anomaly
- `get_SDC_matrix(X_p, X_e, r_c, nu_dot, nu_ddot)` → `(6,6)`: State-dependent coefficient matrix for SDRE
- `dynamics_13d(t, state, u_p, u_e)`: ODE RHS for 13D system `[X_p(6), X_e(6), nu(1)]`
- Default: μ=3.986e5 km³/s², a_c=15000 km, e_c=0.5

**`nerm_2d.py` — `OrbitalDynamics2D`**: 9D in-plane simplification `[X_p(4), X_e(4), nu(1)]`

**`cw.py`**: 6D linear circular-orbit model, `get_state_space()` → `(A, B)`

**`lerm.py`**: 6D time-varying linear elliptical model, `get_A_matrix(nu_dot, nu_ddot)` → `(6,6)`

### Control (`aerospace/control/`)

**`sdre.py` — `SDREGameController`** (primary controller)
- `compute_control(A_SDC, x_rel, t, solve_are)` → `(u_p, u_e)`: Optimal thrust for pursuer and evader
- Solves ARE via `scipy.linalg.solve_continuous_are` with symplectic balancing
- `solve_are=False` reuses cached P matrix (sparse updates between timesteps)

**`sdre_2d.py`**: 2D in-plane variant of SDRE game controller

**`tpbvp_collocation.py` — `TPBVPProblem`**: CasADi + IPOPT direct collocation (Hermite-Simpson) solver for the NERM pursuit-evasion TPBVP. Produces ground-truth optimal trajectories `(t, x_traj, u_traj, J)` for 7D augmented state `[x_rel(6), nu(1)]`.

**`optimal_control_solver.py`**: CasADi + IPOPT 2D time-energy optimal control with trapezoidal collocation and normalized time. Outputs optimal `(t, x, u, λ)` with Pontryagin co-states.

**`inverse_optimal_fit.py`**: Given optimal trajectories (from TPBVP), recovers the state-dependent control penalty `R(x)` that makes SDRE feedback match the optimal control.

**`lqdg.py`**: `ChaserController` / `EvaderController` for time-invariant CW systems. Requires M = Rp⁻¹ - Re⁻¹ to be positive definite.

**`neural.py` / `neural_2d.py`**: Neural surrogate controllers (imports from `aerospace.pinn.*` which have been removed — effectively dead code).

### State Estimation (`aerospace/estimation/`)

**`ekf.py` — `RelativeStateEKF`**
- State: relative position+velocity `[dx, dy, dz, dvx, dvy, dvz]` (6D)
- Measurement: range-azimuth-elevation `[ρ, az, el]` or angles-only `[az, el]`
- `angles_only` parameter switches between 3D and 2D measurement models
- `predict(A, B, u_p, u_e, dt)`: Linearized propagation using SDC matrix
- `update(z_meas)` → `y_innov`: Nonlinear measurement update with angle wrapping
- `step(...)`: Combined predict+update
- `measure(X_p, X_e)`: Static method, computes measurements from absolute states
- **Critical**: When `Q=0` and `R≈0` (no noise), skip EKF and use true relative state directly — otherwise `K→0` and estimation drifts

**`ekf_2d.py` — `RelativeStateEKF2D`**: 4D in-plane EKF with range-angle or angle-only measurements.

### Simulation Engines (`aerospace/simulation/`)

**`nerm_ekf_sdre.py` — `EKFSDRESimulation`** (main simulation)
- `__init__(dynamics, controller, ekf, X_p0, X_e0, nu0, dt, are_interval, capture_dist, rng)`
- `run(t_end)` → `EKFSDRESimResult`
- `rng=None` → no-noise mode: skips EKF predict/update, uses true relative state for control
- `EKFSDRESimResult`: `t`, `states (13,N)`, `x_est_history (6,N)`, `u_p/u_e_history (3,N)`, `dist_history`, `ekf_err_history`, `innov_history`, `P_diag_history`, `captured`

**`nerm_sdre.py` — `SDRESimulation`**: Ideal state feedback (no EKF), used by `main_nerm_sdre.py`

Other engines: `nerm_sdre_2d`, `nerm_ekf_sdre_2d`, `cw_sdre`, `cw_lqdg`, `lerm_sdre`, `batch_benchmark`, `batch_benchmark_2d`

### Visualization (`aerospace/visualization/`)

**`ekf_plots.py`**
- `plot_single_simulation(result, orb, title, out_path)`: 6-subplot layout (inertial 3D, LVLH 3D, distance, inertial 2D, EKF error, x-y projection)
- `plot_comparison(result_noisy, result_clean, orb, out_path)`: 3×2 noisy vs ideal comparison

### Shared Utilities (`aerospace/shared/`)

**`coord_transform.py`**: `lvlh_to_eci`, `chief_eci_pos`, `eci_to_lvlh_state`, `keplerian_to_eci`, `plot_eci_trajectory` — used by `run_scenarios.py` and `main_nerm_sdre.py` for ECI↔LVLH conversions.

### Experiments (`aerospace/experiments/`)

- **Sweeps**: `altitude_sweep.py`, `eccentricity_sweep.py`, `sensor_noise_sweep.py` — parameter sweeps measuring capture rate vs orbital parameters
- **Monte Carlo**: `monte_carlo.py` — statistical performance analysis with random seeds
- **Diagnostics**: `geo_diagnostic.py`, `leo_diagnostic.py`, `prediction_error_diagnostic.py` — per-seed analysis for specific orbit regimes
- **TPBVP**: `compare_tpbvp_vs_sdre.py` — SDRE suboptimality quantification against ground-truth TPBVP solutions
- **Plotting**: `plot_all.py` — reads CSV results and generates summary figures

## Key Relationships

- SDRE control operates on 6D **relative** state `x_rel = X_p - X_e`; the 13D absolute state is only for ODE propagation
- `get_SDC_matrix` 每步调用一次，算出的 `A_SDC` 同时用于 SDRE 控制和 EKF 预测——滤波与控制共享同一线性化点 x̂_{k|k}
- `nu` evolves via `dnu/dt = nu_dot(nu)` embedded in `dynamics_13d`
- Control input enters as differential acceleration: `B @ (u_p - u_e)` where `B = [0₃; I₃]`
- LERM degenerates to CW when `nu_ddot=0`, `nu_dot=n` (circular limit)
- TPBVP solvers (CasADi+IPOPT) provide ground-truth optimal solutions for validating SDRE suboptimality; inverse optimal fitting then recovers R(x) that best matches the optimal control

## Important Notes

- **State dimensions**: CW/LERM use 6D relative state; NERM uses 13D absolute; NERM-2D uses 9D
- **No `from __future__ import annotations` needed**: project targets Python 3.12+
- **Angle wrapping**: EKF innovation `y[1:3] = wrap_angle(y[1:3])` is required for azimuth/elevation
- **Simulation defaults**: dt=10s, t_end=10×T_orbit (~50h for e=0.5, a=15000km orbit)
- **Noise parameters**: σ_ang=0.008° (angles), σ_range=10m (range); process noise: pos 5×10⁻⁴ km², vel 5×10⁻⁸ km²/s²
- **Angles-only mode**: EKF supports 2D measurement `[az, el]` (no range) — used to study observability/degradation
- **zhplot**: Used across experiment scripts for matplotlib Chinese font support; installed in project venv
