# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python-based spacecraft pursuit-evasion game simulation combining nonlinear orbital dynamics (NERM), Extended Kalman Filter (EKF) state estimation, and SDRE optimal control. The pursuer tracks the evader using noisy range-azimuth-elevation measurements, estimates relative state via EKF, then computes optimal thrust commands via SDRE game-theoretic control.

Three orbital dynamics models for spacecraft relative motion in LVLH coordinates:

1. **CW (Clohessy-Wiltshire)**: Linear dynamics for near-circular orbits
2. **LERM (Linearized Elliptical Relative Motion)**: Time-varying linear dynamics for elliptical orbits
3. **NERM (Nonlinear Elliptical Relative Motion)**: Full nonlinear dynamics for elliptical orbits using SDRE approach

## Development Commands

```bash
# Run main simulation (NERM + EKF + SDRE, generates 3 PNG files)
uv run python main.py

# Run ideal simulation (NERM + SDRE, no EKF)
uv run python main_nerm_sdre.py
```

Python 3.12+ required. Use `uv run python` — the system `python` may be 3.8.

## Architecture

### Package Structure (`aerospace/`)

```
aerospace/
├── dynamics/       # Orbital dynamics models (CW, LERM, NERM, NERM-2D)
├── control/        # Control strategies (SDRE, LQDG, Neural/PINN surrogates)
├── estimation/     # State estimation (EKF)
├── simulation/     # Closed-loop simulation engines
├── visualization/  # Plotting utilities
├── shared/         # Coordinate transforms, general visualization
├── pinn/           # PINN training, data generation, model export
├── experiments/    # Research experiments and comparisons
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

**`sdre.py` — `SDREGameController`**
- `compute_control(A_SDC, x_rel, t, solve_are)` → `(u_p, u_e)`: Optimal thrust for pursuer and evader
- Solves ARE via `scipy.linalg.solve_continuous_are` with symplectic balancing for numerical stability
- `solve_are=False` reuses cached P matrix (sparse updates between timesteps)

**`lqdg.py`**: `ChaserController` / `EvaderController` for time-invariant CW systems. Requires M = Rp⁻¹ - Re⁻¹ to be positive definite.

**`neural.py` / `neural_2d.py` — `NeuralSDREController`**: PINN surrogate for ARE solving. Log-Cholesky parameterization for SPD output. Same interface as `SDREGameController`. Includes OOD detection with fallback to classical ARE.

### State Estimation (`aerospace/estimation/`)

**`ekf.py` — `RelativeStateEKF`**
- State: relative position+velocity `[dx, dy, dz, dvx, dvy, dvz]` (6D)
- Measurement: range-azimuth-elevation `[ρ, az, el]`
- `predict(A, B, u_p, u_e, dt)`: Linearized propagation using SDC matrix
- `update(z_meas)` → `y_innov`: Nonlinear measurement update with angle wrapping
- `step(...)`: Combined predict+update
- `measure(X_p, X_e)`: Static method, computes `[ρ, az, el]` from absolute states
- **Critical**: When `Q=0` and `R≈0` (no noise), skip EKF and use true relative state directly — otherwise `K→0` and estimation drifts

### Simulation Engines (`aerospace/simulation/`)

**`nerm_ekf_sdre.py` — `EKFSDRESimulation`** (main simulation)
- `__init__(dynamics, controller, ekf, X_p0, X_e0, nu0, dt, are_interval, capture_dist, rng)`
- `run(t_end)` → `EKFSDRESimResult`
- `rng=None` → no-noise mode: skips EKF predict/update, uses true relative state for control
- `EKFSDRESimResult`: `t`, `states (13,N)`, `x_est_history (6,N)`, `u_p/u_e_history (3,N)`, `dist_history`, `ekf_err_history`, `captured`

**`nerm_sdre.py` — `SDRESimulation`**: Ideal state feedback (no EKF), used by `main_nerm_sdre.py`

Other engines: `nerm_sdre_2d`, `cw_sdre`, `cw_lqdg`, `lerm_sdre`, `batch_benchmark`, `batch_benchmark_2d`

### Visualization (`aerospace/visualization/`)

**`ekf_plots.py`**
- `plot_single_simulation(result, orb, title, out_path)`: 6-subplot layout (inertial 3D, LVLH 3D, distance, inertial 2D, EKF error, x-y projection)
- `plot_comparison(result_noisy, result_clean, orb, out_path)`: 3×2 noisy vs ideal comparison

### Shared Utilities (`aerospace/shared/`)

**`coord_transform.py`**: `lvlh_to_eci`, `chief_eci_pos`, `plot_eci_trajectory` — used by `main_nerm_sdre.py` for ECI trajectory plots

## Key Relationships

- SDRE control operates on 6D **relative** state `x_rel = X_p - X_e`; the 13D absolute state is only for ODE propagation
- `get_SDC_matrix` is called twice per timestep: once for control (using estimated state), once for EKF prediction (using updated state after propagation)
- `nu` evolves via `dnu/dt = nu_dot(nu)` embedded in `dynamics_13d`
- Control input enters as differential acceleration: `B @ (u_p - u_e)` where `B = [0₃; I₃]`
- LERM degenerates to CW when `nu_ddot=0`, `nu_dot=n` (circular limit)

## Important Notes

- **State dimensions**: CW/LERM use 6D relative state; NERM uses 13D absolute; NERM-2D uses 9D
- **No `from __future__ import annotations` needed**: project targets Python 3.12+
- **Angle wrapping**: EKF innovation `y[1:3] = wrap_angle(y[1:3])` is required for azimuth/elevation
- **Simulation defaults**: dt=10s, t_end=10×T_orbit (~50h for e=0.5, a=15000km orbit)
- **Noise parameters**: σ_r=10m (range), σ_az=σ_el=10⁻⁴ rad; process noise: pos 5×10⁻⁴ km², vel 5×10⁻⁸ km²/s²
