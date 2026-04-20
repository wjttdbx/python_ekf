# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python-based spacecraft pursuit-evasion game simulation combining nonlinear orbital dynamics (NERM), Extended Kalman Filter (EKF) state estimation, and SDRE optimal control. The pursuer tracks the evader using noisy range-azimuth-elevation measurements, estimates relative state via EKF, then computes optimal thrust commands via SDRE game-theoretic control.

Three orbital dynamics models for spacecraft relative motion in LVLH coordinates:

1. **CW (Clohessy-Wiltshire)**: Linear dynamics for near-circular orbits
2. **LERM (Linearized Elliptical Relative Motion)**: Time-varying linear dynamics for elliptical orbits, linearized around the reference orbit
3. **NERM (Nonlinear Elliptical Relative Motion)**: Full nonlinear dynamics for elliptical orbits using SDRE approach

## Architecture

### Coordinate System
All models use LVLH coordinates:
- x: radial direction (away from Earth center)
- y: along-track direction (orbital motion direction)  
- z: cross-track direction (orbit normal, right-hand system)

### Dynamics Models (`dynamics/` package)

**`cw.py`**: Circular orbit approximation
- State: `[x, y, z, vx, vy, vz]` (6D)
- Constant orbital rate `n`
- Returns state-space matrices `(A, B)` via `get_state_space()`

**`lerm.py`**: Elliptical orbit linearization
- State: `[x, y, z, vx, vy, vz]` (6D relative state)
- Time-varying parameters: `ν̇` (true anomaly rate), `ν̈` (true anomaly acceleration)
- Requires `OrbitalDynamics` instance for Keplerian parameter computation
- Key method: `get_A_matrix(nu_dot, nu_ddot)` builds time-varying system matrix
- Degenerates to CW when `ν̈=0` and `ν̇=n` (circular limit)

**`nerm.py`**: Full nonlinear 3D model
- State: `[X_p(6), X_e(6), nu(1)]` (13D: pursuer + evader + true anomaly)
- Implements `get_SDC_matrix()` for SDRE control (state-dependent coefficient matrix)
- ODE integration via `dynamics_13d(t, state, u_p, u_e)`
- Default parameters: μ=3.986e5 km³/s², a_c=15000 km, e_c=0.5

**`nerm_2d.py`**: In-plane (2D) simplification
- State: `[X_p(4), X_e(4), nu(1)]` (9D: x,y,vx,vy per spacecraft + true anomaly)
- z-direction decoupled and set to zero
- ODE integration via `dynamics_9d(t, state, u_p, u_e)`

### Control Strategies (`control/` package)

**`sdre.py`**: SDRE zero-sum differential game controller
- Solves Algebraic Riccati Equation (ARE) at each timestep
- Uses symplectic balancing for numerical stability when ||Q|| and ||S|| differ by orders of magnitude
- Computes optimal control for both pursuer and evader
- Key class: `SDREGameController` with `compute_control(A_SDC, x_rel)` method
- Supports sparse ARE updates via `solve_are=False` parameter

**`lqdg.py`**: Linear Quadratic Differential Game (LQDG) controller
- Solves Game Algebraic Riccati Equation (GARE) for time-invariant systems
- Separate controllers: `ChaserController` (pursuer) and `EvaderController`
- Requires M = Rp⁻¹ - Re⁻¹ to be positive definite (pursuer must have control advantage)
- Used with CW dynamics for constant-parameter scenarios

**`neural.py`**: Neural network PINN-based SDRE surrogate (3D)
- Approximates online ARE solving using Physics-Informed Neural Networks
- Uses Log-Cholesky parameterization for SPD matrix output
- Includes OOD (out-of-distribution) detection with fallback to classical ARE
- Key class: `NeuralSDREController` with same interface as `SDREGameController`
- Extracts 10 independent features from A_SDC matrix

**`neural_2d.py`**: 2D version of neural SDRE surrogate
- Same architecture as `neural.py` but for 2D in-plane dynamics

### Key Relationships

- LERM linearizes NERM around `r_d ≈ r_c` (deputy distance ≈ chief distance)
- Gravity term approximation: `μ/r_c² - μ(r_c+x)/r_d³ ≈ 2ν̇²·x` for small deviations
- Parameters `ν̇`, `ν̈`, `r_c` are computed from Keplerian mechanics via `get_orbital_params(nu)`
- All models use control inputs `u_p` (pursuer thrust) and `u_e` (evader thrust) in km/s²

### Main Simulation (`main.py`)

Implements complete NERM + EKF + SDRE closed-loop simulation:

**State Propagation**:
- 13D true state: `[X_p(6), X_e(6), nu(1)]` propagated via RK4 integration
- Time step: 5s, duration: 2 orbital periods (~10 hours for e=0.6 orbit)

**Measurement & Estimation**:
- Measurement model: range-azimuth-elevation `[ρ, az, el]` from relative position
- Noise: σ_r=10m, σ_az=σ_el=10⁻⁴ rad
- EKF: predicts relative state using SDC matrix linearization, updates with measurements
- Process noise: position 5×10⁻⁴ km², velocity 5×10⁻⁸ km²/s²

**Control**:
- SDRE controller solves ARE every 5 steps (25s interval) for computational efficiency
- Between ARE updates, reuses cached P matrix (sparse ARE updates)
- Thrust limits: pursuer 1.2×10⁻³ km/s², evader 2×10⁻⁴ km/s²
- Evader uses 35% of commanded thrust (asymmetric game)

**Output**:
- PNG with 4 subplots: 3D trajectories, relative distance, EKF error, x-y projection
- Typical result: 153 km → 3.6 km capture, final EKF error ~30m

## Development Commands

```bash
# Run main simulation (generates nerm_ekf_sdre_trajectory.png)
python main.py

# Python 3.12+ required
python3.12 main.py
```

## Important Notes

- **Type annotations**: All files using `tuple[...]` or `dict[...]` syntax MUST include `from __future__ import annotations` at the top (Python 3.9+ compatibility)
- State dimensions: CW/LERM use 6D relative state, NERM uses 13D absolute states, NERM-2D uses 9D
- Control inputs are differential thrust accelerations: `B @ (u_p - u_e)`
- True anomaly `nu` evolves according to `dnu/dt = ν̇(nu)` in closed-loop simulations
- EKF uses SDC matrix from estimated evader position for prediction step
- Angle wrapping required for azimuth/elevation innovation: `y[1:3] = wrap_angle(y[1:3])`
