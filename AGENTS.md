# AGENTS.md

## Archived development repository

Since 2026-09-12, active development has moved to `../pinn-ekf`.
The original study is under `../pinn-ekf/studies/python_ekf/` and shares the
main repository's `aerospace` package. This directory is a historical backup.
For new implementation, experiment, or manuscript tasks, use the main
repository unless the user explicitly requests work on this historical copy.
Preserve the original uncommitted J2 study and its results.

Agent instructions for this repository.

## Task scope and completion

Complete requested edits, affected checks, and fixes within the authorized scope.
Resolve routine implementation details without repeated approval; ask only when
missing information changes the result or authority to act. Review-only and
draft-only requests remain read-only for the source artifacts.

Read the files and references needed for the current change, not the whole
project. Run affected tests for behavior changes; documentation-only edits need
a content/diff check, not training or simulation. Broaden testing only for shared
behavior changes, failures, or a concrete unresolved concern. Passing the
necessary checks is the stopping point.

Keep frozen experiment seeds, thresholds, metrics, and stop conditions intact.
Formal research acceptance follows its agreed protocol; a smoke run or an
implemented function does not establish a paper claim. Preserve negative results
and ongoing runs. Hashes belong at major freezes or provenance milestones, not
every edit or run.

Keep changes focused, avoid unrelated refactors, and preserve existing style.

## Project At A Glance

- Domain: spacecraft pursuit-evasion dynamics in LVLH coordinates.
- Core dynamics modules are in [aerospace/dynamics/](aerospace/dynamics/).
- Controllers are in [aerospace/control/](aerospace/control/).
- Use [CLAUDE.md](CLAUDE.md) for architecture or experiment details needed by the task; it is not required reading for every edit.

## Run Commands

- Use Python 3.12+ with `uv run python` (see [pyproject.toml](pyproject.toml)).
- Run main entry points:
  - `uv run python main.py`
  - `uv run python main_nerm_sdre.py`
  - `uv run python run_scenarios.py`
- See [CLAUDE.md](CLAUDE.md) for the full list of experiment commands.

## High-Value File Map

- Dynamics models:
  - [aerospace/dynamics/cw.py](aerospace/dynamics/cw.py): CW linear near-circular model (6D relative state).
  - [aerospace/dynamics/lerm.py](aerospace/dynamics/lerm.py): linearized elliptical relative motion (6D, time-varying).
  - [aerospace/dynamics/nerm.py](aerospace/dynamics/nerm.py): nonlinear elliptical model (13D closed-loop state with true anomaly).
  - [aerospace/dynamics/nerm_2d.py](aerospace/dynamics/nerm_2d.py): 2D in-plane simplification (9D closed-loop state).
- Control methods:
  - [aerospace/control/sdre.py](aerospace/control/sdre.py): 3D SDRE game-theoretic controller (primary).
  - [aerospace/control/sdre_2d.py](aerospace/control/sdre_2d.py): 2D SDRE variant.
  - [aerospace/control/tpbvp_collocation.py](aerospace/control/tpbvp_collocation.py): CasADi+IPOPT TPBVP solver (ground-truth optimal).
  - [aerospace/control/optimal_control_solver.py](aerospace/control/optimal_control_solver.py): CasADi 2D time-energy optimal control.
  - [aerospace/control/inverse_optimal_fit.py](aerospace/control/inverse_optimal_fit.py): recover R(x) from optimal trajectories.
  - [aerospace/control/lqdg.py](aerospace/control/lqdg.py): LQDG for CW systems.
  - [aerospace/control/neural.py](aerospace/control/neural.py): neural SDRE surrogate (note: imports from removed pinn module).
- State estimation:
  - [aerospace/estimation/ekf.py](aerospace/estimation/ekf.py): 3D EKF with angles-only support.
  - [aerospace/estimation/ekf_2d.py](aerospace/estimation/ekf_2d.py): 2D in-plane EKF.
- Reusable simulation engines:
  - [aerospace/simulation/nerm_ekf_sdre.py](aerospace/simulation/nerm_ekf_sdre.py): main EKF+SDRE closed-loop engine.
  - [aerospace/simulation/nerm_sdre.py](aerospace/simulation/nerm_sdre.py): ideal state-feedback engine.
- Key experiment files:
  - [aerospace/experiments/monte_carlo.py](aerospace/experiments/monte_carlo.py)
  - [aerospace/experiments/compare_tpbvp_vs_sdre.py](aerospace/experiments/compare_tpbvp_vs_sdre.py)
  - [aerospace/experiments/altitude_sweep.py](aerospace/experiments/altitude_sweep.py)
  - [aerospace/experiments/eccentricity_sweep.py](aerospace/experiments/eccentricity_sweep.py)

## Known Pitfalls (Consult When Relevant)

- If [README.md](README.md) lacks the needed technical context, consult the relevant section of [CLAUDE.md](CLAUDE.md).
- [main.py](main.py) is the active entry script and imports the simulation engine from [aerospace/simulation/nerm_ekf_sdre.py](aerospace/simulation/nerm_ekf_sdre.py).
- `aerospace/control/neural.py` and `neural_2d.py` import from `aerospace.pinn.*` which no longer exists — they are effectively dead code.
- Use `uv run python` not bare `python` — the system Python may be 3.8.

## Coding Conventions In This Repo

- Respect model dimensionality boundaries:
  - CW/LERM: 6D relative state.
  - NERM: 13D state (`Xp(6), Xe(6), nu(1)`).
  - NERM-2D: 9D state (`Xp(4), Xe(4), nu(1)`).
- Preserve units used in current code and docs:
  - Distances in km, accelerations in km/s^2, Earth gravitational parameter in km^3/s^2.
- Keep LVLH axis interpretation consistent with [CLAUDE.md](CLAUDE.md).

## When Making Changes

- If updating equations or state definitions, verify consistency across:
  - corresponding dynamics file in [aerospace/dynamics/](aerospace/dynamics/)
  - related controller file in [aerospace/control/](aerospace/control/)
  - architecture notes in [CLAUDE.md](CLAUDE.md)
- Prefer adding short runnable examples/tests when behavior changes materially.
