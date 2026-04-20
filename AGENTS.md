# AGENTS.md

Agent instructions for this repository.

## Scope

- Keep changes focused on the user request.
- Do not start unrelated refactors.
- Prefer minimal diffs and preserve existing style.

## Project At A Glance

- Domain: spacecraft pursuit-evasion dynamics in LVLH coordinates.
- Core dynamics modules are in [aerospace/dynamics/](aerospace/dynamics/).
- Controllers are in [aerospace/control/](aerospace/control/).
- Authoritative architecture notes are in [CLAUDE.md](CLAUDE.md).

## Run Commands

- Use Python 3.12+ (see [pyproject.toml](pyproject.toml)).
- Run main entry points:
  - `python main.py`
  - `python3.12 main.py`
  - `python main_nerm_sdre.py`

## High-Value File Map

- Dynamics models:
  - [aerospace/dynamics/cw.py](aerospace/dynamics/cw.py): CW linear near-circular model (6D relative state).
  - [aerospace/dynamics/lerm.py](aerospace/dynamics/lerm.py): linearized elliptical relative motion (6D, time-varying).
  - [aerospace/dynamics/nerm.py](aerospace/dynamics/nerm.py): nonlinear elliptical model (13D closed-loop state with true anomaly).
  - [aerospace/dynamics/nerm_2d.py](aerospace/dynamics/nerm_2d.py): 2D in-plane simplification (9D closed-loop state).
- Control methods:
  - [aerospace/control/lqdg.py](aerospace/control/lqdg.py)
  - [aerospace/control/sdre.py](aerospace/control/sdre.py)
  - [aerospace/control/sdre_2d.py](aerospace/control/sdre_2d.py)
  - [aerospace/control/neural.py](aerospace/control/neural.py)
  - [aerospace/control/neural_2d.py](aerospace/control/neural_2d.py)
- Reusable simulation engines:
  - [aerospace/simulation/nerm_ekf_sdre.py](aerospace/simulation/nerm_ekf_sdre.py)
  - [aerospace/simulation/nerm_sdre.py](aerospace/simulation/nerm_sdre.py)
- Experiment and training entry examples:
  - [aerospace/experiments/control_comparison_3d.py](aerospace/experiments/control_comparison_3d.py)
  - [aerospace/experiments/train_pinn_lc1.py](aerospace/experiments/train_pinn_lc1.py)
  - [aerospace/pinn/pinn_trainer.py](aerospace/pinn/pinn_trainer.py)

## Known Pitfalls (Check Before Debugging)

- Import path mismatch is documented in [CLAUDE.md](CLAUDE.md):
  - [aerospace/dynamics/lerm.py](aerospace/dynamics/lerm.py)
  - [aerospace/control/neural.py](aerospace/control/neural.py)
  - [aerospace/control/neural_2d.py](aerospace/control/neural_2d.py)
- [README.md](README.md) is currently empty; prefer [CLAUDE.md](CLAUDE.md) for technical context.
- [main.py](main.py) is an active entry script and imports the simulation engine from [aerospace/simulation/nerm_ekf_sdre.py](aerospace/simulation/nerm_ekf_sdre.py).

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
