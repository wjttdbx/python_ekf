# AGENTS.md

Agent instructions for this repository.

## Scope

- Keep changes focused on the user request.
- Do not start unrelated refactors.
- Prefer minimal diffs and preserve existing style.

## Project At A Glance

- Domain: spacecraft pursuit-evasion dynamics in LVLH coordinates.
- Core dynamics modules are in [dynamics/](dynamics/).
- Controllers are in [control/](control/).
- Authoritative architecture notes are in [CLAUDE.md](CLAUDE.md).

## Run Commands

- Use Python 3.12+ (see [pyproject.toml](pyproject.toml)).
- Run entry point:
  - `python main.py`
  - `python3.12 main.py`

## High-Value File Map

- Dynamics models:
  - [dynamics/cw.py](dynamics/cw.py): CW linear near-circular model (6D relative state).
  - [dynamics/lerm.py](dynamics/lerm.py): linearized elliptical relative motion (6D, time-varying).
  - [dynamics/nerm.py](dynamics/nerm.py): nonlinear elliptical model (13D closed-loop state with true anomaly).
  - [dynamics/nerm_2d.py](dynamics/nerm_2d.py): 2D in-plane simplification (9D closed-loop state).
- Control methods:
  - [control/lqdg.py](control/lqdg.py)
  - [control/sdre.py](control/sdre.py)
  - [control/sdre_2d.py](control/sdre_2d.py)
  - [control/neural.py](control/neural.py)
  - [control/neural_2d.py](control/neural_2d.py)

## Known Pitfalls (Check Before Debugging)

- Import path mismatch is documented in [CLAUDE.md](CLAUDE.md):
  - [dynamics/lerm.py](dynamics/lerm.py)
  - [control/neural.py](control/neural.py)
  - [control/neural_2d.py](control/neural_2d.py)
- [README.md](README.md) is currently empty; prefer [CLAUDE.md](CLAUDE.md) for technical context.
- [main.py](main.py) is a placeholder entry script.

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
  - corresponding dynamics file in [dynamics/](dynamics/)
  - related controller file in [control/](control/)
  - architecture notes in [CLAUDE.md](CLAUDE.md)
- Prefer adding short runnable examples/tests when behavior changes materially.
