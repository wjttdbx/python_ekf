# Methodology Blueprint: 3-D Angles-Only EKF-SDRE Closed-Loop Approach

> Pipeline stage: Stage 1 / Phase 1 (research scoping)  
> Target venue: *Acta Astronautica*  
> Repository snapshot inspected: commit `dae72c58c78bec67f154b0cc36209c0fb86a6412`, branch `master`, with an uncommitted working tree on 2026-07-11.  
> Boundary: this document designs the study. It does not review, revise, or inherit the argument of `docs/manuscript.tex`; the manuscript and reports were used only as pointers to code, data, and prior runs.

## 1. Research decision

### 1.1 Recommended paper identity

The paper should be rebuilt as a **quantitative computational control study of estimation-limited closed-loop approach quality**, not as a paper claiming that estimation error improves rendezvous and not as a docking study. The common SDC matrix is the proposed implementation architecture; the primary scientific question is whether its 3-D angles-only closed loop reaches and remains in a terminal approach set under matched dynamics, information, control authority, and stochastic conditions.

The surprising shorter time reported for angles-only runs is treated as a phenomenon to explain using pre-specified metrics and causal interventions. It is not treated as evidence of superior control.

### 1.2 Primary research question

Under 3-D nonlinear elliptical relative motion, how does a common-SDC angles-only EKF-SDRE closed loop compare with a full-state SDRE oracle in (i) first passage through a 100-m sphere, (ii) low-relative-speed arrival, and (iii) sustained terminal-set occupancy, when both use the same truth dynamics, target policy, controller weights, actuator model, initial condition, and evaluation horizon?

### 1.3 Secondary research questions

1. Which estimated-state channel causes the difference between first-passage time and soft/sustained approach quality: position, velocity, the SDC matrix supplied to the ARE, or initialization through `P0`?
2. Does angular-noise growth change total control effort, or mainly change the direction and timing of a comparable control impulse?
3. Is reuse of one SDC matrix by the EKF and SDRE merely computationally convenient, or does it change closed-loop performance relative to a separately linearized EKF under otherwise matched conditions?
4. Over which orbital, geometric, sensor, and target-maneuver regimes does the result remain valid?

### 1.4 Confirmatory hypotheses

- **H1, metric divergence:** angles-only EKF-SDRE can have a shorter 100-m first-passage time than the full-state oracle while having a higher boundary-crossing relative speed and a lower probability of sustained terminal-set occupancy.
- **H2, control-budget redistribution:** after evaluation on a common fixed horizon, increased angular noise changes cumulative Delta-V less than it changes first-passage time and terminal speed; any equivalence claim must pass a pre-declared equivalence margin rather than rely on a nonsignificant difference.
- **H3, 3-D velocity channel:** replacing only the estimated velocity supplied to the controller with truth attenuates the early-crossing effect more than replacing only position. This is a new 3-D hypothesis; the existing 2-D result is not confirmatory evidence.
- **H4, SDC-to-ARE channel:** replacing the controller's estimated-state SDC matrix with a per-step truth-state SDC matrix has a practically negligible effect in the reference MEO case. This must be tested by equivalence across multiple paired seeds; a single-seed near-equality is insufficient.
- **H5, terminal quality:** increasing angular noise reduces the probability of soft and sustained approach even when the distance-only first-passage success rate remains high.

## 2. Research paradigm and method

### Research paradigm

**Selected:** positivist, model-based computational experimentation.

**Ontology:** the simulated relative state, sensor observations, controls, and event times are objective quantities conditional on the specified dynamics and numerical implementation.

**Epistemology:** claims are warranted through controlled simulation, numerical verification, paired stochastic comparisons, causal channel interventions, and robustness analysis. Simulation evidence supports conclusions about the modeled system, not unmodeled flight hardware or docking operations.

### Method

**Type:** quantitative.

**Specific method:** controlled factorial simulation with paired Monte Carlo trials, survival/time-to-event analysis, equivalence testing, and mechanism-oriented closed-loop interventions.

**Methodological logic:**

1. Verify the truth model, estimator, controller, units, and numerical convergence.
2. Compare the proposed and oracle loops under paired conditions.
3. separate terminal event definitions before interpreting time-to-event results.
4. Intervene on one closed-loop information channel at a time in the 3-D model.
5. Test generalization only after the reference-case mechanism is established.

## 3. Operational system and comparison contract

### 3.1 Truth plant

- Use `aerospace/dynamics/nerm.py` as the 13-D truth model: chaser state (6), target state (6), and true anomaly (1).
- Preserve the repository units: km, km/s, km/s^2, and seconds.
- Propagate truth with the current RK45 tolerances (`rtol=1e-8`, `atol=1e-10`) initially, then verify convergence.
- The core reference case remains `a_c=15000 km`, `e_c=0.5`, `nu0=0`, `Xp0=[500,500,500,0.01,0.01,0.01]`, and `Xe0=0`, because it is the only case with substantial archived pilot evidence. It is a stress-test case, not a canonical docking case.

### 3.2 Information and target-policy contract

Every comparison must state separately what the chaser and target know.

- The proposed chaser receives noisy azimuth and elevation and uses the EKF estimate.
- The full-state oracle chaser receives the true relative state; it is an information upper bound, not an implementable sensor mode.
- In the current game controller, the target control uses `x_rel_e=x_true_rel` (`nerm_ekf_sdre.py:158-162`; `sdre.py:194-196`). Therefore the default experiment is an **asymmetric-information pursuit-evasion game**: angles-only chaser versus truth-informed evader.
- Use exactly the same target policy and target information in each paired arm. Add a nonmaneuvering-target sensitivity arm; do not mix it into the primary contrast.
- Do not describe the target as nonmaneuvering when `u_e` is generated by the game controller.

### 3.3 Controller and actuator contract

- Primary controller parameters: `Q_ctrl=I6`, `R_ctrl=1e13 I3`, `gamma=sqrt(2)`, ARE update every control step, `dt=10 s`.
- The current controller has **no explicit thrust saturation** (`sdre.py:189-196`). Consequently, existing `max ||u||` values are peak commanded acceleration, not a specified maximum-thrust constraint.
- The rebuilt study must choose one of two transparent scopes:
  - algorithmic scope: keep unconstrained commanded acceleration and avoid hardware/fuel claims; or
  - engineering scope: add and predeclare an acceleration magnitude limit and saturation law, then repeat all primary comparisons.
- Recommended for *Acta Astronautica*: retain the unconstrained result as an algorithmic reference and add at least two physically motivated `u_max` sensitivity levels. Values must be justified from a declared propulsion class before runs are started.
- `Delta-V = integral ||u_p|| dt` is an ideal impulse proxy. It is not fuel mass. Fuel claims require spacecraft mass, specific impulse, and the rocket equation.

### 3.4 Mandatory comparison arms

| ID | Chaser measurement/state | EKF prediction matrix | Controller ARE matrix | Purpose |
|---|---|---|---|---|
| O | True relative state | none | `A_SDC(x_true)` | Full-state SDRE oracle |
| P | Azimuth/elevation EKF | `A_SDC(xhat)` | same `A_SDC(xhat)` | Proposed common-SDC loop |
| RA | Range + azimuth/elevation EKF | `A_SDC(xhat)` | same `A_SDC(xhat)` | Measurement-information control |
| SJ | Azimuth/elevation EKF | independently evaluated nonlinear Jacobian `J_f(xhat)` | `A_SDC(xhat)` | Direct shared-versus-separate linearization comparator |

Arm SJ is required before claiming a performance benefit from the common-matrix architecture. The existing `dual_A` experiment compares pursuer and evader ARE matrices, not EKF-versus-controller model sharing, so it cannot answer this architectural question.

## 4. Terminal-event definitions

All event metrics must be computed offline from a **fixed-horizon trajectory**. The simulation must not terminate at the first distance crossing.

Let `rho(t)=||r_rel(t)||`, `v(t)=||v_rel(t)||`, `r_c=0.1 km`, and `v_c=1e-4 km/s` (0.1 m/s).

### 4.1 Level 1: distance first passage

`T_FP = inf{t: rho(t) <= r_c}`.

- Interpretation: first transit of the 100-m sphere only.
- Report relative speed, radial closing speed, and control direction at `T_FP`.
- Never label this event capture, rendezvous, or docking without a qualifier.

### 4.2 Level 2: instantaneous soft-arrival proxy

`T_soft = inf{t: rho(t) <= r_c and v(t) <= v_c}`.

- Interpretation: the first instant satisfying distance and speed thresholds.
- It does not establish that the trajectory remains nearby.
- The current threshold is a study convention and requires sensitivity checks at 0.05, 0.1, and 0.2 m/s.

### 4.3 Level 3: sustained terminal-set occupancy

Define the terminal set `S={(r,v): rho<=r_c, ||v||<=v_c}` and

`T_sustain = inf{t: (r(tau),v(tau)) in S for every tau in [t,t+tau_dwell]}`.

- Primary `tau_dwell = 0.05 T_orbit`; sensitivity: 600 s and `0.1 T_orbit`.
- Also report number of exits, total residence fraction in `S`, maximum range after first entry, and time to first exit.
- Reserve **stable rendezvous** for a result that combines sustained occupancy with local closed-loop stability/invariance analysis. If no such proof is supplied, call the metric “sustained soft approach.”

### 4.4 Docking boundary

No result from this model is docking performance because attitude, docking-port geometry, line-of-sight/keep-out constraints, contact dynamics, plume constraints, and terminal hardware are absent.

## 5. Outcome measures

### 5.1 Primary outcomes

1. Probability of sustained soft approach by the analysis horizon.
2. `T_sustain`, with unsuccessful trials right-censored.
3. Boundary speed `v(T_FP)` and radial speed `r_hat dot v_rel` at first passage.

### 5.2 Key secondary outcomes

- Distance first-passage probability and `T_FP`.
- Instantaneous soft-arrival probability and `T_soft`.
- Closest approach, exit probability, exit time, number of terminal-set crossings, and terminal-set residence fraction.
- Position and velocity RMSE, component-wise bias, normalized estimation error squared (NEES), normalized innovation squared (NIS), and empirical 3-sigma coverage.
- Fixed-horizon `Delta-V`, `integral ||u||^2 dt`, peak commanded acceleration, time above 75% of peak or at saturation, thrust-direction change, and cumulative Delta-V fractions at 10%, 25%, 50%, and 90% of the horizon.
- ARE residual, number of ARE fallbacks, condition number diagnostics, wall time, and per-step controller time.

### 5.3 Comparison horizons

- Run the reference study for `H_total=10 T_orbit` without early stopping.
- Detect new terminal events only up to `H_event=9.9 T_orbit` when the primary dwell is `0.1 T_orbit`; retain the final interval for follow-up.
- Compare Delta-V and quadratic costs on the same fixed horizon. Event-truncated effort may be reported separately but cannot support an “equal fuel” statement across unequal event times.

## 6. Experimental matrix

### E0. Numerical and implementation verification

| Test | Levels | Acceptance criterion |
|---|---|---|
| Time-step convergence | `dt={10,5,2} s` on at least 10 paired cases | primary metrics change below predeclared tolerances (recommended: event time <1%, Delta-V <1%, boundary speed <5%) |
| Integrator tolerance | baseline and 10x tighter | no material change in conclusions |
| SDC reconstruction | random valid states | `||A(x)x-f(x)||` near machine precision with unit-aware tolerance |
| ARE solution | every solved step | finite symmetric `P`, residual below tolerance, fallback recorded rather than hidden |
| EKF consistency | noise-only calibration cases | NIS/NEES coverage reported; not tuned on test cases |
| Event detection | analytic synthetic trajectories | interpolated crossing and dwell logic passes unit tests |

Use interpolation or event localization between 10-s samples; do not report grid-quantized event times as exact.

### E1. Paired reference-case comparison

- Arms: O, P, RA, SJ.
- Angular noise: baseline `0.008 deg`.
- Initial state and target policy: identical across arms.
- Use at least 50 paired stochastic seeds for P/RA/SJ; O is deterministic for a fixed scenario but should be repeated over the same scenario set when initial conditions vary.
- Primary inference: P versus O. RA and SJ are secondary, multiplicity-controlled contrasts.

### E2. Sensor-noise response

- `sigma_theta={0.001,0.004,0.008,0.02,0.05,0.1} deg`.
- Use the same underlying standard-normal angular-error sequence for every noise level and arm, scaled by `sigma_theta` (common random numbers).
- Minimum 30 paired seeds per level for an initial pass; expand to 100 if the predeclared CI precision target is not met.
- Separate sensor noise from initialization: hold `xhat0` error and `P0` fixed in the confirmatory sweep. Analyze scaled `P0` only in E3.

### E3. Required 3-D causal ablations

All interventions leave the EKF running. They replace only the stated controller input channel and are interpreted as total closed-loop intervention effects, not additive variance decomposition.

| ID | Intervention | Held fixed | Identifies |
|---|---|---|---|
| A0 | Baseline P | none | reference phenomenon |
| A1 | Controller state position replaced by truth; velocity remains estimated | controller `A=A_hat` | direct position-feedback channel |
| A2 | Controller state velocity replaced by truth; position remains estimated | controller `A=A_hat` | direct velocity-feedback/braking channel |
| A3 | Entire controller state replaced by truth | controller `A=A_hat` | total direct state-feedback channel |
| A4 | Controller ARE uses per-step `A_true`; feedback state remains estimated | EKF prediction stays `A_hat` | `A -> P -> u` channel |
| A5 | Truth controller state + `A_true` | target policy and plant | reconstructs oracle control channel |
| A6 | Only radial velocity replaced by truth | `A=A_hat` | radial closing-speed contribution |
| A7 | Only along-track velocity replaced by truth | `A=A_hat` | likely braking/timing contribution |
| A8 | Only cross-track velocity replaced by truth | `A=A_hat` | 3-D out-of-plane contribution |

Required initialization factorial:

- Measurement noise `sigma_theta`: low, baseline, high (`0.001`, `0.008`, `0.1 deg`).
- `P0`: fixed reference, sensor-scaled current rule, and deliberately diffuse.
- Initial estimate error: exact truth versus sampled from a fixed covariance independent of `sigma_theta`.

This factorial separates measurement quality, declared covariance, and realized initial state error. The current setup often changes more than one of them together.

For A0-A5, use at least 30 paired seeds initially and expand to 100 for equivalence claims. A6-A8 are confirmatory only if H3 survives A1-A2; otherwise report them as exploratory.

### E4. Robustness and operating-envelope matrix

Use a stratified Latin hypercube rather than three isolated orbit labels alone.

- Orbit: valid combinations of semi-major axis and eccentricity satisfying a declared minimum perigee altitude.
- Initial true anomaly: cover the orbit, not only `nu0=0`.
- Initial separation: canonical proximity range (1-100 km) and long-range stress range (100-1000 km) analyzed separately.
- LOS geometry: radial, along-track, cross-track, and mixed directions.
- Initial relative velocity: bounded signed components with mission-based ranges.
- Target policy: game-theoretic evader (primary) and nonmaneuvering target (sensitivity).
- Control authority: unconstrained reference plus justified saturation levels.

Use at least 200 paired scenario draws for the proposed-oracle comparison. Do not pool canonical and stress-test ranges into one headline success rate.

### E5. Failure-focused Monte Carlo

- After E4, oversample near the observed failure boundary using a new, predeclared validation set.
- Classify failure modes using objective rules: estimator divergence, ARE failure/fallback excess, actuator saturation, missed first passage, soft-arrival failure, and post-entry escape.
- A classifier or response surface may be exploratory, but all headline rates come from the untouched validation set.

## 7. Data strategy and provenance

### Data type

Primary synthetic data generated by the repository's 3-D simulation. Existing CSV files are secondary pilot data until provenance is repaired.

### Required run bundle

Each experiment writes to an immutable directory such as `outputs/study/<run_id>/` containing:

- `config.json`: every physical, estimator, controller, event, and numerical parameter;
- `environment.json`: git commit, dirty patch hash, command, UTC time, OS, CPU, Python, `uv`, NumPy, SciPy, and CasADi versions;
- `trial_manifest.csv`: scenario ID, arm, seed IDs, random-stream IDs, status, runtime, and failure reason;
- `metrics.csv`: one row per trial with all predeclared metrics and units in column names;
- compressed time series for audit/replotting, including truth, estimate, covariance, innovation, controls, distances, relative speed, and ARE diagnostics;
- `sha256.txt`: hashes of configs, metrics, time series, scripts, and figures;
- `README.md`: exact regeneration and analysis commands.

### Randomness contract

- Split random streams for initial-state sampling, initial-estimate error, measurement noise, and any target disturbance.
- Pair arms with the same scenario and compatible noise draws.
- Never use one mutable RNG sequentially across arms, because arm duration or branching would change later samples.
- Store seed integers and generator type.

### Code-to-data traceability

- Publication data must be regenerated from a clean tagged commit after the event-metric and fixed-horizon changes.
- Driver scripts must be committed. A CSV without its executable driver and config is not publication-grade provenance.
- Figures and tables must be generated only from archived `metrics.csv`/time series, never from manually transcribed manuscript values.

## 8. Statistical analysis plan

### 8.1 General principles

- Unit of analysis: one paired scenario-seed realization, not one time sample and not one aggregate noise level.
- Report effect estimates and 95% confidence intervals before p-values.
- Predeclare primary contrast (P versus O), primary outcomes, equivalence margins, and multiplicity correction.
- Failed event trials remain in the analysis as censored observations; do not compute time statistics only among successes without an explicit conditional label.

### 8.2 Binary outcomes

- Report success proportion with Wilson 95% CI for each arm.
- Report paired risk difference and paired bootstrap CI; use McNemar's test as a secondary test for paired arms.
- For noise trends, use a seed-clustered logistic model or cluster bootstrap, not a correlation computed on six aggregated means.

### 8.3 Time-to-event outcomes

- Plot Kaplan-Meier curves for `T_FP`, `T_soft`, and `T_sustain` separately.
- Report restricted mean time to event (RMTE/RMST complement as appropriate) up to the common horizon, median only when estimable, and paired bootstrap differences.
- Use scenario/seed-clustered inference for repeated noise levels.

### 8.4 Continuous outcomes

- For fixed-horizon Delta-V, boundary speed, RMSE, and residence fraction, report median, IQR, paired median difference or log-ratio, and bootstrap 95% CI.
- Also report mean and SD when finite and scientifically interpretable; retain heavy-tail diagnostics and do not hide outliers.
- Model response to `log10(sigma_theta)` with a seed random intercept or paired cluster bootstrap. Nonlinear/monotone trends should be shown rather than forced into a linear slope.

### 8.5 Equivalence and noninferiority

- “Same Delta-V,” “same peak,” and “negligible A-channel effect” are equivalence claims.
- Recommended provisional margins, to be finalized before reruns: fixed-horizon Delta-V ratio within +/-2%, event-time ratio within +/-5%, and boundary-speed difference within +/-0.02 m/s where applicable.
- Use TOST or the equivalent CI inclusion rule on paired log-ratios/differences. Failure to reject a difference is not equivalence.

### 8.6 Multiplicity and precision

- Apply Holm correction to confirmatory secondary contrasts/outcomes.
- Treat component-wise A6-A8 and broad E4 interactions as exploratory with false-discovery-rate control.
- Use a sequential **precision** rule rather than post-hoc power: start at 30 or 50 paired seeds, then expand to 100 if the 95% CI half-width exceeds 5 percentage points for success probability or the predeclared practical margin for a continuous primary effect. Freeze the stopping rule before inspecting confirmatory results.

## 9. Validity and reliability criteria

| Criterion | Design strategy | Pass condition |
|---|---|---|
| Internal validity | paired scenario/noise streams; one-channel interventions; matched target and actuator policy | no unintended parameter differences in config diff |
| Construct validity | three terminal definitions; fixed-horizon effort; no docking/fuel relabeling | each claim maps to its exact metric and unit |
| Numerical validity | step/tolerance convergence, event interpolation, SDC identity, ARE residuals | E0 thresholds passed before inferential runs |
| Estimator reliability | NIS, NEES, empirical coverage, divergence rule | calibration reported, failures retained |
| Statistical reliability | adequate paired seeds, censoring-aware analysis, bootstrap CIs | predeclared precision target met |
| Reproducibility | clean commit, committed drivers, immutable configs, hashes, exact commands | independent rerun recreates metrics within tolerance |
| External validity | stratified orbit/geometry/range/target-policy matrix | claims restricted to tested strata |
| Objectivity | hypotheses and margins fixed before confirmatory rerun | deviations logged and labeled exploratory |

## 10. Existing evidence: reuse versus rerun ledger

### 10.1 Reusable as implementation basis or pilot evidence

| Artifact | Reuse decision | Permitted use |
|---|---|---|
| `aerospace/dynamics/nerm.py` | reuse after E0 | truth-model implementation |
| `aerospace/estimation/ekf.py` | reuse after consistency tests | angles/range/Doppler measurement and EKF implementation |
| `aerospace/control/sdre.py` | reuse after ARE and unit tests | baseline SDRE game controller; document absence of saturation |
| `aerospace/simulation/nerm_ekf_sdre.py` | refactor, then reuse | common closed-loop engine; remove early stop and add diagnostics/interventions |
| `data/noise_sweep_results.csv` (30 rows) | pilot only | choose noise levels and estimate runtime/variance |
| `data/terminal_vrel_results.csv` | pilot only | motivate boundary-speed outcome |
| `data/monte_carlo_results.csv` (200 rows) | pilot only | identify heavy tails and candidate failure strata |
| `outputs/data/dual_a_matrix/*` | pilot only | motivate multi-seed A-channel equivalence test |
| `REPORT_angle_only.md` 2-D interventions | hypothesis generation only | specify H3 and 3-D channel ablations; never cite as 3-D confirmation |

### 10.2 Must be rerun for publication claims

| Existing result family | Why rerun is mandatory |
|---|---|
| Baseline angles-only versus “ideal SDRE” | current engines stop at `rho<0.1 km`; soft/sustained behavior and fixed-horizon effort are unavailable |
| Noise sweep | the current repository does not contain the driver that generated `data/noise_sweep_results.csv`; initialization and noise effects are confounded; only five seeds per level |
| Tight-capture Ablation C | `experiment_ablation_c_tight_capture.py` calls an engine that stops at the loose 100-m crossing, then post-filters the truncated trajectory. A high-noise “tight failure” therefore means only that the first distance crossing was not soft, not that soft arrival never occurred later in the nominal 10-orbit horizon |
| Delta-V/fuel comparisons | archived Delta-V is integrated to different early-stop times; compare again on a common horizon, and call it Delta-V rather than fuel |
| 3-D position/velocity/A/P0 mechanism | current detailed causal ranking is 2-D; no multi-seed 3-D channel intervention exists |
| Monte Carlo 97.5% success | the current repository lacks the generating driver for `data/monte_carlo_results.csv`; success is distance-only and variable-horizon |
| Eccentricity/CW sweep | generating driver referenced by older documents is absent; all cases use distance-only stopping, and model/initial-condition matching must be reverified |
| LEO/MEO/GEO sweep | generating driver is absent; five seeds per cell are inadequate for stochastic GEO behavior; failure mechanisms require objective diagnostics |
| `dual_A`, observation-model, and game-theory comparisons | single-seed, early-stop pilot scripts; some change controller/game assumptions while changing sensor mode, so they are not clean primary comparisons |
| Prediction-discretization accuracy | rerun with committed driver, long-horizon accumulation check, and current code; a local FE-versus-RK result alone cannot establish closed-loop equivalence |

### 10.3 Provenance problem to resolve before Stage 2 evidence synthesis

Older project documents name `aerospace/experiments/sensor_noise_sweep.py`, `monte_carlo.py`, `eccentricity_sweep.py`, `altitude_sweep.py`, and `prediction_error_diagnostic.py`, but those driver files are not present in the live checkout. The CSVs remain useful as historical pilot evidence, but their numerical claims are not reproducible from the current repository. The rebuild must restore or replace these scripts and regenerate the full dataset from a tagged clean state.

## 11. Required implementation changes before confirmatory runs

1. Add a fixed-horizon mode to `EKFSDRESimulation` and `SDRESimulation`; event detection records events but never terminates truth propagation.
2. Add offline, tested event extraction for `T_FP`, `T_soft`, dwell, exits, and residence time.
3. Log relative velocity, radial speed, cumulative Delta-V, controller state, `A_hat`, optional `A_true`, `P`, ARE residual/fallback, NIS, and NEES.
4. Implement intervention hooks that keep `A_hat` fixed while replacing position/velocity feedback channels; otherwise direct-state and A-matrix effects are confounded.
5. Replace the current static `A_fixed` interface with a per-step matrix provider or explicit controller-matrix policy.
6. Implement the SJ comparator with a documented nonlinear Jacobian and validated mean/covariance propagation.
7. Add deterministic random-stream management and immutable run bundles.
8. Add optional vector-magnitude thrust saturation, with saturation status logged.
9. Restore all experiment drivers and add a single top-level command that produces raw data, analysis tables, and figures without manual edits.

## 12. Boundaries and limitations by design

- Results are conditional on the chosen SDC factorization; `A(x)x=f(x)` does not make the factorization unique or the SDRE law globally optimal.
- Full-state SDRE is an oracle reference, not a shortest-time or globally optimal benchmark. The current quadratic SDRE cost has no explicit minimum-time or terminal-speed constraint.
- The controller is a game-theoretic local feedback construction, not proof of nonlinear global optimality.
- Pure angles-only range observability depends on geometry and excitation. Exact initial truth, sampled initial error, and covariance initialization must be analyzed separately.
- A 100-m sphere at an 866-km initial separation is an approach stress test. It is not a complete servicing or docking mission.
- Unmodeled effects include J2 and higher gravity, drag, solar radiation pressure, sensor field of view/occlusion, attitude coupling, thruster quantization, navigation latency, mass depletion, plume constraints, keep-out zones, and contact dynamics.
- Hardware feasibility cannot be inferred from Python wall time alone; real-time claims require target-processor timing and deterministic worst-case analysis.

## 13. Ethics, reporting, and preregistration

### Ethical considerations and IRB

No human participants, personal data, animals, or clinical data are involved. IRB review and informed consent are not applicable. Research-integrity requirements still apply: complete provenance, retention of failures/outliers, disclosure of post-hoc analyses, and no selective reporting.

### Reporting standard

No EQUATOR guideline is directly applicable to an aerospace computational-control study. Follow the *Acta Astronautica* author instructions plus a simulation verification/validation and computational reproducibility checklist: model equations and assumptions, numerical convergence, uncertainty design, estimator consistency, controller/actuator limits, complete outcome definitions, code/data provenance, and failure reporting.

### Preregistration

- **Recommended:** yes, for H1-H5 and the primary E1-E3 analyses.
- **Platform:** OSF Registries.
- **Status:** planned before confirmatory reruns.
- Register arms, seeds/sample-size precision rule, thresholds, horizons, outcomes, exclusion/failure rules, equivalence margins, and multiplicity handling. Mark E4 response-surface work and any mechanism discovered after data inspection as exploratory.

## 14. Stage 1 exit criteria

The methodology is ready to hand to downstream research and experiment-design phases when the following choices are approved:

1. Paper identity: estimation-limited sustained approach, with common SDC as architecture rather than an assumed performance advantage.
2. Primary contrast: P versus O under an explicitly truth-informed or nonmaneuvering target policy.
3. Primary endpoint: sustained soft approach, with first passage retained as a distinct secondary endpoint.
4. Fixed-horizon, no-early-stop evaluation.
5. Whether the main engineering study includes explicit thrust saturation and which propulsion class defines `u_max`.
6. The provisional practical-equivalence margins and dwell-time definition.
7. Commitment to regenerate publication results from committed drivers and a clean tagged code state.

Downstream work should next produce a research-question brief aligned to this blueprint, recover or replace missing experiment drivers, and create a preregistered run specification. It should not draft the new manuscript until E0 verification and the confirmatory E1-E3 dataset are complete.
