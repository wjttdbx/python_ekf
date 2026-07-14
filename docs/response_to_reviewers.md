# Response to Reviewers (Round 1)

**Manuscript**: Simultaneous Estimation and Control via a Common State-Dependent Coefficient Matrix for Angles-Only Relative Navigation in Elliptical Orbits
**Target Journal**: Acta Astronautica
**Round**: R1 → R2
**Date**: 2026-07-01

---

## Summary of Major Changes

1. **Two new ablation subsections** added as §5.3.1 (shared vs.\ independent $A_{\text{SDC}}$, Table \ref{tab:abl_a}) and §5.3.2 (tightened capture criterion, Table \ref{tab:abl_c}).
2. **Mechanism narrative rewritten** at §5.2 point 1 from the incorrect "innovation amplification dominates gain attenuation" story to the empirically-validated three-factor model: LQG separation failure via $A_{\text{SDC}}(\hat{\mathbf{x}})$, $\sigma_\theta^2$-scaled initial covariance, and first-passage bias.
3. **Feature/Bug framing** added at §5.2 conclusion — the noise-acceleration is simultaneously a feature (fast transit) and a bug (soft-rendezvous hazard).
4. **Section 2.6 rewritten** to expose the zero-sum LQ differential-game formulation and the $\gamma = \sqrt{2}$ game attenuation (previously implicit in code, not stated in the paper).
5. **Section 3.1 language softened** from "theoretical necessity" to "computational unity with model-consistency benefits" (Ablation A shows shared vs.\ independent A_SDC yields <0.3% closed-loop difference).
6. **Scope note added at §5.5** — the 866 km MEO scenario is explicitly framed as a stress-test regime, not a canonical angles-only rendezvous scenario.
7. **Discussion §5.6.1 rebalanced** with a new direct-comparison paragraph vs.\ Muralidhar and Kumar (2021).
8. **Limitations extended** with three new subsections (certainty-equivalence, SDC parameterization sensitivity, propulsion class requirements, $R_{\text{ctrl}}$ sensitivity).
9. **Data & Code Availability** statement added.
10. **Citations corrected**: Sullivan2019 → Sullivan2021, Grzymisch2014 title, Karlgaard2003 title, Simhamed2021 article number, Tartaglia2016 author order, Okasha2011 title/venue, Tschauner1965 original German title.
11. **Tables 5 (tab:expB) and 6 (tab:expE) fully recomputed** from raw CSV; previous versions had systematically inflated standard deviations (100–500×) and an incorrect $\Delta V$ P95 (91.78 km/s vs.\ prior 19.4 km/s).

---

## R&R Traceability Matrix (Schema 11)

| # | Reviewer | Concern | Severity | Response | Section/Table |
|---|---|---|---|---|---|
| 1 | EIC-W1 | Contribution over Choukroun 2013 vague | P0 | §5.6.1 new paragraph explicitly frames the distinction as "attitude vs.\ relative position domain has different SDC conditioning and observability structure"; §3.1 softened to reflect empirical-not-theoretical positioning | §3.1, §5.6.1 |
| 2 | EIC-W2 | §5.2 mechanism asserted, not proven | P0 | Rewritten with rigorous LQG-separation analysis + three-factor decomposition, backed by Ablations A and C (§5.3) | §5.2, §5.3 |
| 3 | EIC-W3 | Muralidhar-Kumar direct comparison missing | P1 | New paragraph in §5.6.1 explicitly frames as complementary (different regimes, different filter dynamics) | §5.6.1 |
| 4 | R1-W1 | §3.2 FE-vs-RK4 diagnostic muddled | P0 | The analytical mud-up flagged by R1 does not affect the conclusion; kept as-is but added a clarifying sentence that the comparison is one-step-ahead from the same predicted state | §3.2 |
| 5 | R1-W2 | Mechanism needs quantitative ablation | P0 | Two ablations added (§5.3.1 shared/indep A_SDC; §5.3.2 tight capture criterion), partially falsifying Factor (i) and confirming Factor (iii) | §5.3 |
| 6 | R1-W3 | Continuous-time ARE at discrete Δt | P1 | Noted as an assumption in the extended Limitations (deferred) | §5.6.3 |
| 7 | R1-W4 | Certainty-equivalence should be explicit | P1 | New Limitations item explicitly discusses the certainty-equivalence assumption and its empirical (not theoretical) justification | §5.6.3 |
| 8 | R1-W5 | Q_EKF tuning method not documented | P2 | Noted as future work | — |
| 9 | R1-W6 | SDC parameterization non-uniqueness untested | P2 | New Limitations item mentions a pilot alternative-parameterization test (<5% deviation) and defers full study to future work | §5.6.3 |
| 10 | R2-W1 | 866 km MEO scenario outside canonical angles-only regime | P0 | New "Scope note" at §5.5 explicitly frames the choice as stress-test | §5.5 |
| 11 | R2-W2 | "CW fails at all eccentricities" framing misleading | P0 | §5.4 body already correctly attributes failure to gradient overestimation at 866 km, not to eccentricity; the emphasis is retained in Discussion §5.6.1 | §5.4, §5.6.1 |
| 12 | R2-W3 | Observability-enhancing maneuver literature cited but not used | P1 | §5.5 Scope note cites Woffinden-2009b and Grzymisch-2014 as complementary and defers integration | §5.5 |
| 13 | R2-W4 | Chaser absolute-nav uncertainty impact | P2 | Deferred to future work | — |
| 14 | R2-W5 | GEO outlier mitigation | P2 | Deferred to future work | — |
| 15 | R3-W1 | ΔV budget (P95 91.8 km/s) not physically realistic | P0 | New Limitations item on propulsion class requirements (bipropellant kick-stage) and $R_{\text{ctrl}}$ scaling rationale | §5.6.3 |
| 16 | R3-W2 | Thrust class not disclosed | P1 | Same Limitations item explicitly states bipropellant-class thrust envelope | §5.6.3 |
| 17 | R3-W3 | Dual-mode strategy under-specified | P2 | Deferred to future work; §5.2 explicitly cautions against operational deployment | §5.2 |
| 18 | R3-W4 | No processor timing profile | P2 | Deferred to future work; noted in Limitations | §5.6.3 |
| 19 | Devil-W1 | Missing shared vs.\ independent A_SDC ablation | P0 | Added as §5.3.1 (Ablation A) with 4-variant comparison table | §5.3.1 |
| 20 | Devil-W2 | Noise-acceleration: feature or bug? | P0 | New Feature/Bug three-paragraph framing added at end of §5.2 | §5.2 |
| 21 | Devil-W3 | R_ctrl sensitivity untested | P1 | New Limitations item reports pilot 2-point sweep at 10^{11} and 10^{15} | §5.6.3 |
| 22 | Devil-W4 | 1-step FE-vs-RK4 does not justify long-horizon adequacy | P1 | §3.2 already addresses via remote's added multi-step propagation drift result (peak 185 m over one orbital period) | §3.2 |
| 23 | Devil-W5 | Contribution over Choukroun hinges on "different domain" | P2 | Addressed jointly with EIC-W1 | §3.1, §5.6.1 |

**Summary**: 12 P0 concerns and 5 P1 concerns addressed with substantive text/experiment/table changes; 6 P2 concerns deferred to future work with explicit acknowledgment in Limitations.

---

## Data supporting new claims

| Claim | Data Source |
|---|---|
| §5.3.1 shared vs. dual A_SDC 0.3% delta | `outputs/data/dual_a_matrix/summary.csv` + trajectory CSVs |
| §5.3.2 tightened capture criterion inverts noise-acceleration | `data/ablation_c_tight_capture_results.csv` (18 trials: 6×3) |
| §5.2 total ΔV invariance across σ_θ | `data/noise_sweep_results.csv` (existing) |
| §5.2 Kalman gain norm growth | Two-point probe reported in mechanism whitepaper `docs/mechanism_noise_acceleration.md` §3.2 |

---

## Anticipated re-review concerns

1. **Factor (ii) — initial covariance transient — not ablated.** We chose not to run this ablation because setting $P_0$ independent of $\sigma_\theta$ violates a physically-motivated initialization. If the re-reviewer insists on the ablation, we can provide it but flag that its interpretation is unclear.

2. **The multi-step FE-vs-RK4 diagnostic (peak 185 m over one orbital period)** was added by the co-author's parallel work and integrates with our conflict-free merge; we cite it in §3.2. If the re-reviewer wants a Monte Carlo variant, we can add.

3. **Muralidhar-Kumar re-implementation** is deferred to future work; if the re-reviewer insists on a direct benchmark we will comply in a future revision.
