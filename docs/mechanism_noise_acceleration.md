# Mechanism of the Counterintuitive Sensor-Noise Acceleration in an Angles-Only EKF+SDRE Spacecraft Rendezvous Loop

*Technical whitepaper — supplement to a manuscript submitted to Acta Astronautica.*

*Scope: this document articulates the mechanism actually producing the observed phenomenon, states the empirical fingerprints that support it, and offers falsifiable predictions and ablation experiments by which the mechanism can be tested — or overturned. It is written in a neutral, peer-review-style register.*

---

## 1. Executive summary

An angles-only EKF+SDRE closed loop with a common state-dependent-coefficient (SDC) matrix $A_{\mathrm{SDC}}(\hat{x}_{k|k})$ driving both filter and controller exhibits a **paradoxical monotone acceleration** of capture time as angular sensor noise $\sigma_\theta$ is raised over two decades (from $0.001^\circ$ to $0.1^\circ$): capture time drops from 87,000 s to 38,220 s (2.3×) while total $\Delta V$ stays flat within 1% (10.27 → 10.19 km/s), peak thrust is invariant at the ARE ceiling (1.955 m/s²), the Kalman gain norm $\|K\|_F$ *rises* by 1.7× rather than falling as LTI Riccati theory predicts, and terminal relative velocity $v_{\mathrm{rel}}$ inflates ~20× (0.09 → 1.98 m/s). Certainty-equivalent LQG with the SDC linearization predicts none of this — under Gaussian estimation error, $\mathbb{E}[u]$ is unchanged by $\sigma_\theta$ and only $\mathrm{Var}[u]$ grows. The mechanism actually responsible has three composable factors: **(F1)** a filter-controller coupling through $A_{\mathrm{SDC}}(\hat{x}_{k|k})$ that breaks LQG separation and lets $K$ grow with $\sigma_\theta$; **(F2)** an initialization effect through the $\sigma_\theta$-scaled $P_0$ that front-loads corrections into the first control cycles; and **(F3)** a first-passage bias — the capture criterion is a fixed-radius sphere transit, and inflated terminal $v_{\mathrm{rel}}$ turns "arriving softly" into "flying past," reducing the crossing time. The paper's original "innovation amplification" heuristic mischaracterizes the effect because it predicts more integrated control effort, which is empirically ruled out by the invariant $\Delta V$. Each factor below is stated together with a falsifiable prediction that a critical reviewer can use to challenge it.

## 2. The phenomenon and why the naive explanation fails

### 2.1 Six-row summary (from `data/noise_sweep_results.csv`, 5 seeds per row)

| $\sigma_\theta$ (deg) | Capture time (s) | Total $\Delta V$ (km/s) | Mean thrust (mm/s²) | Peak thrust (m/s²) | Position RMSE (km) | Terminal $v_{\mathrm{rel}}$ (m/s) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.001 | 86,994 | 10.269 | 0.118 | 1.955 | 5.09 | 0.089 |
| 0.004 | 68,478 | 10.113 | 0.148 | 1.955 | 6.73 | 0.084 |
| 0.008 | 54,946 | 10.034 | 0.183 | 1.955 | 8.04 | 0.160 |
| 0.02  | 40,924 | 10.063 | 0.246 | 1.955 | 9.42 | 1.045 |
| 0.05  | 38,724 | 10.144 | 0.262 | 1.955 | 9.13 | 1.066 |
| 0.10  | 38,220 | 10.189 | 0.267 | 1.955 | 8.66 | 1.757 |

Fixed conditions: $a_c=15{,}000$ km, $e_c=0.5$, $\rho_0=866$ km, $\Delta t=10$ s, $R_{\mathrm{ctrl}}=10^{13}\,I_3$, $Q_{\mathrm{ctrl}}=I_6$, $\gamma=\sqrt{2}$. Success criterion: $\rho<100$ m at first crossing.

### 2.2 The classical LQG reading and why it fails

For the SDRE feedback law $u_c = -R_{\mathrm{ctrl}}^{-1} B^T P\, \hat{x}_{k|k}$, and under the classical certainty-equivalence assumption that $\hat{x}_{k|k} = x + e$ with $e \sim \mathcal{N}(0,\hat{P})$, one obtains
$$
\mathbb{E}[u_c] = -R_{\mathrm{ctrl}}^{-1} B^T P\, x,\qquad
\mathrm{Var}[u_c] = R_{\mathrm{ctrl}}^{-1} B^T P\, \hat{P}\, P B R_{\mathrm{ctrl}}^{-1}.
$$
Only the second moment carries $\sigma_\theta$ dependence, through $\hat{P}$. The LQG cost-to-go absorbs the variance as an additive control-effort penalty $\mathrm{tr}(K^T R_{\mathrm{ctrl}} K\,\hat{P})$ per unit time; it does **not** rescale the effective $Q/R$ balance and does **not** make the deterministic feedback more aggressive on average. Under this reading a modest RMS growth in $\|u_c\|$ is expected, but the *mean trajectory* and hence *mean capture time* should be essentially insensitive to $\sigma_\theta$.

### 2.3 The falsifying observation

The 100× sweep in $\sigma_\theta$ leaves total $\Delta V$ invariant within <1%. Mean thrust ratio (2.33×) matches the reciprocal capture-time ratio (2.34×) — same integrated impulse, compressed. This directly contradicts *any* "more aggressive controller" story, whether classical or heuristic: the closed loop is not injecting more energy, it is depositing the same energy differently. That single fact eliminates a whole class of proposed mechanisms.

## 3. Factor 1 — Nonlinear filter–controller coupling breaks LQG separation

### 3.1 The coupling

In the unified-SDC architecture, the same matrix $A_{\mathrm{SDC}}$ is used by the EKF prediction step and by the SDRE controller. That matrix is evaluated at the filter estimate, not at the true state — see `aerospace/simulation/nerm_ekf_sdre.py` lines 143–155:

```
x_ctrl = self.ekf.x           # when rng is not None
X_e_est = X_p_true - x_ctrl
A_SDC = self.dynamics.get_SDC_matrix(X_p_true, X_e_est, r_c, nu_dot, nu_ddot)
```

Because $x_{\mathrm{ctrl}} = \hat{x}_{k|k}$ carries the estimation error $e_k$, we can write $A_{\mathrm{SDC}} = A_{\mathrm{SDC}}(x + e_k)$. Consequently the ARE
$$
A_{\mathrm{SDC}}^T P + P A_{\mathrm{SDC}} - P B R_{\mathrm{eff}}^{-1} B^T P + Q_{\mathrm{ctrl}} = 0
$$
has a solution $P = P(A_{\mathrm{SDC}}(x+e_k))$, and the feedback gain
$$
K = R_{\mathrm{eff}}^{-1} B^T P
$$
is a **nonlinear function of $e_k$**. In LTI LQG this pathway does not exist — the controller sees no filter output through $A$ — and separation holds by construction. Here separation is broken at the point where $A_{\mathrm{SDC}}$ is evaluated. This is a structural consequence of unified SDC parameterization (paper §3.1) and is not an artifact of implementation.

### 3.2 Empirical fingerprint

At mid-approach the paper reports $\|K\|_F$ growing from $1.44\times 10^{-3}$ at $\sigma_\theta=0.001^\circ$ to $2.50\times 10^{-3}$ at $\sigma_\theta=0.1^\circ$ — a factor of 1.7. Pure LTI Riccati reasoning would predict $\|K\|_F$ to *shrink* with larger measurement noise, because $R_{\mathrm{EKF}}\propto\sigma_\theta^2$ enters the Kalman filter Riccati (not the control Riccati) and enlarges $\hat{P}$; larger $\hat{P}$ does not directly enlarge the control gain in the classical decomposition. In the unified-SDC loop, the growth in $\|K\|_F$ must come from the ARE seeing a *different* $A_{\mathrm{SDC}}$ when $e_k$ is large. The measured 1.7× growth is a direct fingerprint of Factor 1.

### 3.3 Why this contributes to acceleration (rather than instability)

The perturbed $A_{\mathrm{SDC}}(x+e_k)$ generally has different eigenstructure than the ideal $A_{\mathrm{SDC}}(x)$. Empirically the effect is not catastrophic — the ARE remains solvable and the ARE ceiling on $\|u_c\|$ holds (peak thrust flat at 1.955 m/s²), but the *distribution* of thrust in time is redirected: the gain $K$ increases in specific principal directions rather than uniformly, and the same $\Delta V$ ends up biased toward closing the range. Note the interpretation is qualitative — we did not diagonalize $P(e_k)$ analytically in this study.

### 3.4 Falsifiable prediction (F1)

*Freeze $A_{\mathrm{SDC}}$ at the true state.* In simulation this is trivial: pass `A_fixed = get_SDC_matrix(X_p_true, X_e_true, ...)` at every step (the code already supports `A_fixed`, line 78 and lines 150–151 of `nerm_ekf_sdre.py`). Under this ablation, F1 is disabled — the controller sees $A_{\mathrm{SDC}}(x)$ regardless of $e_k$ — while F2 and F3 remain active. Prediction: **the acceleration effect should attenuate substantially**, and the residual acceleration should be attributable to F2 and F3 alone. If instead the sweep still shows 2.3× acceleration under `A_fixed`, F1 is not the dominant factor and this section is wrong.

## 4. Factor 2 — Initial-transient effect through $P_0\propto\sigma_\theta^2$

### 4.1 The initialization

`main.py` lines 57–61 construct
$$
P_0 = \mathrm{diag}\big((\rho_0\sigma_\theta)^2 I_3,\ \sigma_\theta^2 I_3\big).
$$
This is a physically motivated choice — it says "we don't know range at all, so the position covariance scales like the ranging ambiguity, and the velocity covariance is a fixed small fraction of the range-rate uncertainty." But it means that the *initial* filter behavior is $\sigma_\theta$-dependent even though $\hat{x}_0$ is set to the true relative state (line 93).

### 4.2 What that does to the first cycle

At $k=0$, $P_0$ is large and $\hat{x}_0$ is correct. The first predict–update cycle produces a Kalman gain
$$
K_0 = P_{0|0}\,H^T (H P_{0|0} H^T + R_{\mathrm{EKF}})^{-1}
$$
in which $P_{0|0}\propto\sigma_\theta^2$ and $R_{\mathrm{EKF}}\propto\sigma_\theta^2$; the ratio is $O(1)$, so $K_0$ is *not* small even at large $\sigma_\theta$. The first innovation is $z_0-\hat{z}_0 \sim \mathcal{N}(0,R_{\mathrm{EKF}})$, so the first correction $K_0(z_0-\hat{z}_0)$ has magnitude $\sim\sigma_\theta$. That correction perturbs $\hat{x}_{1|1}$ off the truth, and the correction enters the *next* SDRE control update (through both $\hat{x}_{1|1}$ and $A_{\mathrm{SDC}}(\hat{x}_{1|1})$ — Factors 1 and 2 compound here).

This is not a "more control effort" story: total $\Delta V$ is invariant. It is a *timing* story. A larger initial correction rotates the closed-loop trajectory in a way that puts thrust earlier in the trajectory when the range is still large and the geometry has more leverage. Once the range collapses, the ARE-bounded feedback saturates near its ceiling regardless of $\sigma_\theta$, so the late-phase thrust profile is nearly noise-independent — the difference is all in when the closing thrust starts.

### 4.3 Empirical fingerprint

Prediction: at higher $\sigma_\theta$, the *cumulative* $\Delta V$ curve $\int_0^t\|u_c(\tau)\|\,d\tau$ should rise earlier (and hence flatten earlier at the same total). Equivalently, the early-phase (first 10% of the trajectory) fraction of total $\Delta V$ should be a monotone increasing function of $\sigma_\theta$. This has not been computed for the current sweep and is one of the recommended ablations (§8).

### 4.4 Falsifiable prediction (F2)

*Fix $P_0$ across the sweep.* Set $P_0 = P_0^{\mathrm{high\text{-}noise}}$ (i.e. the value used at $\sigma_\theta=0.1^\circ$) for every noise level. This decouples $P_0$ from the actual $\sigma_\theta$. Prediction: at low $\sigma_\theta$ the capture time should decrease (become more like the high-noise case) because the initial transient is now large; and at high $\sigma_\theta$ the capture time should barely change. If instead the sweep is unaffected by $P_0$ scaling, F2 is negligible relative to F1 and F3.

## 5. Factor 3 — First-passage bias at inflated terminal velocity

### 5.1 The success criterion

Capture is defined as the first crossing of $\rho<100$ m (see `nerm_ekf_sdre.py` line 189). It is a *sphere-transit* criterion, not a rendezvous criterion. A trajectory that arrives at the target with $v_{\mathrm{rel}}=0.1$ m/s "arrives" — it spends the last 1000 s inside the ball. A trajectory that flies past at $v_{\mathrm{rel}}=2$ m/s crosses the boundary once and exits; but that single crossing satisfies the criterion.

### 5.2 The terminal-velocity inflation

From `data/terminal_vrel_results.csv` (median across seeds per level): $v_{\mathrm{rel}}$ grows from 0.088 m/s at $\sigma_\theta=0.001^\circ$ to 1.98 m/s at $\sigma_\theta=0.1^\circ$ — a factor of ~22, roughly linear in $\sigma_\theta$ above $\sigma_\theta\gtrsim 0.008^\circ$.

### 5.3 The first-passage argument

Consider a stochastic trajectory approaching a small sphere of radius $r_c=100$ m at bulk speed $v_{\mathrm{rel}}$. In a "soft" approach ($v_{\mathrm{rel}}$ small), the trajectory decelerates in a neighborhood of the target, and the first crossing occurs near the (long) mean arrival time. In a "flyby" approach ($v_{\mathrm{rel}}$ large), the trajectory approaches at nearly constant speed and the first crossing is close to the deterministic geometric prediction $\rho_0/v_{\mathrm{rel}}$, which shortens roughly linearly in $v_{\mathrm{rel}}$. The observed capture-time compression saturates around $\sigma_\theta \sim 0.02^\circ$ (Table 4 shows 40.9 ks → 38.7 ks → 38.2 ks over the last three rows) — consistent with the geometric bound $\rho_0/v_{\mathrm{rel}}$ becoming the binding constraint once $v_{\mathrm{rel}}$ is large enough.

### 5.4 This is not free lunch

The mission physics matters here. If the mission is truly "get inside 100 m of the target and then coast," a flyby at 2 m/s is *not* a mission success — the chaser exits the ball in less than a minute. The paper's success criterion accepts this, but a follow-up mission with a soft-rendezvous constraint would penalize it heavily.

### 5.5 Falsifiable prediction (F3)

*Tighten the success criterion.* Require simultaneously $\rho<100$ m and $v_{\mathrm{rel}}<0.1$ m/s. Under this criterion, prediction: at high $\sigma_\theta$ the "capture" event of the current definition is not a rendezvous; the tightened criterion will not be satisfied at the current crossing time and — if satisfied at all in the tail — will occur at a much later time. The monotone acceleration should *reverse*, with high-$\sigma_\theta$ runs failing the tightened criterion more often. If instead the tightened success rate is high across the sweep, F3 is not the dominant factor.

## 6. Composition — how the three factors combine

The three factors are not independent contributions in the additive sense; they are compositions. F2 sets a $\sigma_\theta$-scaled initial estimation error and covariance; F1 turns that estimation error into a *gain* perturbation and a redistribution of the closed-loop trajectory; F3 turns the resulting closed-loop trajectory into a shortened first-passage time through the capture sphere. Removing any one of them should attenuate the effect; removing all three should return the loop to the LQG-baseline prediction (no acceleration).

A quantitative decomposition of the observed 2.3× acceleration into F1, F2, F3 contributions requires running the three ablations of §8 (A, B, C). Absent those runs, we can only state qualitative shares:

- F3 (sphere-transit bias) likely dominates at the high-$\sigma_\theta$ end of the sweep, where terminal $v_{\mathrm{rel}}$ is large and the capture time saturates around 38 ks — consistent with a geometric bound.
- F2 (initial transient) likely dominates the *knee* of the acceleration curve near $\sigma_\theta \in [0.001^\circ, 0.02^\circ]$, where capture time drops most steeply while $v_{\mathrm{rel}}$ is still small.
- F1 (gain coupling) is the mechanism most responsible for the observed 1.7× growth in $\|K\|_F$ — a signature no other factor can explain — and modulates both F2 and F3 by shaping how the estimation error propagates into control.

This decomposition is a hypothesis. It should be treated as such. The paper's original "innovation amplification dominates gain attenuation" heuristic is directionally suggestive — larger innovations do produce visibly larger corrections, especially in the early phase where F2 lives — but it is not analytically valid because it predicts *more control effort*, which is empirically ruled out. The heuristic conflates innovation magnitude with integrated control effort; the correct reading is that innovations *rearrange* control effort in time, not augment it.

## 7. What this means for the paper

1. **Diagnostic significance of the phenomenon.** The counterintuitive noise sensitivity is not an anomaly to be explained away — it is a *positive test* of the unified-SDC architecture's central claim (paper §3.1). Under a decoupled architecture where the EKF and the SDRE use different linearizations, F1 would not exist. The presence of F1 in the measurements (the 1.7× growth in $\|K\|_F$) is direct evidence that the filter's error is entering the control gain — which is exactly what the paper argues the unified architecture achieves.

2. **A mission-planning caveat that the paper should surface.** If a follow-up mission requires soft rendezvous (low terminal $v_{\mathrm{rel}}$), then higher-noise operation is *worse*, not better. The "sensor accuracy is less stringent than conventionally assumed" claim (paper §5.2 engineering implication) is contingent on the sphere-transit success criterion. Under a $(\rho<100\ \mathrm{m}) \land (v_{\mathrm{rel}}<0.1\ \mathrm{m/s})$ criterion the recommendation may reverse.

3. **What the whitepaper does not claim.** The three factors have not been isolated by ablation. The composition argument in §6 is a plausible hypothesis, not a proof. A reviewer who wants to challenge the mechanism should run the ablations of §8; if any of the F1/F2/F3 predictions fail, the corresponding factor is out. If all three predictions fail simultaneously, the mechanism proposed here is wrong.

## 8. Recommended experimental ablations for a follow-up paper

**Ablation A — Freeze $A_{\mathrm{SDC}}$ at $x_{\mathrm{true}}$.** Use `A_fixed = get_SDC_matrix(X_p, X_e, r_c, nu_dot, nu_ddot)` recomputed each step from the truth (already supported by the simulation code via `A_fixed` — the parameter is stored but currently accepts only a single static matrix; a minor extension would allow per-step recomputation). Isolates F1. Sweep $\sigma_\theta \in \{0.001, 0.008, 0.04, 0.1\}^\circ$ with 20+ seeds. Report capture-time ratio vs. baseline; report $\|K\|_F$ vs. $\sigma_\theta$ (predicted: flat, unlike the 1.7× growth in the baseline).

**Ablation B — Fix $P_0$ at $\sigma_\theta = 0.1^\circ$ scale for all runs.** Removes the $\sigma_\theta$ dependence of the initial covariance. Isolates F2. Report the fraction of total $\Delta V$ deposited in the first 10% of the trajectory as a function of $\sigma_\theta$; predicted: nearly flat.

**Ablation C — Tighten success criterion to $(\rho<100\ \mathrm{m}) \land (v_{\mathrm{rel}}<0.1\ \mathrm{m/s})$.** Isolates F3. Report tightened-success rate vs. $\sigma_\theta$ and the ratio (tightened capture time) / (baseline capture time). Predicted: monotone acceleration should not survive, and may reverse.

**Ablation D — Sweep $R_{\mathrm{ctrl}}$ at fixed $\sigma_\theta=0.008^\circ$.** Determines whether the observed peak-thrust ceiling of 1.955 m/s² scales as $R_{\mathrm{ctrl}}^{-1/2}$ (the ARE bound $\|R_{\mathrm{ctrl}}^{-1} B^T P\|$ prediction for a fixed-$A$ Riccati). This decouples the ARE-ceiling mechanism from the noise-acceleration mechanism. If the peak-thrust ceiling holds, the invariance of peak thrust across the noise sweep is confirmed as an ARE-structural property and not a noise-related artifact.

Suggested primary output: a table of capture times and $\Delta V$s across the (A, B, C) $\times$ ($\sigma_\theta$) grid, plus one figure showing the cumulative-$\Delta V$ curve for the four sweep endpoints in each ablation.

---

## Appendix A — Code inspection

The mechanism claims of §3–§5 depend on specific structural properties of the closed-loop code. This appendix cites the load-bearing lines.

### A.1 Unified-SDC linearization at $\hat{x}$ (support for F1)

`aerospace/simulation/nerm_ekf_sdre.py`:

- Lines 143–155: choice of linearization point.
  - Line 146: `if self.rng is not None:` selects the noisy branch.
  - Line 147: `x_ctrl = self.ekf.x` — the controller reads the *filter estimate*, not the truth. (In the noise-free branch, line 149 uses `state[0:6] - state[6:12]`, i.e. the true relative state.)
  - Lines 153–155: `A_SDC = self.dynamics.get_SDC_matrix(X_p_true, X_e_est, ...)` where `X_e_est = X_p_true - x_ctrl`. So $A_{\mathrm{SDC}}$ inherits $e_k$ through $X_e_{\mathrm{est}}$.
- Lines 159–162: `self.controller.compute_control(A_SDC, x_ctrl, ...)`. The controller's `A_SDC` and its argument state `x_ctrl` come from the same $\hat{x}_{k|k}$.
- Lines 173–183: EKF predict/update *reuses* the same `A_SDC` (line 178: `self.ekf.predict(A_SDC, ...)`). This is the "same matrix" architecture — its explicit code-level manifestation.

`aerospace/control/sdre.py`:

- Lines 56–73: $R_{\mathrm{eff}} = R/(1-\gamma^{-2})$ and $S = B R_{\mathrm{eff}}^{-1} B^T$ are precomputed once; $B_p = [0_3;\ I_3]$, $B_e = -B_p$.
- Lines 133–203: `compute_control` — line 164 solves the ARE if `solve_are=True`, line 166 calls `_solve_are_balanced(A_SDC)` which returns $P$ dependent on $A_{\mathrm{SDC}}$; line 192 forms $u_p = -R_{\mathrm{eff}}^{-1} B_p^T P\, x_{\mathrm{rel}}$; line 196 forms $u_e$. Both use the same $P$ and thus the same $A_{\mathrm{SDC}}$. Any $e_k$ dependence in $A_{\mathrm{SDC}}$ propagates through $P$ into both feedback laws.

### A.2 $\sigma_\theta$-scaled initial covariance (support for F2)

`main.py` lines 23–66, specifically:

- Line 38: `sigma_ang = (0.008 * DEG2RAD) if noisy else 0.0`.
- Lines 41–44 (angles-only branch): $R_{\mathrm{meas}} = \mathrm{diag}(\sigma_\theta^2, \sigma_\theta^2)$.
- Line 54: process noise $Q_{\mathrm{proc}}$ is fixed independent of $\sigma_\theta$.
- Lines 57–61: **the initial covariance construction**:
  ```
  sigma_pos = initial_dist * sigma_ang    # km
  sigma_vel = 1.0 * sigma_ang             # km/s
  P0 = np.diag([sigma_pos**2, sigma_pos**2, sigma_pos**2,
                sigma_vel**2, sigma_vel**2, sigma_vel**2])
  ```
  Both position and velocity variances scale as $\sigma_\theta^2$.

### A.3 EKF linearization and gain (support for F1)

`aerospace/estimation/ekf.py`:

- Lines 36–46 (`measure`): standard angles-only observation.
- Lines 53–77 (`meas_jacobian`): the $H$ matrix is a function of $\hat{x}_{k|k}$; it too depends on the estimate. This is a *second* coupling channel — the observation Jacobian is state-dependent — that the mechanism argument of F1 does not need but that reinforces it. LTI Kalman-filter Riccati bounds do not apply because $H = H(\hat{x})$.
- Lines 81–93 (`predict`): the transition matrix is $F = I + A\Delta t$ where $A = A_{\mathrm{SDC}}$ from the controller. This is the exact structural manifestation of "same matrix."
- Lines 95–113 (`update`): innovation is wrapped (line 106); Kalman gain $K = P_{\mathrm{priori}} H^T S^{-1}$ (line 110). $H$ carries $e_k$, $S$ carries both $\hat{P}$ and $R_{\mathrm{EKF}}$, and hence $K = K(e_k, \sigma_\theta)$.

### A.4 Support for the invariant peak-thrust observation (context for Ablation D)

The ARE returns $P$ with $\|R_{\mathrm{eff}}^{-1} B^T P\|$ bounded above; the SDRE code applies symplectic balancing (`sdre.py` lines 75–113) to keep $P$ well-conditioned. The peak thrust of 1.955 m/s² across the noise sweep is the ARE-structural ceiling for the choice $(Q_{\mathrm{ctrl}}=I_6, R_{\mathrm{ctrl}}=10^{13} I_3, \gamma=\sqrt{2})$; the noise sweep does not change this ceiling because $A_{\mathrm{SDC}}$ perturbations under F1 change the *direction* of the feedback more than its worst-case magnitude.

---

## Appendix B — Notation

| Symbol | Meaning |
|---|---|
| $A_{\mathrm{SDC}}(x)$ | $6\times 6$ state-dependent coefficient matrix for the relative-motion dynamics; evaluated at $\hat{x}_{k|k}$ in the loop |
| $B$ | $6\times 3$ control input matrix, $[0_3;\ I_3]$ |
| $P$ | ARE solution matrix, $6\times 6$ |
| $K$ | Feedback gain, $K = R_{\mathrm{eff}}^{-1} B^T P$ |
| $R_{\mathrm{ctrl}}$ | SDRE control weighting matrix ($10^{13} I_3$ baseline) |
| $R_{\mathrm{eff}}$ | Effective control weighting after game modification, $R_{\mathrm{ctrl}}/(1-\gamma^{-2})$ |
| $Q_{\mathrm{ctrl}}$ | SDRE state weighting matrix ($I_6$ baseline) |
| $R_{\mathrm{EKF}}$ | Measurement noise covariance in the EKF, $\mathrm{diag}(\sigma_\theta^2, \sigma_\theta^2)$ (angles-only) |
| $\hat{P}$ | Filter covariance $P_{k|k}$ |
| $\sigma_\theta$ | Angular measurement 1-$\sigma$ noise, in degrees or radians as noted |
| $e_k$ | Filter estimation error, $\hat{x}_{k|k} - x_k$ |
| $\rho_0$ | Initial relative range, 866 km in the baseline |
| $v_{\mathrm{rel}}$ | Relative speed magnitude at capture crossing |
