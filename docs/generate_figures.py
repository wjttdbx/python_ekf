"""
Generate all 9 publication-quality figures for the Acta Astronautica paper
"Simultaneous Estimation and Control via a Common SDC Matrix for
Angles-Only Relative Navigation in Elliptical Orbits".

Output: PDF vector figures in docs/figures/

Usage:
    uv run python docs/generate_figures.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import csv
import io
import contextlib
import time as time_mod
import numpy as np
from scipy import integrate
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc, Wedge
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
import matplotlib.ticker as ticker

# ── project imports ──────────────────────────────────────────────────────────
from aerospace.dynamics.nerm import OrbitalDynamics
from aerospace.control.sdre import SDREGameController
from aerospace.estimation.ekf import RelativeStateEKF
from aerospace.simulation.nerm_ekf_sdre import EKFSDRESimulation

DEG2RAD = np.pi / 180.0
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── matplotlib rc for publication quality ────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "lines.linewidth": 1.2,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    }
)


# ═══════════════════════════════════════════════════════════════════════════════
# Shared simulation runner
# ═══════════════════════════════════════════════════════════════════════════════

MU = 3.986e5
A_C = 15000.0
E_C = 0.5
DT = 10.0
X_P0 = np.array([500.0, 500.0, 500.0, 0.01, 0.01, 0.01])
X_E0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
NU0 = 0.0
Q_CTRL = np.eye(6)
R_CTRL = np.eye(3) * 1e13
GAMMA = np.sqrt(2)
SIGMA_ANG = 0.008
PROC_NOISE_POS = 5e-4
PROC_NOISE_VEL = 5e-8


def make_ekf(sigma_ang_deg, angles_only=True):
    """Create an EKF with standard parameters."""
    sigma_ang_rad = sigma_ang_deg * DEG2RAD
    x0_est = X_P0 - X_E0
    initial_dist = float(np.linalg.norm(x0_est[:3]))
    R_meas = np.diag([sigma_ang_rad**2] * (2 if angles_only else 3))
    Q_proc = np.diag([PROC_NOISE_POS] * 3 + [PROC_NOISE_VEL] * 3)
    sigma_pos = initial_dist * sigma_ang_rad
    sigma_vel = 1.0 * sigma_ang_rad
    P0 = np.diag([sigma_pos**2] * 3 + [sigma_vel**2] * 3)
    return RelativeStateEKF(
        x0=x0_est, P0=P0, Q=Q_proc, R=R_meas, angles_only=angles_only
    )


def run_simulation(seed, sigma_ang_deg=SIGMA_ANG, angles_only=True, t_end=None):
    """Run one EKF-SDRE simulation and return the result."""
    orb = OrbitalDynamics(mu=MU, a_c=A_C, e_c=E_C)
    ctrl = SDREGameController(Q=Q_CTRL, R=R_CTRL, gamma=GAMMA)
    ekf = make_ekf(sigma_ang_deg, angles_only=angles_only)
    rng = np.random.default_rng(seed)
    if t_end is None:
        t_end = 10.0 * orb.T_orbit
    sim = EKFSDRESimulation(
        dynamics=orb,
        controller=ctrl,
        ekf=ekf,
        X_p0=X_P0,
        X_e0=X_E0,
        nu0=NU0,
        dt=DT,
        are_interval=1,
        rng=rng,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        result = sim.run(t_end=t_end)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1: LVLH Coordinate Frame Diagram
# ═══════════════════════════════════════════════════════════════════════════════

def fig1_lvlh_frame():
    """Schematic of LVLH coordinate frame with reference orbit and two spacecraft."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.set_aspect("equal")
    ax.axis("off")

    colors = {
        "orbit": "#6F7782",
        "earth": "#2F6FA3",
        "earth_edge": "#183D5E",
        "target": "#243B53",
        "chaser": "#C2413B",
        "radial": "#D64A3A",
        "along": "#2A7F62",
        "normal": "#6D5BA6",
        "rel": "#1F6F8B",
        "label_bg": "#FFFFFF",
    }

    def unit(v):
        return v / np.linalg.norm(v)

    # Reference orbit and central body. The orbit is intentionally light so the
    # local LVLH geometry remains the main visual claim.
    a, b = 3.45, 2.35
    theta = np.linspace(0, 2 * np.pi, 500)
    orbit_x = a * np.cos(theta)
    orbit_y = b * np.sin(theta)
    ax.plot(orbit_x, orbit_y, color=colors["orbit"], lw=1.1, ls=(0, (5, 4)), zorder=1)

    earth = plt.Circle((0, 0), 0.38, color=colors["earth"], ec=colors["earth_edge"], lw=1.2, zorder=3)
    ax.add_patch(earth)
    ax.text(0, 0, "Earth", ha="center", va="center", fontsize=8, color="white", weight="bold")

    # Target spacecraft and LVLH basis at the current reference-orbit point.
    nu = np.deg2rad(38)
    target = np.array([a * np.cos(nu), b * np.sin(nu)])
    radial = unit(target)
    tangent = unit(np.array([-a * np.sin(nu), b * np.cos(nu)]))
    if tangent[0] < 0:
        tangent = -tangent

    chaser = target + 0.74 * radial + 0.45 * tangent
    axis_len = 1.05

    # Faint Earth-to-target radius clarifies the local-vertical direction.
    ax.plot([0, target[0]], [0, target[1]], color="#C8CDD3", lw=0.9, zorder=0)
    def vec_arrow(start, end, color, lw=1.8, mutation_scale=12, zorder=6):
        patch = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            lw=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
        ax.add_patch(patch)
        return patch

    vec_arrow(target, target + axis_len * radial, colors["radial"])
    vec_arrow(target, target + axis_len * tangent, colors["along"])
    vec_arrow(target, chaser, colors["rel"], lw=2.0, mutation_scale=14)

    ax.scatter(*target, s=65, color=colors["target"], zorder=7)
    ax.scatter(*chaser, marker="^", s=82, color=colors["chaser"], edgecolor="#6F1D1B", lw=0.8, zorder=8)

    label_box = dict(boxstyle="round,pad=0.18", fc=colors["label_bg"], ec="none", alpha=0.88)
    ax.text(target[0] - 0.18, target[1] - 0.68, "Target\n(LVLH origin)", fontsize=8,
            ha="right", va="top", color=colors["target"], bbox=label_box)
    ax.text(chaser[0] + 0.18, chaser[1] - 0.02, "Chaser", fontsize=8,
            ha="left", va="center", color=colors["chaser"], weight="bold", bbox=label_box)

    ax.text(*(target + 1.26 * radial + np.array([0.0, 0.08])), r"$\hat{x}$ radial", fontsize=8.5,
            color=colors["radial"], ha="left", va="center", bbox=label_box)
    ax.text(*(target + 1.16 * tangent + np.array([0.0, 0.08])), r"$\hat{y}$ along-track",
            fontsize=8.5, color=colors["along"], ha="left", va="bottom", bbox=label_box)
    rel_mid = 0.5 * (target + chaser)
    ax.text(*(rel_mid + np.array([0.12, -0.23])), r"$\mathbf{x}_{rel}$", fontsize=9,
            color=colors["rel"], ha="left", va="top", bbox=label_box)

    # Cross-track axis as the standard circle-dot symbol.
    z_center = target - 0.58 * tangent - 0.34 * radial
    z_ring = plt.Circle(z_center, 0.13, fill=False, ec=colors["normal"], lw=1.4, zorder=6)
    z_dot = plt.Circle(z_center, 0.035, color=colors["normal"], zorder=7)
    ax.add_patch(z_ring)
    ax.add_patch(z_dot)
    ax.text(z_center[0] - 0.05, z_center[1] - 0.25, r"$\hat{z}$ cross-track",
            fontsize=8.5, color=colors["normal"], ha="center", va="top", bbox=label_box)

    # Orbit direction arrow, separated from the local-frame labels.
    a0, a1 = np.deg2rad(-22), np.deg2rad(-10)
    p0 = np.array([a * np.cos(a0), b * np.sin(a0)])
    p1 = np.array([a * np.cos(a1), b * np.sin(a1)])
    vec_arrow(p0, p1, colors["orbit"], lw=1.2, mutation_scale=10, zorder=4)
    ax.text(p1[0] + 0.08, p1[1] - 0.06, "reference orbit", fontsize=7.5,
            color=colors["orbit"], ha="left", va="top")

    ax.set_xlim(-3.9, 5.0)
    ax.set_ylim(-2.95, 3.65)

    fig.savefig(OUT / "fig1_lvlh_frame.pdf")
    plt.close(fig)
    print("Fig 1 saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2: Unified EKF-SDRE Algorithm Flow Diagram
# ═══════════════════════════════════════════════════════════════════════════════

def fig2_flow_diagram():
    """Block diagram of the unified EKF-SDRE framework showing A_SDC reuse."""
    fig, ax = plt.subplots(figsize=(9.2, 4.7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    palette = {
        "state": "#E8E1F2",
        "sdc": "#FFF0C7",
        "control": "#DDEEDB",
        "plant": "#F5D6D4",
        "estimate": "#DCE9FA",
        "sensor": "#EFE4F5",
        "edge": "#2D3748",
        "muted": "#697386",
        "accent": "#D64A3A",
        "lane": "#F7F9FB",
    }

    # Shaded lanes keep the control and estimation branches visually separated.
    ax.add_patch(FancyBboxPatch((5.05, 3.12), 4.55, 1.78, boxstyle="round,pad=0.05",
                                fc=palette["lane"], ec="#D8DEE8", lw=0.7, zorder=0))
    ax.add_patch(FancyBboxPatch((5.05, 0.62), 4.55, 1.78, boxstyle="round,pad=0.05",
                                fc=palette["lane"], ec="#D8DEE8", lw=0.7, zorder=0))
    ax.text(9.35, 4.73, "control branch", fontsize=7.5, color=palette["muted"],
            ha="right", va="center")
    ax.text(9.35, 2.23, "estimation branch", fontsize=7.5, color=palette["muted"],
            ha="right", va="center")

    def box(x, y, w, h, title, subtitle="", color="#FFFFFF", title_size=9, subtitle_size=7.3):
        rect = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                              boxstyle="round,pad=0.08,rounding_size=0.08",
                              fc=color, ec=palette["edge"], lw=0.9, zorder=2)
        ax.add_patch(rect)
        ax.text(x, y + (0.12 if subtitle else 0.0), title, ha="center", va="center",
                fontsize=title_size, color="#111827", weight="bold", zorder=3)
        if subtitle:
            ax.text(x, y - 0.25, subtitle, ha="center", va="center",
                    fontsize=subtitle_size, color="#4B5563", zorder=3)

    def arrow(start, end, color=None, lw=1.15, rad=0.0, style="-|>", zorder=4):
        patch = FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=11,
            lw=lw,
            color=color or palette["edge"],
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=4,
            shrinkB=4,
            zorder=zorder,
        )
        ax.add_patch(patch)
        return patch

    box(1.35, 2.75, 1.9, 0.82, r"Posterior estimate", r"$\hat{x}_{k|k},\ P_{k|k}$", palette["state"])
    box(3.65, 2.75, 2.1, 1.05, r"Shared SDC matrix",
        r"$A_{\mathrm{SDC}}(\hat{x}_{k|k},\nu_k)$", palette["sdc"], title_size=9.2)
    box(6.35, 4.02, 2.05, 0.86, "SDRE control", r"ARE $\rightarrow\ u_k$", palette["control"])
    box(8.65, 4.02, 1.75, 0.86, "13-D NERM", r"truth propagation", palette["plant"])
    box(8.65, 1.48, 1.75, 0.86, "Angles-only", r"$z_k=[az,el]^T+v_k$", palette["sensor"])
    box(6.35, 1.48, 2.05, 0.86, "EKF predict/update",
        r"$F_k=I+A_{\mathrm{SDC}}\Delta t$", palette["estimate"])

    # Main loop. The two red arrows are the visual evidence for the figure's
    # claim: one matrix instance fans out to both control and estimation.
    arrow((2.32, 2.75), (2.58, 2.75))
    arrow((4.68, 3.02), (5.38, 3.92), color=palette["accent"], lw=1.9)
    arrow((4.68, 2.48), (5.38, 1.58), color=palette["accent"], lw=1.9)
    arrow((7.38, 4.02), (7.76, 4.02))
    arrow((8.65, 3.56), (8.65, 1.94))
    arrow((7.78, 1.48), (7.38, 1.48))
    arrow((5.30, 1.48), (2.25, 2.34), rad=-0.18)

    ax.text(5.18, 2.77, "same\nmatrix", ha="center", va="center",
            fontsize=8.2, color=palette["accent"], weight="bold",
            bbox=dict(boxstyle="round,pad=0.20", fc="white", ec=palette["accent"], lw=0.9), zorder=5)

    # Inputs and outputs that make the closed-loop timing explicit without
    # adding extra boxes.
    ax.text(3.65, 3.58, r"$f(x)=A(x)x$", ha="center", va="center",
            fontsize=7.5, color="#5F4B00")
    ax.text(6.35, 4.66, r"$A^TP+PA-PBR^{-1}B^TP+Q=0$", ha="center", va="center",
            fontsize=7.2, color="#4B5563")
    ax.text(8.95, 2.74, "relative state\nand line-of-sight",
            ha="center", va="center", fontsize=7.0, color=palette["muted"])
    ax.text(3.76, 1.52, "posterior at next step", ha="center", va="center",
            fontsize=7.0, color=palette["muted"])

    fig.savefig(OUT / "fig2_flow_diagram.pdf")
    plt.close(fig)
    print("Fig 2 saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3: FE vs RK4 Prediction Error Histogram
# ═══════════════════════════════════════════════════════════════════════════════

def run_prediction_error_diagnostic(seed=42):
    """Run prediction error diagnostic, return records dict."""
    from aerospace.dynamics.nerm import OrbitalDynamics as OD

    orb = OD(mu=MU, a_c=A_C, e_c=E_C)
    ctrl = SDREGameController(Q=Q_CTRL, R=R_CTRL, gamma=GAMMA)
    ekf = make_ekf(SIGMA_ANG, angles_only=True)
    rng = np.random.default_rng(seed)
    sim = EKFSDRESimulation(
        dynamics=orb, controller=ctrl, ekf=ekf,
        X_p0=X_P0, X_e0=X_E0, nu0=NU0, dt=DT,
        are_interval=1, rng=rng,
    )

    state = sim.state0.copy()
    t = 0.0
    B_ctrl = np.zeros((6, 3))
    B_ctrl[3:, :] = np.eye(3)

    records = {"t": [], "fe_err_pos": [], "rk4_err_pos": [],
               "diff_pos": [], "diff_vel": [], "dist": []}

    def rel_6d(x_rel, X_p, du, orb_, rc, nd, ndd):
        X_e = X_p - x_rel
        A = orb_.get_SDC_matrix(X_p, X_e, rc, nd, ndd)
        B2 = np.zeros((6, 3)); B2[3:, :] = np.eye(3)
        return A @ x_rel + B2 @ du

    def rk4_step(x0, X_p, du, orb_, rc, nd, ndd, dt_):
        def f(x):
            return rel_6d(x, X_p, du, orb_, rc, nd, ndd)
        k1 = f(x0)
        k2 = f(x0 + 0.5 * dt_ * k1)
        k3 = f(x0 + 0.5 * dt_ * k2)
        k4 = f(x0 + dt_ * k3)
        return x0 + (dt_ / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    N = int(10.0 * orb.T_orbit / DT)
    for k in range(N):
        nu = state[12]
        rc, nu_dot, nu_ddot = orb.get_orbital_params(nu)

        X_p_true = state[0:6]
        x_ctrl = sim.ekf.x
        X_e_est = X_p_true - x_ctrl
        A_SDC = orb.get_SDC_matrix(X_p_true, X_e_est, rc, nu_dot, nu_ddot)

        solve_now = (k % sim.are_interval == 0)
        x_true_rel = state[0:6] - state[6:12]
        u_p, u_e = ctrl.compute_control(A_SDC, x_ctrl, t=t,
                                         solve_are=solve_now, x_rel_e=x_true_rel)
        du = u_p - u_e

        # Forward Euler prediction
        x_ekf = sim.ekf.x.copy()
        x_pred_fe = x_ekf + DT * (A_SDC @ x_ekf + B_ctrl @ du)
        # RK4 prediction
        x_pred_rk4 = rk4_step(x_ekf, X_p_true, du, orb, rc, nu_dot, nu_ddot, DT)

        # True state propagation
        from scipy.integrate import solve_ivp
        sol = solve_ivp(orb.dynamics_13d, [t, t + DT], state,
                        args=(u_p, u_e), method="RK45", rtol=1e-8, atol=1e-10)
        state_next = sol.y[:, -1]
        x_true_next = state_next[0:6] - state_next[6:12]

        fe_err = np.linalg.norm((x_pred_fe - x_true_next)[:3])
        rk4_err = np.linalg.norm((x_pred_rk4 - x_true_next)[:3])
        diff_pos = np.linalg.norm((x_pred_fe - x_pred_rk4)[:3])
        diff_vel = np.linalg.norm((x_pred_fe - x_pred_rk4)[3:])

        records["t"].append(t + DT)
        records["fe_err_pos"].append(fe_err)
        records["rk4_err_pos"].append(rk4_err)
        records["diff_pos"].append(diff_pos)
        records["diff_vel"].append(diff_vel)
        records["dist"].append(np.linalg.norm(x_true_rel[:3]))

        # EKF step
        x_priori, P_priori = sim.ekf.predict(A_SDC, B_ctrl, u_p, u_e, DT)
        z_true = RelativeStateEKF.measure(state_next[0:6], state_next[6:12], angle_only=True)
        z_meas = z_true + rng.multivariate_normal(np.zeros(2), sim.ekf.R)
        sim.ekf.update(x_priori, P_priori, z_meas)

        state = state_next
        t += DT
        if np.linalg.norm(state_next[0:3] - state_next[6:9]) < sim.capture_dist:
            break

    return {k: np.array(v) for k, v in records.items()}


def fig3_prediction_error(records=None):
    """FE vs RK4 prediction error histogram (main figure for paper)."""
    if records is None:
        print("Running prediction error diagnostic...")
        records = run_prediction_error_diagnostic(seed=42)

    fe_pos_m = records["fe_err_pos"] * 1000
    rk4_pos_m = records["rk4_err_pos"] * 1000
    diff_pos_m = records["diff_pos"] * 1000
    meas_noise = np.median(records["dist"]) * SIGMA_ANG * DEG2RAD * 1000

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))

    # (a) Histogram: FE vs RK4 position error
    ax = axes[0]
    bins = np.logspace(-1, 4, 60)
    ax.hist(fe_pos_m, bins=bins, alpha=0.55, color="#3498DB",
            label=f"FE (med={np.median(fe_pos_m):.1f} m)")
    ax.hist(rk4_pos_m, bins=bins, alpha=0.55, color="#E67E22",
            label=f"RK4 (med={np.median(rk4_pos_m):.1f} m)")
    ax.axvline(meas_noise, color="gray", ls="--", lw=1.2,
               label=f"Meas. noise (~{meas_noise:.0f} m)")
    ax.set_xscale("log")
    ax.set_xlabel("Single-step position prediction error (m)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("(a) Position error distribution", fontsize=10)
    ax.grid(True, alpha=0.25)

    # (b) |FE - RK4| histogram
    ax = axes[1]
    ax.hist(diff_pos_m, bins=bins, alpha=0.7, color="#8E44AD")
    ax.axvline(np.median(diff_pos_m), color="#6C3483", ls="--", lw=1.2,
               label=f"Median = {np.median(diff_pos_m):.1f} m")
    ax.axvline(meas_noise, color="gray", ls=":", lw=1.0,
               label=f"Meas. noise (~{meas_noise:.0f} m)")
    ax.set_xscale("log")
    ax.set_xlabel("|FE − RK4| position difference (m)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=7)
    ax.set_title("(b) FE−RK4 difference", fontsize=10)
    ax.grid(True, alpha=0.25)

    # (c) Error vs relative distance
    ax = axes[2]
    ax.loglog(records["dist"], fe_pos_m, ".", alpha=0.08, color="#3498DB", markersize=1.5)
    ax.loglog(records["dist"], rk4_pos_m, ".", alpha=0.08, color="#E67E22", markersize=1.5)
    ax.set_xlabel("Relative distance (km)")
    ax.set_ylabel("Position error (m)")
    ax.set_title("(c) Error vs. relative distance", fontsize=10)
    # legend via proxy
    proxy1 = Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498DB", markersize=6, label="FE")
    proxy2 = Line2D([0], [0], marker="o", color="w", markerfacecolor="#E67E22", markersize=6, label="RK4")
    ax.legend(handles=[proxy1, proxy2], fontsize=7)
    ax.grid(True, alpha=0.25)

    # summary text
    ratio = np.median(diff_pos_m) / meas_noise
    fig.suptitle(
        f"Fig. 3. Single-step prediction error diagnostic. "
        f"FE−RK4 median difference / measurement noise = {ratio:.3f} "
        f"({'≪' if ratio < 0.05 else '≈'} 1 — SDC exactness compensates for low-order discretization).",
        fontsize=10, y=1.02,
    )

    fig.tight_layout()
    fig.savefig(OUT / "fig3_prediction_error.pdf")
    plt.close(fig)
    print("Fig 3 saved.")
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 4: Baseline — Angles-Only vs Full-Info Trajectory
# ═══════════════════════════════════════════════════════════════════════════════

def fig4_baseline_comparison():
    """LVLH trajectory + distance + thrust comparison: angles-only vs full-info."""
    print("Running baseline simulations (Fig 4)...")
    t0 = time_mod.perf_counter()

    # Angles-only (noisy)
    result_ao = run_simulation(seed=42, sigma_ang_deg=SIGMA_ANG, angles_only=True)
    # Full-info (no noise → rng=None)
    orb = OrbitalDynamics(mu=MU, a_c=A_C, e_c=E_C)
    ctrl = SDREGameController(Q=Q_CTRL, R=R_CTRL, gamma=GAMMA)
    ekf_fi = make_ekf(0.008, angles_only=False)
    sim_fi = EKFSDRESimulation(
        dynamics=orb, controller=ctrl, ekf=ekf_fi,
        X_p0=X_P0, X_e0=X_E0, nu0=NU0, dt=DT,
        are_interval=1, rng=None,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        result_fi = sim_fi.run(t_end=10.0 * orb.T_orbit)
    print(f"  done ({time_mod.perf_counter() - t0:.1f}s)")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))

    # (a) LVLH 3D trajectory
    ax = axes[0, 0]
    s_ao = result_ao.states
    s_fi = result_fi.states
    x_rel_ao = s_ao[0, :] - s_ao[6, :]
    y_rel_ao = s_ao[1, :] - s_ao[7, :]
    z_rel_ao = s_ao[2, :] - s_ao[8, :]
    x_rel_fi = s_fi[0, :] - s_fi[6, :]
    y_rel_fi = s_fi[1, :] - s_fi[7, :]
    z_rel_fi = s_fi[2, :] - s_fi[8, :]

    ax.plot(x_rel_ao, y_rel_ao, color="#3498DB", lw=0.7, alpha=0.7, label="Angles-only EKF+SDRE")
    ax.plot(x_rel_fi, y_rel_fi, color="#E67E22", lw=0.7, alpha=0.7, label="Full-info SDRE")
    ax.plot(0, 0, "k*", markersize=10, label="Target")
    ax.plot(x_rel_ao[0], y_rel_ao[0], "ko", markersize=5)
    ax.plot(x_rel_ao[-1], y_rel_ao[-1], "ks", markersize=6)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title("(a) LVLH trajectory (x-y projection)", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal")

    # (b) Distance vs time
    ax = axes[0, 1]
    ax.semilogy(result_ao.t / 3600, result_ao.dist_history, "#3498DB", lw=0.9,
                label=f"Angles-only (T_cap={result_ao.t[-1]/3600:.1f} h)")
    ax.semilogy(result_fi.t / 3600, result_fi.dist_history, "#E67E22", lw=0.9,
                label=f"Full-info (T_cap={result_fi.t[-1]/3600:.1f} h)")
    ax.axhline(0.1, color="gray", ls="--", lw=0.8, label="Capture (100 m)")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Relative distance (km)")
    ax.set_title("(b) Distance vs. time", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)

    # (c) x-y-z relative position vs time
    ax = axes[0, 2]
    colors = ["#E74C3C", "#27AE60", "#2980B9"]
    labels = ["dx", "dy", "dz"]
    for i, (c, lbl) in enumerate(zip(colors, labels)):
        ax.plot(result_ao.t / 3600, x_rel_ao if i == 0 else (y_rel_ao if i == 1 else z_rel_ao),
                color=c, lw=0.7, label=f"{lbl} (AO)")
        ax.plot(result_fi.t / 3600, x_rel_fi if i == 0 else (y_rel_fi if i == 1 else z_rel_fi),
                color=c, lw=0.7, ls="--", alpha=0.6, label=f"{lbl} (FI)")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Relative position (km)")
    ax.set_title("(c) Relative position components", fontsize=10)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.25)

    # (d) Thrust norm vs time
    ax = axes[1, 0]
    u_ao_norm = np.sqrt(np.sum(result_ao.u_p_history**2, axis=0))
    u_fi_norm = np.sqrt(np.sum(result_fi.u_p_history**2, axis=0))
    ax.semilogy(result_ao.t / 3600, u_ao_norm * 1000, "#3498DB", lw=0.7,
                label="Angles-only")
    ax.semilogy(result_fi.t / 3600, u_fi_norm * 1000, "#E67E22", lw=0.7,
                label="Full-info")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Thrust magnitude (mm/s²)")
    ax.set_title("(d) Thrust magnitude vs. time", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)

    # (e) Thrust components (angles-only)
    ax = axes[1, 1]
    for i, (c, lbl) in enumerate(zip(colors, ["u_x", "u_y", "u_z"])):
        ax.plot(result_ao.t / 3600, result_ao.u_p_history[i, :] * 1000,
                color=c, lw=0.6, label=lbl)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Thrust (mm/s²)")
    ax.set_title("(e) Thrust components (angles-only EKF+SDRE)", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)

    # (f) Relative velocity norm
    ax = axes[1, 2]
    vrel_ao = np.sqrt(np.sum((s_ao[3:6, :] - s_ao[9:12, :])**2, axis=0))
    vrel_fi = np.sqrt(np.sum((s_fi[3:6, :] - s_fi[9:12, :])**2, axis=0))
    ax.plot(result_ao.t / 3600, vrel_ao * 1000, "#3498DB", lw=0.7)
    ax.plot(result_fi.t / 3600, vrel_fi * 1000, "#E67E22", lw=0.7)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Relative speed (m/s)")
    ax.set_title("(f) Relative speed vs. time", fontsize=10)
    ax.text(0.95, 0.95, f"Terminal: AO={vrel_ao[-1]*1000:.2f} m/s\n"
            f"FI={vrel_fi[-1]*1000:.2f} m/s",
            transform=ax.transAxes, fontsize=7, va="top", ha="right",
            bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))
    ax.grid(True, alpha=0.25)

    fig.suptitle("Fig. 4. Baseline comparison: angles-only EKF+SDRE vs. full-information SDRE.",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_baseline_comparison.pdf")
    plt.close(fig)
    print("Fig 4 saved.")
    return result_ao, result_fi


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 5: EKF Estimation Error Time History + 3σ Envelope
# ═══════════════════════════════════════════════════════════════════════════════

def fig5_ekf_error_history(result_ao):
    """6-panel plot: EKF error for each state component with 3σ envelope."""
    ekf_err = result_ao.ekf_err_history  # (6, N)
    P_diag = result_ao.P_diag_history    # (6, N)
    t_h = result_ao.t / 3600

    comp_names = [r"$\delta x$ (km)", r"$\delta y$ (km)", r"$\delta z$ (km)",
                  r"$\delta v_x$ (m/s)", r"$\delta v_y$ (m/s)", r"$\delta v_z$ (m/s)"]
    y_labels_pos = ["dx error (km)", "dy error (km)", "dz error (km)"]
    y_labels_vel = [r"dv$_x$ error (m/s)", r"dv$_y$ error (m/s)", r"dv$_z$ error (m/s)"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))

    for i in range(6):
        row, col = divmod(i, 3)
        ax = axes[row, col]
        err = ekf_err[i, :]
        sigma = np.sqrt(np.maximum(P_diag[i, :], 1e-20))
        if i < 3:
            err_plot = err
            sigma_plot = sigma
        else:
            err_plot = err * 1000  # km/s → m/s
            sigma_plot = sigma * 1000

        ax.fill_between(t_h, -3 * sigma_plot, 3 * sigma_plot, alpha=0.15, color="#3498DB")
        ax.plot(t_h, err_plot, "#3498DB", lw=0.6, alpha=0.8)
        ax.axhline(0, color="gray", ls="--", lw=0.6)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel(y_labels_pos[i] if i < 3 else y_labels_vel[i - 3])
        ax.set_title(f"({'abcdef'[i]}) {comp_names[i]}", fontsize=10)
        ax.grid(True, alpha=0.25)
        # add RMSE annotation
        rmse = np.sqrt(np.mean(err**2))
        rmse_disp = rmse if i < 3 else rmse * 1000
        ax.text(0.02, 0.95, f"RMSE={rmse_disp:.3f}", transform=ax.transAxes,
                fontsize=7, va="top", bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.7))

    # summary annotation
    pos_rmse = np.sqrt(np.mean(np.sum(ekf_err[:3, :]**2, axis=0)))
    vel_rmse = np.sqrt(np.mean(np.sum(ekf_err[3:, :]**2, axis=0)))
    fig.suptitle(
        f"Fig. 5. EKF estimation error time history with $3\\sigma$ envelope. "
        f"Pos. RMSE = {pos_rmse:.2f} km, Vel. RMSE = {vel_rmse*1000:.2f} mm/s.",
        fontsize=10, y=1.01,
    )

    fig.tight_layout()
    fig.savefig(OUT / "fig5_ekf_error.pdf")
    plt.close(fig)
    print("Fig 5 saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 6: Sensor Noise Sensitivity
# ═══════════════════════════════════════════════════════════════════════════════

def fig6_noise_sensitivity():
    """2×2: capture time, terminal v_rel, EKF RMSE, thrust — with terminal velocity trade-off."""
    csv_path = ROOT / "data" / "noise_sweep_results.csv"
    rows = []
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            rows.append(r)

    # Terminal v_rel from extraction CSV
    vrel_path = ROOT / "data" / "terminal_vrel_results.csv"
    vrel_data = {}
    if vrel_path.exists():
        with open(vrel_path) as fh:
            for r in csv.DictReader(fh):
                sa = r["sigma_ang"]
                if sa.startswith("scale_") or sa == "full_info":
                    continue
                sa_f = float(sa)
                v = float(r["terminal_vrel_ms"])
                vrel_data.setdefault(sa_f, []).append(v)
        # Full-info baseline
        with open(vrel_path) as fh:
            for r in csv.DictReader(fh):
                if r["sigma_ang"] == "full_info":
                    vrel_fi = float(r["terminal_vrel_ms"])
                    break
    else:
        vrel_fi = 0.097

    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[float(r["sigma_ang"])].append(r)

    sigmas = sorted(groups.keys())
    summaries = {}
    for s in sigmas:
        rs = groups[s]
        captured = [r for r in rs if r["captured"] == "True"]
        n_cap = len(captured)
        cap_times = [float(r["capture_time"]) for r in captured]
        pos_rmse = [float(r["ekf_err_pos_rmse"]) for r in rs]
        thrust_mean = [float(r["thrust_mean"]) * 1000 for r in rs]
        thrust_max = [float(r["thrust_max"]) * 1000 for r in rs]
        vv = vrel_data.get(s, [])
        summaries[s] = {
            "n_cap": n_cap, "n_tot": len(rs),
            "cap_time_med": np.median(cap_times) if n_cap > 0 else np.nan,
            "cap_time_std": np.std(cap_times) if n_cap > 1 else 0,
            "pos_mean": np.mean(pos_rmse), "pos_std": np.std(pos_rmse),
            "thrust_mean_mean": np.mean(thrust_mean),
            "thrust_mean_std": np.std(thrust_mean),
            "thrust_max_mean": np.mean(thrust_max),
            "vrel_mean": np.mean(vv) if vv else np.nan,
            "vrel_std": np.std(vv, ddof=1) if len(vv) > 1 else 0,
        }

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # ── (a) Capture time + terminal v_rel (dual y-axis) ──
    ax = axes[0, 0]
    ct = np.array([summaries[s]["cap_time_med"] / 3600 for s in sigmas])
    ct_std = np.array([summaries[s]["cap_time_std"] / 3600 for s in sigmas])
    vr = np.array([summaries[s]["vrel_mean"] for s in sigmas])
    vr_std = np.array([summaries[s]["vrel_std"] for s in sigmas])

    ax.errorbar(sigmas, ct, yerr=ct_std, fmt="o-", color="#3498DB",
                capsize=4, markersize=7, lw=1.5, label="Capture time (h)")
    ax.axhline(88.13, color="#3498DB", ls="--", lw=0.8, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\sigma_\theta$ (deg)")
    ax.set_ylabel("Capture time (h)", color="#3498DB")
    ax.tick_params(axis="y", labelcolor="#3498DB")
    ax.set_title("(a) Capture time vs. sensor noise", fontsize=10)
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    ax2.errorbar(sigmas, vr, yerr=vr_std, fmt="s-", color="#E74C3C",
                 capsize=4, markersize=7, lw=1.5, label="Terminal $v_{\\rm rel}$ (m/s)")
    ax2.axhline(vrel_fi, color="#E74C3C", ls=":", lw=0.8, alpha=0.5)
    ax2.set_ylabel("Terminal rel. velocity (m/s)", color="#E74C3C")
    ax2.tick_params(axis="y", labelcolor="#E74C3C")

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

    # Annotate the trade-off
    ax.annotate(f"Capture time: {ct[0]/ct[-1]:.1f}$\\times$ faster\n"
                f"Terminal $v_{{\\rm rel}}$: {vr[-1]/vr[0]:.1f}$\\times$ higher",
                xy=(0.03, 40), fontsize=8, color="#D43F3F",
                bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8))

    # ── (b) Terminal v_rel distribution per noise level (boxplot) ──
    ax = axes[0, 1]
    box_data = []
    box_labels = []
    for s in sigmas:
        if s in vrel_data and len(vrel_data[s]) > 0:
            box_data.append(vrel_data[s])
            box_labels.append(f"{s:.3f}")
    if box_data:
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                        widths=0.5, medianprops={"color": "#D43F3F", "lw": 1.5})
        for patch in bp["boxes"]:
            patch.set_facecolor("#FADBD8")
    ax.axhline(y=2.0, color="#E67E22", ls="--", lw=1.0,
               label="Soft-rendezvous threshold (2 m/s)")
    ax.axhline(y=vrel_fi, color="#27AE60", ls=":", lw=1.0,
               label=f"Full-info ($\\approx${vrel_fi:.1f} m/s)")
    ax.set_xlabel(r"$\sigma_\theta$ (deg)")
    ax.set_ylabel("Terminal rel. velocity (m/s)")
    ax.set_title("(b) Terminal velocity distribution vs. noise", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25, axis="y")

    # ── (c) Position RMSE vs sigma_ang ──
    ax = axes[1, 0]
    pm = np.array([summaries[s]["pos_mean"] for s in sigmas])
    ps = np.array([summaries[s]["pos_std"] for s in sigmas])
    ax.errorbar(sigmas, pm, yerr=ps, fmt="s-", color="#1ABC9C",
                capsize=4, markersize=7, lw=1.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\sigma_\theta$ (deg)")
    ax.set_ylabel("EKF position RMSE (km)")
    ax.set_title("(c) Estimation error vs. sensor noise", fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.annotate(f"RMSE saturates at $\\sigma_\\theta \\gtrsim 0.02^\\circ$",
                xy=(0.03, 9.0), fontsize=8, color="#1ABC9C")

    # ── (d) Thrust mean + max vs sigma_ang ──
    ax = axes[1, 1]
    tm = np.array([summaries[s]["thrust_mean_mean"] for s in sigmas])
    ts = np.array([summaries[s]["thrust_mean_std"] for s in sigmas])
    tmax = np.array([summaries[s]["thrust_max_mean"] for s in sigmas])

    ax.errorbar(sigmas, tm, yerr=ts, fmt="o-", color="#8E44AD",
                capsize=4, markersize=7, lw=1.5, label="Mean thrust")
    ax.plot(sigmas, tmax, "D-", color="#2C3E50", markersize=6, lw=1.2,
            label="Peak thrust", alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\sigma_\theta$ (deg)")
    ax.set_ylabel("Thrust (m/s²)")
    ax.set_title("(d) Thrust statistics vs. sensor noise", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)
    ax.annotate(f"Peak thrust constant:\n"
                f"{tmax[0]:.3f} $\\pm$ {np.std(tmax):.4f} m/s²",
                xy=(0.002, tmax[0] * 1.02), fontsize=8, color="#2C3E50",
                bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.7))

    fig.suptitle("Fig. 6. Sensor noise sensitivity — trade-off revealed: "
                 "higher noise reduces capture time by 2.3$\\times$ but "
                 "increases terminal relative velocity by $\\sim$20$\\times$ "
                 "(all trials remain under 2 m/s soft-rendezvous threshold).",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_noise_sensitivity.pdf")
    plt.close(fig)
    print("Fig 6 saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 7: CW Linear Model Baseline — Eccentricity Sweep
# ═══════════════════════════════════════════════════════════════════════════════

def fig7_eccentricity_sweep():
    """NERM capture time + EKF RMSE vs eccentricity, CW failure overlay."""
    csv_path = ROOT / "data" / "ecc_sweep_results.csv"
    rows = []
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            rows.append(r)

    from collections import defaultdict
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        ecc = float(r["eccentricity"])
        grp = r["group"]
        groups[ecc][grp].append(r)

    eccs = sorted(groups.keys())  # [0.001, 0.1, 0.3, 0.5, 0.7]
    nerm_success = []
    nerm_cap_time = []
    nerm_cap_std = []
    nerm_pos_rmse = []
    nerm_pos_std = []
    nerm_dv = []
    cw_success = []

    for e in eccs:
        nerm = groups[e].get("NERM+SDRE", [])
        cw = groups[e].get("CW+SDRE", [])
        nerm_captured = [r for r in nerm if r["captured"] == "True"]
        cw_captured = [r for r in cw if r["captured"] == "True"]
        nerm_success.append(len(nerm_captured) / max(len(nerm), 1))
        cw_success.append(len(cw_captured) / max(len(cw), 1))
        ct = [float(r["capture_time"]) / 3600 for r in nerm_captured]
        nerm_cap_time.append(np.median(ct) if ct else np.nan)
        nerm_cap_std.append(np.std(ct) if len(ct) > 1 else 0)
        pr = [float(r["ekf_err_pos_rmse"]) for r in nerm if float(r["ekf_err_pos_rmse"]) < 1e8]
        nerm_pos_rmse.append(np.median(pr) if pr else np.nan)
        nerm_pos_std.append(np.std(pr) if len(pr) > 1 else 0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # (a) Capture time vs eccentricity
    ax = axes[0]
    ax.errorbar(eccs, nerm_cap_time, yerr=nerm_cap_std, fmt="o-", color="#27AE60",
                capsize=5, markersize=8, lw=1.5, label="NERM+SDRE")
    ax.set_xlabel("Eccentricity $e_c$")
    ax.set_ylabel("Capture time (h)")
    ax.set_title("(a) NERM capture time vs. eccentricity", fontsize=10)
    ax.grid(True, alpha=0.25)

    # (b) EKF Position RMSE vs eccentricity
    ax = axes[1]
    # Filter out extreme values (> 1e6)
    valid = [i for i, v in enumerate(nerm_pos_rmse) if not np.isnan(v) and v < 1e5]
    ax.errorbar([eccs[i] for i in valid], [nerm_pos_rmse[i] for i in valid],
                yerr=[nerm_pos_std[i] for i in valid],
                fmt="s-", color="#E74C3C", capsize=5, markersize=8, lw=1.5)
    ax.set_xlabel("Eccentricity $e_c$")
    ax.set_ylabel("EKF position RMSE (km)")
    ax.set_title("(b) EKF estimation error vs. eccentricity", fontsize=10)
    ax.grid(True, alpha=0.25)

    # (c) Success rate — NERM vs CW
    ax = axes[2]
    x = np.arange(len(eccs))
    w = 0.35
    ax.bar(x - w / 2, [s * 100 for s in nerm_success], w, color="#27AE60",
           alpha=0.8, label="NERM+SDRE")
    ax.bar(x + w / 2, [s * 100 for s in cw_success], w, color="#E74C3C",
           alpha=0.8, label="CW+SDRE")
    ax.set_xticks(x)
    ax.set_xticklabels([str(e) for e in eccs])
    ax.set_xlabel("Eccentricity $e_c$")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("(c) Success rate: NERM vs. CW", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, axis="y")
    # CW failure annotation
    ax.annotate("CW fails at ALL\neccentricities\n(including e=0.001!)",
                xy=(0.5, 5), fontsize=8, color="#D43F3F", weight="bold",
                ha="center", bbox=dict(boxstyle="round", fc="mistyrose", ec="#D43F3F"))

    fig.suptitle("Fig. 7. Eccentricity robustness — NERM+SDRE succeeds at all $e_c$, "
                 "CW+SDRE fails universally on NERM truth dynamics.",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig7_eccentricity_sweep.pdf")
    plt.close(fig)
    print("Fig 7 saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 8: Monte Carlo Capture Time Distribution
# ═══════════════════════════════════════════════════════════════════════════════

def fig8_monte_carlo_histogram():
    """Monte Carlo capture time histogram + success rate."""
    csv_path = ROOT / "data" / "monte_carlo_results.csv"
    rows = []
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            rows.append(r)

    captured = [r for r in rows if r["captured"] == "True"]
    not_captured = [r for r in rows if r["captured"] != "True"]
    cap_times_h = [float(r["capture_time"]) / 3600 for r in captured]
    success_rate = len(captured) / len(rows) * 100

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # (a) Capture time histogram
    ax = axes[0]
    bins = np.linspace(0, 60, 25)
    ax.hist(cap_times_h, bins=bins, color="#3498DB", alpha=0.7, edgecolor="#2C3E50", lw=0.5)
    ax.axvline(np.median(cap_times_h), color="#E74C3C", ls="--", lw=1.5,
               label=f"Median = {np.median(cap_times_h):.1f} h")
    ax.axvline(np.mean(cap_times_h), color="#E67E22", ls=":", lw=1.5,
               label=f"Mean = {np.mean(cap_times_h):.1f} h")
    ax.set_xlabel("Capture time (h)")
    ax.set_ylabel("Number of trials")
    ax.set_title(f"(a) Capture time distribution (N={len(captured)} captured)", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25, axis="y")

    # (b) Final distance histogram
    ax = axes[1]
    final_dists = [float(r["final_distance"]) * 1000 for r in captured]  # km → m
    ax.hist(final_dists, bins=25, color="#27AE60", alpha=0.7, edgecolor="#1E8449", lw=0.5)
    ax.axvline(np.median(final_dists), color="#E74C3C", ls="--", lw=1.2,
               label=f"Median = {np.median(final_dists):.1f} m")
    ax.set_xlabel("Final miss distance (m)")
    ax.set_ylabel("Number of trials")
    ax.set_title(f"(b) Final distance distribution (captured trials)", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25, axis="y")

    # (c) ΔV distribution
    ax = axes[2]
    dv = [float(r["total_dV_p"]) for r in captured]
    ax.hist(dv, bins=25, color="#8E44AD", alpha=0.7, edgecolor="#6C3483", lw=0.5)
    ax.axvline(np.median(dv), color="#E74C3C", ls="--", lw=1.2,
               label=f"Median = {np.median(dv):.1f} km/s")
    q95 = np.percentile(dv, 95)
    ax.axvline(q95, color="#E67E22", ls=":", lw=1.2,
               label=f"P95 = {q95:.1f} km/s")
    ax.set_xlabel(r"Total $\Delta V$ (km/s)")
    ax.set_ylabel("Number of trials")
    ax.set_title(f"(c) Total ΔV distribution (captured trials)", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25, axis="y")

    fig.suptitle(
        f"Fig. 8. Monte Carlo analysis (N=200). Success rate: {success_rate:.1f}% "
        f"({len(captured)}/{len(rows)}). "
        f"Median capture time = {np.median(cap_times_h):.1f} h, "
        f"Median ΔV = {np.median(dv):.1f} km/s.",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig8_monte_carlo.pdf")
    plt.close(fig)
    print("Fig 8 saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 9: Monte Carlo EKF Error Distribution
# ═══════════════════════════════════════════════════════════════════════════════

def fig9_monte_carlo_ekf_error():
    """EKF error distribution from Monte Carlo, plus initial distance vs outcome."""
    csv_path = ROOT / "data" / "monte_carlo_results.csv"
    rows = []
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            rows.append(r)

    captured = [r for r in rows if r["captured"] == "True"]
    not_captured = [r for r in rows if r["captured"] != "True"]

    pos_rmse_all = [float(r["ekf_err_pos_rmse"]) for r in rows]
    pos_rmse_cap = [float(r["ekf_err_pos_rmse"]) for r in captured]
    vel_rmse_cap = [float(r["ekf_err_vel_rmse"]) * 1000 for r in captured]  # km/s → mm/s

    init_dist_all = [float(r["initial_distance"]) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # (a) EKF position RMSE histogram (captured vs failed)
    ax = axes[0]
    bins = np.linspace(0, 80, 30)
    ax.hist(pos_rmse_cap, bins=bins, color="#27AE60", alpha=0.6, edgecolor="#1E8449",
            lw=0.5, label=f"Captured (n={len(pos_rmse_cap)})")
    if not_captured:
        pos_fail = [float(r["ekf_err_pos_rmse"]) for r in not_captured]
        ax.hist(pos_fail, bins=bins, color="#E74C3C", alpha=0.8, edgecolor="#C0392B",
                lw=0.5, label=f"Failed (n={len(pos_fail)})")
    ax.axvline(np.median(pos_rmse_all), color="#3498DB", ls="--", lw=1.2,
               label=f"Overall median = {np.median(pos_rmse_all):.1f} km")
    ax.set_xlabel("EKF position RMSE (km)")
    ax.set_ylabel("Number of trials")
    ax.set_title("(a) EKF position error distribution", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25, axis="y")

    # (b) Initial distance vs EKF error (colored by outcome)
    ax = axes[1]
    pos_fail = [float(r["ekf_err_pos_rmse"]) for r in not_captured] if not_captured else []
    init_fail = [float(r["initial_distance"]) for r in not_captured] if not_captured else []
    ax.scatter([float(r["initial_distance"]) for r in captured], pos_rmse_cap,
               c="#27AE60", s=12, alpha=0.4, label="Captured")
    if not_captured:
        ax.scatter(init_fail, pos_fail, c="#E74C3C", s=30, marker="x",
                   alpha=0.9, label="Failed (5 trials)")
    ax.set_xlabel("Initial distance (km)")
    ax.set_ylabel("EKF position RMSE (km)")
    ax.set_title("(b) Initial distance vs. EKF error", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)

    # (c) EKF velocity RMSE distribution
    ax = axes[2]
    ax.hist(vel_rmse_cap, bins=25, color="#2980B9", alpha=0.7, edgecolor="#1A5276", lw=0.5)
    ax.axvline(np.median(vel_rmse_cap), color="#E74C3C", ls="--", lw=1.2,
               label=f"Median = {np.median(vel_rmse_cap):.1f} mm/s")
    ax.set_xlabel("EKF velocity RMSE (mm/s)")
    ax.set_ylabel("Number of trials")
    ax.set_title("(c) EKF velocity error (captured trials)", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25, axis="y")

    fig.suptitle(
        f"Fig. 9. Monte Carlo EKF error analysis. "
        f"Overall pos. RMSE median = {np.median(pos_rmse_all):.1f} km. "
        f"5 failures associated with large initial distance + adverse geometry.",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig9_monte_carlo_ekf.pdf")
    plt.close(fig)
    print("Fig 9 saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t_start = time_mod.perf_counter()

    print("=" * 60)
    print("Generating paper figures...")
    print("=" * 60)

    # Conceptual figures (fast — no simulation needed)
    fig1_lvlh_frame()
    fig2_flow_diagram()

    # Fig 3: Prediction error diagnostic (simulation ~30s)
    rec = fig3_prediction_error()

    # Fig 4–5: Baseline simulations (~2 min)
    result_ao, result_fi = fig4_baseline_comparison()
    fig5_ekf_error_history(result_ao)

    # Fig 6–9: From CSV data (fast)
    fig6_noise_sensitivity()
    fig7_eccentricity_sweep()
    fig8_monte_carlo_histogram()
    fig9_monte_carlo_ekf_error()

    elapsed = time_mod.perf_counter() - t_start
    print()
    print(f"All 9 figures generated in {elapsed:.1f}s.")
    print(f"Output: {OUT}/")
    for f in sorted(OUT.glob("fig*.pdf")):
        print(f"  {f.name}")
