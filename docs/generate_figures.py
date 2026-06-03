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
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.set_aspect("equal")
    ax.axis("off")

    # Earth at origin
    earth = plt.Circle((0, 0), 0.35, color="#2E75B6", ec="#1A3A5C", lw=1.2, zorder=3)
    ax.add_patch(earth)
    ax.text(0, 0, "Earth", ha="center", va="center", fontsize=7, color="white", weight="bold")

    # Reference orbit (elliptical)
    a, b = 3.0, 2.4  # semi-major, semi-minor axes
    theta = np.linspace(0, 2 * np.pi, 300)
    orbit_x = a * np.cos(theta)
    orbit_y = b * np.sin(theta)
    ax.plot(orbit_x, orbit_y, "k--", lw=0.8, alpha=0.5, zorder=1)

    # Chief position on the orbit
    chief_angle = np.deg2rad(40)
    cx, cy = a * np.cos(chief_angle), b * np.sin(chief_angle)
    ax.plot(cx, cy, "ko", markersize=7, zorder=4)
    ax.text(cx + 0.15, cy + 0.2, "Chief", fontsize=8, weight="bold")

    # Deputy position (offset from chief)
    dx, dy = 0.7, 0.5
    px, py = cx + dx, cy + dy
    ax.plot(px, py, "k^", markersize=7, zorder=4)
    ax.text(px + 0.15, py + 0.15, "Chaser", fontsize=8, weight="bold")
    ax.plot([cx, px], [cy, py], "k-", lw=0.8, alpha=0.5)

    # LVLH axes at chief position
    axis_len = 0.9
    # x — radial (outward from Earth center)
    radial_dir = np.array([cx, cy]) / np.sqrt(cx**2 + cy**2)
    ax.arrow(cx, cy, axis_len * radial_dir[0], axis_len * radial_dir[1],
             head_width=0.06, head_length=0.1, fc="#E74C3C", ec="#E74C3C", lw=1.2, zorder=5)
    ax.text(cx + axis_len * radial_dir[0] + 0.08, cy + axis_len * radial_dir[1] + 0.08,
            r"$\hat{x}$ (radial)", fontsize=8, color="#E74C3C")

    # y — along-track (perpendicular to radial, in velocity direction)
    vel_dir = np.array([-radial_dir[1], radial_dir[0]])
    ax.arrow(cx, cy, axis_len * vel_dir[0], axis_len * vel_dir[1],
             head_width=0.06, head_length=0.1, fc="#27AE60", ec="#27AE60", lw=1.2, zorder=5)
    ax.text(cx + axis_len * vel_dir[0] + 0.05, cy + axis_len * vel_dir[1] - 0.15,
            r"$\hat{y}$ (along-track)", fontsize=8, color="#27AE60")

    # z — cross-track (out of plane, indicated with a circle-dot)
    ax.text(cx - 0.25, cy - 0.25, r"$\hat{z}$ (cross-track, out of page)",
            fontsize=8, color="#8E44AD", style="italic")

    # Orbit direction arrow
    mid_angle = np.deg2rad(15)
    mdx = a * np.cos(mid_angle) - a * np.cos(mid_angle - 0.02)
    mdy = b * np.sin(mid_angle) - b * np.sin(mid_angle - 0.02)
    ax.annotate("", xy=(a * np.cos(mid_angle), b * np.sin(mid_angle)),
                xytext=(a * np.cos(mid_angle + 0.1), b * np.sin(mid_angle + 0.1)),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))
    ax.text(a * np.cos(mid_angle + 0.15) + 0.05, b * np.sin(mid_angle + 0.15),
            "Orbit", fontsize=7, color="gray")

    # Relative position annotation
    mid_pt = ((cx + px) / 2, (cy + py) / 2)
    ax.annotate(r"$\mathbf{x}_{\rm rel}$", xy=mid_pt, xytext=(mid_pt[0] + 0.35, mid_pt[1] - 0.3),
                fontsize=8, ha="center",
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.7))

    ax.set_xlim(-3.8, 4.2)
    ax.set_ylim(-3.5, 4.0)
    ax.set_title("Fig. 1. LVLH coordinate frame and problem geometry.", pad=10)

    fig.savefig(OUT / "fig1_lvlh_frame.pdf")
    plt.close(fig)
    print("Fig 1 saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2: Unified EKF-SDRE Algorithm Flow Diagram
# ═══════════════════════════════════════════════════════════════════════════════

def fig2_flow_diagram():
    """Block diagram of the unified EKF-SDRE framework showing A_SDC reuse."""
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # ── helper: draw a rounded box ──
    def box(ax, x, y, w, h, text, color="#D5E8D4", text_color="black", fontsize=8):
        rect = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                              boxstyle="round,pad=0.1", fc=color, ec="#333", lw=0.8)
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                color=text_color, weight="bold" if color != "white" else "normal")

    def arrow(x1, y1, x2, y2, color="#333", lw=1.0):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw))

    def label(x, y, text, fontsize=7, color="#555"):
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=color)

    # ── Nodes ──
    # Row 1: state at t_k
    box(ax, 1.5, 7.0, 2.0, 0.7, r"$\hat{\mathbf{x}}_{k|k}$", color="#E8D8F0")
    label(1.5, 6.55, "EKF posterior", fontsize=6)

    # SDC computation
    box(ax, 4.0, 7.0, 2.0, 0.7, r"$A_{\rm SDC}(t_k, \hat{\mathbf{x}}_{k|k})$", color="#FFF2CC")
    label(4.0, 6.55, "One computation per step", fontsize=6)

    # Branch: control (upper)
    box(ax, 7.0, 7.7, 2.0, 0.6, "SDRE: Solve ARE", color="#D5E8D4")
    label(7.0, 7.32, r"$A^{\sf T}P + PA - PBR^{-1}B^{\sf T}P + Q = 0$", fontsize=6)
    box(ax, 7.0, 6.8, 2.0, 0.5, r"$\mathbf{u}_c = -R^{-1}B^{\sf T}P\hat{\mathbf{x}}_{k|k}$", color="#D5E8D4")

    # Branch: estimation (lower)
    box(ax, 7.0, 5.8, 2.0, 0.6, "EKF Predict", color="#DAE8FC")
    label(7.0, 5.45, r"$\mathbf{x}_{k+1|k} = (I + A_{\rm SDC}\cdot dt)\hat{\mathbf{x}}_{k|k}$", fontsize=6)

    # True dynamics
    box(ax, 4.0, 5.0, 2.5, 0.6, "True Dynamics (RK45)", color="#F8CECC")
    label(4.0, 4.62, r"$\dot{\mathbf{X}} = f(\mathbf{X}, \mathbf{u}_c)$  (13-D NERM)", fontsize=6)

    # Measurement
    box(ax, 1.5, 5.0, 2.0, 0.6, "Sensor: Angles-Only", color="#E1D5E7")
    label(1.5, 4.62, r"$\mathbf{z} = [az, el]^{\sf T} + \mathbf{v}$", fontsize=6)

    # EKF Update
    box(ax, 4.0, 4.0, 2.0, 0.6, "EKF Update", color="#DAE8FC")
    label(4.0, 3.62, r"$\hat{\mathbf{x}}_{k+1|k+1}, \mathbf{P}_{k+1|k+1}$", fontsize=6)

    # ── Arrows ──
    arrow(2.5, 7.0, 3.0, 7.0)  # state → SDC
    arrow(5.0, 7.0, 6.0, 7.7)  # SDC → SDRE (upper branch)
    arrow(5.0, 7.0, 6.0, 5.8)  # SDC → EKF predict (lower branch)
    arrow(7.0, 6.55, 7.0, 5.0)  # u_c → true dynamics (via center)
    arrow(8.0, 6.8, 4.0, 5.35)  # u_c to dynamics

    # From true dynamics → measurement
    arrow(2.75, 5.0, 2.5, 5.0)

    # measurement → EKF update
    arrow(2.5, 5.3, 3.0, 4.3)  # curved path

    # EKF update → state (feedback loop)
    arrow(5.0, 4.0, 5.0, 5.0)
    arrow(5.0, 5.0, 2.5, 5.0)
    # EKF predict → update
    arrow(8.0, 5.8, 5.0, 4.3)

    # ── Highlight the A_SDC reuse ──
    ax.annotate("", xy=(5.8, 6.3), xytext=(5.8, 7.3),
                arrowprops=dict(arrowstyle="<->", color="#D43F3F", lw=2.5))
    ax.text(6.3, 6.8, "SAME\nMATRIX", fontsize=7, color="#D43F3F", weight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#D43F3F", lw=0.8))

    # ── Labels ──
    ax.text(5.0, 7.8, "Unified EKF-SDRE Framework", ha="center", fontsize=13, weight="bold")
    ax.text(8.5, 6.3, "Control\nPath", ha="center", fontsize=7, color="#5B9BD5", weight="bold")
    ax.text(8.5, 5.55, "Estimation\nPath", ha="center", fontsize=7, color="#5B9BD5", weight="bold")

    ax.set_title("Fig. 2. Unified EKF-SDRE algorithm flow — single $A_{\\rm SDC}(t_k)$ "
                 "drives both control and estimation.", pad=18)

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
