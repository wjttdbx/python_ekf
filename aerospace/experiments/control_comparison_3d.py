"""
3D 控制效果综合对比实验
========================

四组实验:
  1. 静态精度: 万点 ΔP / Δu 统计
  2. 标准轨迹闭环: 单条深度分析 + 画图
  3. 蒙特卡洛鲁棒性: 多组统计
  4. 极端条件边界测试

运行: uv run python -m aerospace.experiments.control_comparison_3d
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
try:
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: F401
except Exception:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import LinAlgError, solve_continuous_are

from aerospace.dynamics.nerm import OrbitalDynamics
from aerospace.control.sdre import SDREGameController
from aerospace.control.neural import NeuralSDREController
from aerospace.simulation.nerm_sdre import SDRESimulation, SDRESimResult
from aerospace.pinn.data_generator import (
    STATE_DIM, CTRL_DIM, FEAT_DIM, extract_asdc_features, _solve_are_balanced,
)

CHECKPOINT = "checkpoints/sdre_pinn/ctrl_loss05_phys_loss20/best_model.pt"
OUTPUT_DIR = Path("outputs/ex_2/")

MU = 3.986e5
A_C = 6771
E_C = 0
Q = np.eye(STATE_DIM)
R = np.eye(CTRL_DIM) * 1e13
GAMMA = np.sqrt(2)
R_EFF = R / (1.0 - GAMMA ** (-2))
R_INV = np.linalg.inv(R)
B_P = np.zeros((STATE_DIM, CTRL_DIM))
B_P[3:, :] = np.eye(CTRL_DIM)
B_E = -B_P


def _compute_u(P, x_rel):
    u_p = -R_INV @ B_P.T @ P @ x_rel
    u_e = GAMMA ** (-2) * R_INV @ B_E.T @ P @ x_rel
    return u_p, u_e


# ═══════════════════════════════════════════════════════════════════
#  Experiment 1: Static Accuracy
# ═══════════════════════════════════════════════════════════════════

@dataclass
class StaticAccuracyResult:
    n_samples: int
    n_valid: int
    n_pinn_ood: int
    n_are_fail: int
    dp_fro_rel: np.ndarray
    du_p_rel: np.ndarray
    du_e_rel: np.ndarray
    dJ_rel: np.ndarray


def experiment_1_static_accuracy(n_samples: int = 10000, seed: int = 42) -> StaticAccuracyResult:
    print("=" * 60)
    print(f"  Experiment 1: Static Accuracy 3D ({n_samples} samples)")
    print("=" * 60)

    dyn = OrbitalDynamics(MU, A_C, E_C)
    pinn = NeuralSDREController(CHECKPOINT, device="cpu")

    rng = np.random.default_rng(seed)
    pos_range, vel_range = 1000.0, 0.1

    dp_fro, du_p, du_e, dj = [], [], [], []
    n_valid, n_ood, n_are_fail = 0, 0, 0

    import torch
    from aerospace.pinn.pinn_trainer import reconstruct_spd_p

    for i in range(n_samples):
        Xp = np.zeros(6); Xe = np.zeros(6)
        Xp[:3] = rng.uniform(-pos_range, pos_range, 3)
        Xp[3:] = rng.uniform(-vel_range, vel_range, 3)
        Xe[:3] = rng.uniform(-pos_range, pos_range, 3)
        Xe[3:] = rng.uniform(-vel_range, vel_range, 3)
        nu = rng.uniform(0, 2 * np.pi)

        rc, nd, ndd = dyn.get_orbital_params(nu)
        A_SDC = dyn.get_SDC_matrix(Xp, Xe, rc, nd, ndd)
        x_rel = Xp - Xe

        try:
            P_exact = _solve_are_balanced(A_SDC, B_P, Q, R_EFF)
        except (LinAlgError, ValueError):
            n_are_fail += 1
            continue

        feat = extract_asdc_features(A_SDC)
        feat_z = np.abs((feat - pinn.feat_mean) / pinn.feat_std)
        if np.any(feat_z > pinn.ood_zscore_threshold):
            n_ood += 1
            continue

        feat_norm = (feat - pinn.feat_mean) / pinn.feat_std
        feat_t = torch.from_numpy(feat_norm.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            l_norm = pinn.model(feat_t)
            l_pred = l_norm * pinn.l_std_t + pinn.l_mean_t
            P_pinn = reconstruct_spd_p(l_pred, pinn.delta_spd).squeeze(0).numpy().astype(np.float64)

        n_valid += 1
        norm_P = np.linalg.norm(P_exact, "fro")
        if norm_P < 1e-30:
            continue
        dp_fro.append(np.linalg.norm(P_pinn - P_exact, "fro") / norm_P)

        u_p_exact, u_e_exact = _compute_u(P_exact, x_rel)
        u_p_pinn, u_e_pinn = _compute_u(P_pinn, x_rel)

        norm_up = np.linalg.norm(u_p_exact)
        norm_ue = np.linalg.norm(u_e_exact)
        if norm_up > 1e-30:
            du_p.append(np.linalg.norm(u_p_pinn - u_p_exact) / norm_up)
        if norm_ue > 1e-30:
            du_e.append(np.linalg.norm(u_e_pinn - u_e_exact) / norm_ue)

        xtPx = x_rel @ P_exact @ x_rel
        if abs(xtPx) > 1e-30:
            dj.append(abs(x_rel @ (P_pinn - P_exact) @ x_rel) / abs(xtPx))

        if (i + 1) % 2000 == 0:
            print(f"  ... {i+1}/{n_samples} processed, {n_valid} valid, {n_are_fail} ARE fail, {n_ood} OOD")

    result = StaticAccuracyResult(
        n_samples=n_samples, n_valid=n_valid, n_pinn_ood=n_ood, n_are_fail=n_are_fail,
        dp_fro_rel=np.array(dp_fro), du_p_rel=np.array(du_p),
        du_e_rel=np.array(du_e), dJ_rel=np.array(dj),
    )
    _print_static_stats(result)
    _plot_static_accuracy(result)
    return result


def _print_static_stats(r: StaticAccuracyResult):
    print(f"\n  Samples: {r.n_samples} total, {r.n_valid} valid, "
          f"{r.n_are_fail} ARE fail, {r.n_pinn_ood} OOD skipped")
    for name, arr in [("ΔP/P (Frobenius)", r.dp_fro_rel),
                      ("Δu_p/u_p", r.du_p_rel),
                      ("Δu_e/u_e", r.du_e_rel),
                      ("ΔJ/J (cost)", r.dJ_rel)]:
        if len(arr) == 0:
            print(f"  {name}: no data")
            continue
        print(f"  {name}:")
        print(f"    mean={np.mean(arr):.4e}  median={np.median(arr):.4e}  "
              f"std={np.std(arr):.4e}")
        print(f"    p50={np.percentile(arr,50):.4e}  p95={np.percentile(arr,95):.4e}  "
              f"p99={np.percentile(arr,99):.4e}  max={np.max(arr):.4e}")


def _plot_static_accuracy(r: StaticAccuracyResult):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Experiment 1: Static Accuracy 3D — PINN vs Exact CARE", fontsize=14)
    data_labels = [
        (r.dp_fro_rel, "Relative P Error (Frobenius)", "ΔP/P"),
        (r.du_p_rel, "Relative u_p Error", "Δu_p/u_p"),
        (r.du_e_rel, "Relative u_e Error", "Δu_e/u_e"),
        (r.dJ_rel, "Relative Cost Perturbation", "ΔJ/J"),
    ]
    for ax, (data, title, xlabel) in zip(axes.flat, data_labels):
        if len(data) == 0:
            ax.set_title(title + " (no data)"); continue
        ax.hist(data, bins=100, alpha=0.7, color="steelblue", edgecolor="white")
        ax.axvline(np.median(data), color="red", linestyle="--",
                   label=f"median={np.median(data):.2e}")
        ax.axvline(np.percentile(data, 95), color="orange", linestyle="--",
                   label=f"p95={np.percentile(data,95):.2e}")
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("Count")
        ax.legend(fontsize=8); ax.set_yscale("log")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "exp1_static_accuracy.png", dpi=150)
    print(f"  -> Saved: {OUTPUT_DIR / 'exp1_static_accuracy.png'}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Experiment 2: Standard Trajectory Closed-loop
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TrajectoryCompareResult:
    sdre_result: SDRESimResult
    pinn_result: SDRESimResult
    sdre_time_s: float
    pinn_time_s: float
    pinn_fallback_count: int
    pinn_total_calls: int


def _run_sim(dyn, ctrl, Xp0, Xe0, nu0, dt):
    sim = SDRESimulation(dyn, ctrl, Xp0, Xe0, nu0=nu0, dt=dt, log_interval=200)
    t0 = time.perf_counter()
    res = sim.run()
    elapsed = time.perf_counter() - t0
    return res, elapsed


def experiment_2_trajectory_compare(
    Xp0=np.array([500.0, 500.0, 500.0, 0.01, 0.01, 0.01]),
    Xe0=np.zeros(6),
    nu0=0.0, dt=20.0,
) -> TrajectoryCompareResult:
    print("\n" + "=" * 60)
    print("  Experiment 2: Trajectory Closed-loop 3D")
    print("=" * 60)

    dyn = OrbitalDynamics(MU, A_C, E_C)

    print("\n  --- Running SDRE baseline ---")
    sdre_ctrl = SDREGameController(Q, R, GAMMA)
    sdre_res, sdre_t = _run_sim(dyn, sdre_ctrl, Xp0, Xe0, nu0, dt)

    print("\n  --- Running PINN ---")
    pinn_ctrl = NeuralSDREController(CHECKPOINT, device="cpu")
    pinn_res, pinn_t = _run_sim(dyn, pinn_ctrl, Xp0, Xe0, nu0, dt)

    result = TrajectoryCompareResult(
        sdre_result=sdre_res, pinn_result=pinn_res,
        sdre_time_s=sdre_t, pinn_time_s=pinn_t,
        pinn_fallback_count=pinn_ctrl.fallback_calls,
        pinn_total_calls=pinn_ctrl.total_calls,
    )
    _print_trajectory_stats(result)
    _plot_trajectory_compare(result)
    return result


def _rel_dist(states):
    dx = states[0] - states[6]
    dy = states[1] - states[7]
    dz = states[2] - states[8]
    return np.sqrt(dx**2 + dy**2 + dz**2)


def _delta_v(u_hist, t):
    dt_arr = np.diff(t)
    n = min(u_hist.shape[1], len(dt_arr))
    mag = np.linalg.norm(u_hist[:, :n], axis=0)
    return np.cumsum(mag * dt_arr[:n])


def _print_trajectory_stats(r: TrajectoryCompareResult):
    s, p = r.sdre_result, r.pinn_result
    sd, pd = _rel_dist(s.states), _rel_dist(p.states)
    sdv_p, pdv_p = _delta_v(s.u_p_history, s.t), _delta_v(p.u_p_history, p.t)
    sdv_e, pdv_e = _delta_v(s.u_e_history, s.t), _delta_v(p.u_e_history, p.t)

    print(f"\n  {'Metric':<30s} {'SDRE':>12s} {'PINN':>12s} {'Diff':>12s}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'Final distance (km)':<30s} {sd[-1]:>12.3f} {pd[-1]:>12.3f} {pd[-1]-sd[-1]:>+12.3f}")
    print(f"  {'Min distance (km)':<30s} {sd.min():>12.3f} {pd.min():>12.3f} {pd.min()-sd.min():>+12.3f}")
    print(f"  {'Pursuer ΔV (km/s)':<30s} {sdv_p[-1]:>12.6f} {pdv_p[-1]:>12.6f} {pdv_p[-1]-sdv_p[-1]:>+12.6f}")
    print(f"  {'Evader ΔV (km/s)':<30s} {sdv_e[-1]:>12.6f} {pdv_e[-1]:>12.6f} {pdv_e[-1]-sdv_e[-1]:>+12.6f}")
    print(f"  {'Compute time (s)':<30s} {r.sdre_time_s:>12.2f} {r.pinn_time_s:>12.2f} {r.pinn_time_s/r.sdre_time_s:>11.2f}x")
    print(f"  {'PINN OOD fallbacks':<30s} {'':>12s} {r.pinn_fallback_count:>12d}")

    n = min(s.P_history.shape[0], p.P_history.shape[0])
    p_err = np.array([np.linalg.norm(p.P_history[i] - s.P_history[i], "fro") /
                       max(np.linalg.norm(s.P_history[i], "fro"), 1e-30) for i in range(n)])
    print(f"\n  P matrix trajectory error (PINN vs SDRE):")
    print(f"    mean={np.mean(p_err):.4e}  median={np.median(p_err):.4e}  "
          f"max={np.max(p_err):.4e}  std={np.std(p_err):.4e}")


def _plot_trajectory_compare(r: TrajectoryCompareResult):
    s, p = r.sdre_result, r.pinn_result
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle("Experiment 2: Closed-loop Trajectory 3D — SDRE vs PINN", fontsize=14)

    # 1) 3D trajectory (x-y-z)
    ax = fig.add_subplot(3, 2, 1, projection='3d')
    axes[0, 0].set_visible(False)
    sx = s.states[0] - s.states[6]; sy = s.states[1] - s.states[7]; sz = s.states[2] - s.states[8]
    px = p.states[0] - p.states[6]; py = p.states[1] - p.states[7]; pz = p.states[2] - p.states[8]
    ax.plot(sx, sy, sz, "b-", label="SDRE", alpha=0.8)
    ax.plot(px, py, pz, "r--", label="PINN", alpha=0.8)
    ax.scatter([0], [0], [0], c="k", s=50, marker="*", label="Target")
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)"); ax.set_zlabel("z (km)")
    ax.set_title("Relative Trajectory"); ax.legend(fontsize=7)

    # 2) Distance
    ax = axes[0, 1]
    sd, pd = _rel_dist(s.states), _rel_dist(p.states)
    ax.plot(s.t / 3600, sd, "b-", label="SDRE")
    ax.plot(p.t / 3600, pd, "r--", label="PINN")
    ax.set_xlabel("Time (hours)"); ax.set_ylabel("Distance (km)"); ax.set_title("Distance vs Time")
    ax.legend(); ax.grid(True, alpha=0.3)

    # 3) Pursuer thrust
    ax = axes[1, 0]
    su_p = np.linalg.norm(s.u_p_history, axis=0); pu_p = np.linalg.norm(p.u_p_history, axis=0)
    ax.plot(s.t[:len(su_p)] / 3600, su_p, "b-", label="SDRE", alpha=0.7)
    ax.plot(p.t[:len(pu_p)] / 3600, pu_p, "r--", label="PINN", alpha=0.7)
    ax.set_xlabel("Time (hours)"); ax.set_ylabel("|u_p| (km/s²)"); ax.set_title("Pursuer Thrust")
    ax.legend(); ax.grid(True, alpha=0.3)

    # 4) Evader thrust
    ax = axes[1, 1]
    su_e = np.linalg.norm(s.u_e_history, axis=0); pu_e = np.linalg.norm(p.u_e_history, axis=0)
    ax.plot(s.t[:len(su_e)] / 3600, su_e, "b-", label="SDRE", alpha=0.7)
    ax.plot(p.t[:len(pu_e)] / 3600, pu_e, "r--", label="PINN", alpha=0.7)
    ax.set_xlabel("Time (hours)"); ax.set_ylabel("|u_e| (km/s²)"); ax.set_title("Evader Thrust")
    ax.legend(); ax.grid(True, alpha=0.3)

    # 5) ΔV
    ax = axes[2, 0]
    sdv_p = _delta_v(s.u_p_history, s.t); pdv_p = _delta_v(p.u_p_history, p.t)
    sdv_e = _delta_v(s.u_e_history, s.t); pdv_e = _delta_v(p.u_e_history, p.t)
    ax.plot(s.t[1:1+len(sdv_p)] / 3600, sdv_p, "b-", label="SDRE pursuer")
    ax.plot(p.t[1:1+len(pdv_p)] / 3600, pdv_p, "r--", label="PINN pursuer")
    ax.plot(s.t[1:1+len(sdv_e)] / 3600, sdv_e, "b:", label="SDRE evader")
    ax.plot(p.t[1:1+len(pdv_e)] / 3600, pdv_e, "r:", label="PINN evader")
    ax.set_xlabel("Time (hours)"); ax.set_ylabel("Cumulative ΔV (km/s)"); ax.set_title("Fuel Consumption")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 6) P error
    ax = axes[2, 1]
    n = min(s.P_history.shape[0], p.P_history.shape[0])
    p_err = np.array([np.linalg.norm(p.P_history[i] - s.P_history[i], "fro") /
                       max(np.linalg.norm(s.P_history[i], "fro"), 1e-30) for i in range(n)])
    ax.plot(s.t[:n] / 3600, p_err, "m-", alpha=0.7)
    ax.axhline(np.median(p_err), color="orange", linestyle="--", label=f"median={np.median(p_err):.2e}")
    ax.set_xlabel("Time (hours)"); ax.set_ylabel("Relative P Error"); ax.set_title("P Matrix Error Along Trajectory")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_yscale("log")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "exp2_trajectory_compare.png", dpi=150)
    print(f"  -> Saved: {OUTPUT_DIR / 'exp2_trajectory_compare.png'}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Experiment 3: Monte Carlo
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MonteCarloResult:
    n_runs: int
    sdre_final_dist: np.ndarray
    pinn_final_dist: np.ndarray
    sdre_min_dist: np.ndarray
    pinn_min_dist: np.ndarray
    sdre_dv_p: np.ndarray
    pinn_dv_p: np.ndarray
    sdre_times: np.ndarray
    pinn_times: np.ndarray
    pinn_fallback_rates: np.ndarray
    failed_runs: list[int] = field(default_factory=list)


def experiment_3_monte_carlo(n_runs: int = 30, dt: float = 20.0, seed: int = 123) -> MonteCarloResult:
    print("\n" + "=" * 60)
    print(f"  Experiment 3: Monte Carlo Robustness 3D ({n_runs} runs)")
    print("=" * 60)

    dyn = OrbitalDynamics(MU, A_C, E_C)
    rng = np.random.default_rng(seed)

    mc = MonteCarloResult(
        n_runs=n_runs,
        sdre_final_dist=np.zeros(n_runs), pinn_final_dist=np.zeros(n_runs),
        sdre_min_dist=np.zeros(n_runs), pinn_min_dist=np.zeros(n_runs),
        sdre_dv_p=np.zeros(n_runs), pinn_dv_p=np.zeros(n_runs),
        sdre_times=np.zeros(n_runs), pinn_times=np.zeros(n_runs),
        pinn_fallback_rates=np.zeros(n_runs),
    )

    for run in range(n_runs):
        Xp0 = np.zeros(6); Xe0 = np.zeros(6)
        Xp0[:3] = rng.uniform(100, 800, 3) * rng.choice([-1, 1], 3)
        Xp0[3:] = rng.uniform(-0.05, 0.05, 3)
        Xe0[3:] = rng.uniform(-0.02, 0.02, 3)
        nu0 = rng.uniform(0, 2 * np.pi)

        try:
            sdre_ctrl = SDREGameController(Q, R, GAMMA)
            sdre_res, sdre_t = _run_sim(dyn, sdre_ctrl, Xp0, Xe0, nu0, dt)

            pinn_ctrl = NeuralSDREController(CHECKPOINT, device="cpu")
            pinn_res, pinn_t = _run_sim(dyn, pinn_ctrl, Xp0, Xe0, nu0, dt)
        except Exception as e:
            print(f"  [Run {run+1}] FAILED: {e}")
            mc.failed_runs.append(run)
            continue

        sd, pd = _rel_dist(sdre_res.states), _rel_dist(pinn_res.states)
        mc.sdre_final_dist[run] = sd[-1]
        mc.pinn_final_dist[run] = pd[-1]
        mc.sdre_min_dist[run] = sd.min()
        mc.pinn_min_dist[run] = pd.min()

        sdv = _delta_v(sdre_res.u_p_history, sdre_res.t)
        pdv = _delta_v(pinn_res.u_p_history, pinn_res.t)
        mc.sdre_dv_p[run] = sdv[-1] if len(sdv) else 0
        mc.pinn_dv_p[run] = pdv[-1] if len(pdv) else 0

        mc.sdre_times[run] = sdre_t
        mc.pinn_times[run] = pinn_t
        mc.pinn_fallback_rates[run] = pinn_ctrl.fallback_calls / max(pinn_ctrl.total_calls, 1)

        if (run + 1) % 5 == 0:
            print(f"  ... {run+1}/{n_runs} done")

    _print_mc_stats(mc)
    _plot_monte_carlo(mc)
    return mc


def _print_mc_stats(mc: MonteCarloResult):
    valid = [i for i in range(mc.n_runs) if i not in mc.failed_runs]
    if not valid:
        print("  All runs failed!"); return
    print(f"\n  Completed: {len(valid)}/{mc.n_runs}, Failed: {len(mc.failed_runs)}")
    print(f"\n  {'Metric':<30s} {'SDRE (mean±std)':>22s} {'PINN (mean±std)':>22s}")
    print(f"  {'-'*30} {'-'*22} {'-'*22}")
    for name, sv, pv in [
        ("Final dist (km)", mc.sdre_final_dist[valid], mc.pinn_final_dist[valid]),
        ("Min dist (km)", mc.sdre_min_dist[valid], mc.pinn_min_dist[valid]),
        ("Pursuer ΔV (km/s)", mc.sdre_dv_p[valid], mc.pinn_dv_p[valid]),
        ("Compute time (s)", mc.sdre_times[valid], mc.pinn_times[valid]),
    ]:
        print(f"  {name:<30s} {np.mean(sv):>10.4f}±{np.std(sv):<10.4f} "
              f"{np.mean(pv):>10.4f}±{np.std(pv):<10.4f}")
    fb = mc.pinn_fallback_rates[valid]
    print(f"  {'PINN fallback rate':<30s} {'':>22s} {np.mean(fb):>10.2%}±{np.std(fb):.2%}")


def _plot_monte_carlo(mc: MonteCarloResult):
    valid = [i for i in range(mc.n_runs) if i not in mc.failed_runs]
    if len(valid) < 2: return
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Experiment 3: Monte Carlo 3D ({len(valid)} runs)", fontsize=14)

    ax = axes[0, 0]
    ax.scatter(mc.sdre_final_dist[valid], mc.pinn_final_dist[valid], alpha=0.5, s=15)
    lim = max(mc.sdre_final_dist[valid].max(), mc.pinn_final_dist[valid].max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="y=x")
    ax.set_xlabel("SDRE Final Distance (km)"); ax.set_ylabel("PINN Final Distance (km)")
    ax.set_title("Final Distance"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(mc.sdre_dv_p[valid], mc.pinn_dv_p[valid], alpha=0.5, s=15)
    lim = max(mc.sdre_dv_p[valid].max(), mc.pinn_dv_p[valid].max(), 1e-10) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3)
    ax.set_xlabel("SDRE Pursuer ΔV"); ax.set_ylabel("PINN Pursuer ΔV")
    ax.set_title("Fuel Consumption"); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    diff = mc.pinn_final_dist[valid] - mc.sdre_final_dist[valid]
    ax.hist(diff, bins=25, alpha=0.7, color="steelblue", edgecolor="white")
    ax.axvline(0, color="k", linestyle="--")
    ax.set_xlabel("PINN − SDRE distance (km)"); ax.set_ylabel("Count")
    ax.set_title("Distance Difference"); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    fb = mc.pinn_fallback_rates[valid] * 100
    ax.hist(fb, bins=20, alpha=0.7, color="orange", edgecolor="white")
    ax.axvline(np.mean(fb), color="red", linestyle="--", label=f"mean={np.mean(fb):.1f}%")
    ax.set_xlabel("Fallback Rate (%)"); ax.set_ylabel("Count")
    ax.set_title("PINN OOD Fallback"); ax.legend()

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "exp3_monte_carlo.png", dpi=150)
    print(f"  -> Saved: {OUTPUT_DIR / 'exp3_monte_carlo.png'}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Experiment 4: Extreme Conditions
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ExtremeResult:
    name: str
    sdre_final_dist: float
    pinn_final_dist: float
    sdre_dv_p: float
    pinn_dv_p: float
    p_err_mean: float
    p_err_max: float
    pinn_fallback_rate: float
    success: bool


def experiment_4_extreme_conditions(dt: float = 20.0) -> list[ExtremeResult]:
    print("\n" + "=" * 60)
    print("  Experiment 4: Extreme Conditions 3D")
    print("=" * 60)

    cases = [
        ("Periapsis (ν≈0)",
         np.array([300, 300, 300, 0.01, 0.01, 0.01]), np.zeros(6), 0.01),
        ("Apoapsis (ν≈π)",
         np.array([300, 300, 300, 0.01, 0.01, 0.01]), np.zeros(6), np.pi),
        ("Close range (15 km)",
         np.array([5, 5, 5, 0.001, 0.001, 0.001]), np.zeros(6), 0.5),
        ("Far range (1500 km)",
         np.array([900, 700, 600, 0.02, 0.02, 0.02]), np.zeros(6), 1.0),
        ("High velocity",
         np.array([200, 200, 200, 0.1, -0.1, 0.05]), np.zeros(6), 0.8),
        ("Evader maneuvers",
         np.array([400, 400, 400, 0.01, 0.01, 0.01]),
         np.array([0, 0, 0, 0.03, -0.02, 0.01]), 0.3),
    ]

    dyn = OrbitalDynamics(MU, A_C, E_C)
    results = []

    for name, Xp0, Xe0, nu0 in cases:
        print(f"\n  --- {name} ---")
        try:
            sdre_ctrl = SDREGameController(Q, R, GAMMA)
            sdre_res, _ = _run_sim(dyn, sdre_ctrl, Xp0, Xe0, nu0, dt)

            pinn_ctrl = NeuralSDREController(CHECKPOINT, device="cpu")
            pinn_res, _ = _run_sim(dyn, pinn_ctrl, Xp0, Xe0, nu0, dt)

            sd, pd = _rel_dist(sdre_res.states), _rel_dist(pinn_res.states)
            sdv = _delta_v(sdre_res.u_p_history, sdre_res.t)
            pdv = _delta_v(pinn_res.u_p_history, pinn_res.t)

            n = min(sdre_res.P_history.shape[0], pinn_res.P_history.shape[0])
            p_err = np.array([
                np.linalg.norm(pinn_res.P_history[i] - sdre_res.P_history[i], "fro") /
                max(np.linalg.norm(sdre_res.P_history[i], "fro"), 1e-30) for i in range(n)
            ])
            er = ExtremeResult(
                name=name, sdre_final_dist=sd[-1], pinn_final_dist=pd[-1],
                sdre_dv_p=sdv[-1] if len(sdv) else 0,
                pinn_dv_p=pdv[-1] if len(pdv) else 0,
                p_err_mean=np.mean(p_err), p_err_max=np.max(p_err),
                pinn_fallback_rate=pinn_ctrl.fallback_calls / max(pinn_ctrl.total_calls, 1),
                success=True,
            )
        except Exception as e:
            print(f"    FAILED: {e}")
            er = ExtremeResult(
                name=name, sdre_final_dist=np.nan, pinn_final_dist=np.nan,
                sdre_dv_p=np.nan, pinn_dv_p=np.nan,
                p_err_mean=np.nan, p_err_max=np.nan,
                pinn_fallback_rate=np.nan, success=False,
            )
        results.append(er)

    _print_extreme_table(results)
    _plot_extreme(results)
    return results


def _print_extreme_table(results):
    print(f"\n  {'Scenario':<25s} {'SDRE dist':>10s} {'PINN dist':>10s} "
          f"{'SDRE ΔV':>10s} {'PINN ΔV':>10s} {'P err':>10s} {'Fallback':>8s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for r in results:
        if not r.success:
            print(f"  {r.name:<25s} {'FAIL':>10s}"); continue
        print(f"  {r.name:<25s} {r.sdre_final_dist:>10.2f} {r.pinn_final_dist:>10.2f} "
              f"{r.sdre_dv_p:>10.6f} {r.pinn_dv_p:>10.6f} "
              f"{r.p_err_mean:>10.2e} {r.pinn_fallback_rate:>7.1%}")


def _plot_extreme(results):
    ok = [r for r in results if r.success]
    if len(ok) < 2: return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Experiment 4: Extreme Conditions 3D", fontsize=14)
    names = [r.name for r in ok]; x = np.arange(len(ok)); w = 0.35

    ax = axes[0]
    ax.bar(x - w/2, [r.sdre_final_dist for r in ok], w, label="SDRE", color="steelblue")
    ax.bar(x + w/2, [r.pinn_final_dist for r in ok], w, label="PINN", color="coral")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Final Distance (km)"); ax.set_title("Final Distance"); ax.legend()

    ax = axes[1]
    ax.bar(x - w/2, [r.sdre_dv_p for r in ok], w, label="SDRE", color="steelblue")
    ax.bar(x + w/2, [r.pinn_dv_p for r in ok], w, label="PINN", color="coral")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Pursuer ΔV (km/s)"); ax.set_title("Fuel Consumption"); ax.legend()

    ax = axes[2]
    ax.bar(x, [r.p_err_mean for r in ok], color="mediumpurple", label="mean")
    ax.bar(x, [r.p_err_max - r.p_err_mean for r in ok],
           bottom=[r.p_err_mean for r in ok], color="plum", alpha=0.6, label="max-mean")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Relative P Error"); ax.set_title("P Matrix Error"); ax.legend(); ax.set_yscale("log")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "exp4_extreme_conditions.png", dpi=150)
    print(f"  -> Saved: {OUTPUT_DIR / 'exp4_extreme_conditions.png'}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Report Generation
# ═══════════════════════════════════════════════════════════════════

def generate_report(r1, r2, r3, r4):
    lines = [
        "# 3D 控制效果对比报告: SDRE vs PINN", "",
        "## 实验 1: 静态精度", "",
        f"- 总样本: {r1.n_samples}, 有效: {r1.n_valid}, ARE 失败: {r1.n_are_fail}, OOD: {r1.n_pinn_ood}",
        "", "| 指标 | mean | median | p95 | p99 | max |",
        "|------|------|--------|-----|-----|-----|",
    ]
    for name, arr in [("ΔP/P", r1.dp_fro_rel), ("Δu_p/u_p", r1.du_p_rel),
                      ("Δu_e/u_e", r1.du_e_rel), ("ΔJ/J", r1.dJ_rel)]:
        if len(arr) == 0: continue
        lines.append(f"| {name} | {np.mean(arr):.2e} | {np.median(arr):.2e} | "
                     f"{np.percentile(arr,95):.2e} | {np.percentile(arr,99):.2e} | {np.max(arr):.2e} |")

    lines += ["", "![](exp1_static_accuracy.png)", "", "## 实验 2: 标准轨迹闭环", ""]
    s, p = r2.sdre_result, r2.pinn_result
    sd, pd = _rel_dist(s.states), _rel_dist(p.states)
    sdv, pdv = _delta_v(s.u_p_history, s.t), _delta_v(p.u_p_history, p.t)
    lines += [
        "| 指标 | SDRE | PINN | 差值 |",
        "|------|------|------|------|",
        f"| 终端距离 (km) | {sd[-1]:.2f} | {pd[-1]:.2f} | {pd[-1]-sd[-1]:+.2f} |",
        f"| 追踪者 ΔV (km/s) | {sdv[-1]:.6f} | {pdv[-1]:.6f} | {pdv[-1]-sdv[-1]:+.6f} |",
        f"| 计算耗时 (s) | {r2.sdre_time_s:.1f} | {r2.pinn_time_s:.1f} | {r2.pinn_time_s/r2.sdre_time_s:.2f}x |",
        f"| PINN OOD 回退 | — | {r2.pinn_fallback_count} / {r2.pinn_total_calls} | — |",
        "", "![](exp2_trajectory_compare.png)", "",
    ]

    lines += ["## 实验 3: 蒙特卡洛鲁棒性", ""]
    valid = [i for i in range(r3.n_runs) if i not in r3.failed_runs]
    lines.append(f"- 运行: {len(valid)}/{r3.n_runs} 成功")
    if valid:
        lines += ["", "| 指标 | SDRE | PINN |", "|------|------|------|"]
        for name, sv, pv in [
            ("终端距离 (km)", r3.sdre_final_dist[valid], r3.pinn_final_dist[valid]),
            ("追踪者 ΔV", r3.sdre_dv_p[valid], r3.pinn_dv_p[valid]),
        ]:
            lines.append(f"| {name} | {np.mean(sv):.3f}±{np.std(sv):.3f} | {np.mean(pv):.3f}±{np.std(pv):.3f} |")
    lines += ["", "![](exp3_monte_carlo.png)", "", "## 实验 4: 极端条件", ""]
    lines += ["| 场景 | SDRE 终端距离 | PINN 终端距离 | P 误差 | 回退率 |",
              "|------|-------------|-------------|--------|--------|"]
    for r in r4:
        if not r.success:
            lines.append(f"| {r.name} | FAIL | — | — | — |")
        else:
            lines.append(f"| {r.name} | {r.sdre_final_dist:.1f} | {r.pinn_final_dist:.1f} | "
                         f"{r.p_err_mean:.2e} | {r.pinn_fallback_rate:.1%} |")
    lines += ["", "![](exp4_extreme_conditions.png)"]

    report_path = OUTPUT_DIR / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  -> Report saved: {report_path}")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")

    r1 = experiment_1_static_accuracy(n_samples=10000)
    r2 = experiment_2_trajectory_compare()
    r3 = experiment_3_monte_carlo(n_runs=30)
    r4 = experiment_4_extreme_conditions()
    generate_report(r1, r2, r3, r4)

    print("\n" + "=" * 60)
    print("  All 3D experiments completed!")
    print(f"  Results in: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
