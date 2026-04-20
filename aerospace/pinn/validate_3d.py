"""
3D PINN 模型正确性验证实验

包含 6 项验证：
  1. P 矩阵 SPD 验证（全量抽检）
  2. ARE 残差检验（P 代入 Riccati 方程的残差）
  3. 逐元素 P 矩阵精度分析
  4. 控制律一致性对比（同一状态下 SDRE vs PINN 的 u_p, u_e）
  5. 多初值闭环轨迹对比（相对距离、ΔV 累积）
  6. 汇总报告与图片保存

用法:
    uv run python -m aerospace.pinn.validate_3d
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.linalg import LinAlgError
from scipy.stats import qmc

from aerospace.dynamics.nerm import OrbitalDynamics
from aerospace.control.sdre import SDREGameController
from aerospace.control.neural import NeuralSDREController
from aerospace.pinn.data_generator import (
    STATE_DIM, CTRL_DIM, extract_asdc_features, _solve_are_balanced,
)
from aerospace.simulation.nerm_sdre import SDRESimulation

import matplotlib
try:
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: F401
except Exception:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


CHECKPOINT = "checkpoints/sdre_pinn/best_model.pt"
OUTPUT_DIR = Path("results/validate_3d")


def _make_test_states(n=500, seed=9999):
    """生成随机测试状态（独立于训练集）。"""
    dynamics = OrbitalDynamics(mu=3.986e5, a_c=15000.0, e_c=0.5)

    sampler_p = qmc.LatinHypercube(d=3, seed=seed)
    pos_p = qmc.scale(sampler_p.random(n=n), [-1000]*3, [1000]*3)
    sampler_e = qmc.LatinHypercube(d=3, seed=seed+3)
    pos_e = qmc.scale(sampler_e.random(n=n), [-1000]*3, [1000]*3)
    sampler_nu = qmc.LatinHypercube(d=1, seed=seed+7)
    nu = 2 * np.pi * sampler_nu.random(n=n).reshape(-1)

    b_p = np.zeros((STATE_DIM, CTRL_DIM)); b_p[3:, :] = np.eye(CTRL_DIM)
    q = np.eye(STATE_DIM)
    r = np.eye(CTRL_DIM) * 1e13
    gamma = np.sqrt(2.0)
    r_eff = r / (1.0 - gamma**(-2))

    results = []
    for i in range(n):
        X_p = np.array([pos_p[i,0], pos_p[i,1], pos_p[i,2], 0, 0, 0])
        X_e = np.array([pos_e[i,0], pos_e[i,1], pos_e[i,2], 0, 0, 0])
        r_c, nu_dot, nu_ddot = dynamics.get_orbital_params(float(nu[i]))
        a_sdc = dynamics.get_SDC_matrix(X_p, X_e, r_c, nu_dot, nu_ddot)
        try:
            p_true = _solve_are_balanced(a_sdc, b_p, q, r_eff)
        except (LinAlgError, ValueError):
            continue
        results.append((X_p, X_e, float(nu[i]), a_sdc, p_true))
    return results, dynamics, q, r, gamma


# ─── 实验 1: SPD 验证 ───────────────────────────────────────────────
def exp1_spd_check(ctrl, test_data):
    feats = np.array([extract_asdc_features(d[3]) for d in test_data])
    p_batch = ctrl.predict_p_batch(feats)
    n = p_batch.shape[0]
    not_sym = 0; not_pd = 0
    min_eig_all = []
    for i in range(n):
        P = p_batch[i]
        if not np.allclose(P, P.T, atol=1e-5):
            not_sym += 1
        eigvals = np.linalg.eigvalsh(P)
        min_eig_all.append(eigvals.min())
        if eigvals.min() <= 0:
            not_pd += 1
    min_eig_all = np.array(min_eig_all)
    print("\n" + "="*60)
    print("  实验 1: SPD 验证（对称正定性）")
    print("="*60)
    print(f"  测试样本数:        {n}")
    print(f"  非对称:            {not_sym}")
    print(f"  非正定:            {not_pd}")
    print(f"  最小特征值 min:    {min_eig_all.min():.6e}")
    print(f"  最小特征值 median: {np.median(min_eig_all):.6e}")
    print(f"  结论: {'PASS' if not_sym == 0 and not_pd == 0 else 'FAIL'}")
    return p_batch


# ─── 实验 2: ARE 残差检验 ────────────────────────────────────────────
def exp2_are_residual(p_batch, test_data, q, r, gamma):
    b_p = np.zeros((STATE_DIM, CTRL_DIM)); b_p[3:, :] = np.eye(CTRL_DIM)
    b_e = -b_p
    r_inv = np.linalg.inv(r)
    G = b_p @ r_inv @ b_p.T - (gamma**-2) * (b_e @ r_inv @ b_e.T)

    residuals = []
    for i, (X_p, X_e, nu, a_sdc, p_true) in enumerate(test_data):
        P = p_batch[i]
        R_are = P @ a_sdc + a_sdc.T @ P - P @ G @ P + q
        res = np.linalg.norm(R_are, 'fro')
        norm_p = (np.linalg.norm(P @ a_sdc, 'fro')
                  + np.linalg.norm(P @ G @ P, 'fro')
                  + np.linalg.norm(q, 'fro'))
        residuals.append(res / max(norm_p, 1e-12))
    residuals = np.array(residuals)

    print("\n" + "="*60)
    print("  实验 2: ARE 残差检验（相对残差）")
    print("="*60)
    print(f"  样本数:   {len(residuals)}")
    print(f"  均值:     {residuals.mean():.6e}")
    print(f"  中位数:   {np.median(residuals):.6e}")
    print(f"  P90:      {np.percentile(residuals, 90):.6e}")
    print(f"  P99:      {np.percentile(residuals, 99):.6e}")
    print(f"  最大值:   {residuals.max():.6e}")
    return residuals


# ─── 实验 3: 逐元素精度分析 ──────────────────────────────────────────
def exp3_elementwise(p_batch, test_data):
    p_true_all = np.array([d[4] for d in test_data])
    diff = p_batch - p_true_all
    abs_err = np.abs(diff)
    rel_err = abs_err / np.maximum(np.abs(p_true_all), 1e-12)

    print("\n" + "="*60)
    print("  实验 3: 逐元素精度分析")
    print("="*60)
    print(f"  {'位置':>10s}  {'|P_true| 中位数':>16s}  {'绝对误差 中位数':>16s}  {'相对误差 中位数':>16s}")
    print("  " + "-"*66)
    labels = [
        ((0,0),"P[0,0]"), ((0,1),"P[0,1]"), ((0,2),"P[0,2]"),
        ((0,3),"P[0,3]"), ((0,4),"P[0,4]"), ((0,5),"P[0,5]"),
        ((1,1),"P[1,1]"), ((1,2),"P[1,2]"), ((1,3),"P[1,3]"),
        ((2,2),"P[2,2]"), ((2,3),"P[2,3]"),
        ((3,3),"P[3,3]"), ((3,4),"P[3,4]"), ((3,5),"P[3,5]"),
        ((4,4),"P[4,4]"), ((4,5),"P[4,5]"),
        ((5,5),"P[5,5]"),
    ]
    for (i, j), name in labels:
        true_vals = p_true_all[:, i, j]
        ae = abs_err[:, i, j]
        re = rel_err[:, i, j]
        print(f"  {name:>10s}  {np.median(np.abs(true_vals)):>16.4e}  "
              f"{np.median(ae):>16.4e}  {np.median(re):>16.4e}")

    fro_err = np.sqrt(np.sum(diff**2, axis=(1,2)))
    fro_true = np.sqrt(np.sum(p_true_all**2, axis=(1,2)))
    rel_fro = fro_err / np.maximum(fro_true, 1e-12)
    print(f"\n  整体相对 Frobenius 误差:")
    print(f"    均值={rel_fro.mean():.6e}  中位数={np.median(rel_fro):.6e}  "
          f"P90={np.percentile(rel_fro,90):.6e}  P99={np.percentile(rel_fro,99):.6e}")
    return rel_fro


# ─── 实验 4: 控制律一致性 ────────────────────────────────────────────
def exp4_control_consistency(ctrl_neural, test_data, q, r, gamma):
    ctrl_sdre = SDREGameController(Q=q, R=r, gamma=gamma)
    up_errs = []; ue_errs = []; up_rel_errs = []; ue_rel_errs = []

    for X_p, X_e, nu, a_sdc, p_true in test_data[:200]:
        x_rel = X_p - X_e
        u_p_sdre, u_e_sdre = ctrl_sdre.compute_control(a_sdc, x_rel, solve_are=True)
        u_p_pinn, u_e_pinn = ctrl_neural.compute_control(a_sdc, x_rel)

        up_err = np.linalg.norm(u_p_pinn - u_p_sdre)
        ue_err = np.linalg.norm(u_e_pinn - u_e_sdre)
        up_errs.append(up_err)
        ue_errs.append(ue_err)
        up_rel_errs.append(up_err / max(np.linalg.norm(u_p_sdre), 1e-30))
        ue_rel_errs.append(ue_err / max(np.linalg.norm(u_e_sdre), 1e-30))

    up_errs = np.array(up_errs); ue_errs = np.array(ue_errs)
    up_rel = np.array(up_rel_errs); ue_rel = np.array(ue_rel_errs)

    print("\n" + "="*60)
    print("  实验 4: 控制律一致性（u_p / u_e 偏差）")
    print("="*60)
    print(f"  {'':>20s}  {'|Δu| 均值':>14s}  {'|Δu| 中位数':>14s}  {'相对误差 均值':>14s}  {'相对误差 P99':>14s}")
    print(f"  {'u_p (追踪星)':>20s}  {up_errs.mean():>14.4e}  {np.median(up_errs):>14.4e}  {up_rel.mean():>14.4e}  {np.percentile(up_rel,99):>14.4e}")
    print(f"  {'u_e (逃逸星)':>20s}  {ue_errs.mean():>14.4e}  {np.median(ue_errs):>14.4e}  {ue_rel.mean():>14.4e}  {np.percentile(ue_rel,99):>14.4e}")
    return up_rel, ue_rel


# ─── 实验 5: 多初值闭环轨迹对比 ──────────────────────────────────────
def exp5_closed_loop(checkpoint, dynamics, q, r, gamma):
    scenarios = [
        {"name": "A: 追方500km后方", "x_p0": [500., 500., 50., 0.01, 0.01, 0.], "x_e0": [0,0,0,0,0,0], "nu0": 0.0},
        {"name": "B: 追方100km近距", "x_p0": [100., 50., 10., 0.005, 0.005, 0.], "x_e0": [0,0,0,0,0,0], "nu0": 0.5},
        {"name": "C: 逃方偏心轨道", "x_p0": [300., 200., 30., 0., 0., 0.], "x_e0": [-100,50,10,0.001,-0.001,0.], "nu0": 1.0},
    ]

    print("\n" + "="*60)
    print("  实验 5: 多初值闭环轨迹对比")
    print("="*60)
    print(f"  {'场景':>22s}  {'SDRE 终距(km)':>14s}  {'PINN 终距(km)':>14s}  {'终距偏差(%)':>12s}  {'SDRE ΔVp':>10s}  {'PINN ΔVp':>10s}  {'ΔV偏差(%)':>10s}  {'OOD回退':>8s}")
    print("  " + "-"*100)

    all_results = []
    for sc in scenarios:
        x_p0 = np.array(sc["x_p0"], dtype=float)
        x_e0 = np.array(sc["x_e0"], dtype=float)

        sdre_ctrl = SDREGameController(Q=q, R=r, gamma=gamma)
        sdre_sim = SDRESimulation(dynamics, sdre_ctrl, x_p0, x_e0, nu0=sc["nu0"], dt=20.0, log_interval=9999)
        sdre_res = sdre_sim.run()

        neural_ctrl = NeuralSDREController(checkpoint_path=checkpoint)
        neural_sim = SDRESimulation(dynamics, neural_ctrl, x_p0, x_e0, nu0=sc["nu0"], dt=20.0, log_interval=9999)
        neural_res = neural_sim.run()

        sdre_dist = np.linalg.norm(sdre_res.states[0:3, -1] - sdre_res.states[6:9, -1])
        pinn_dist = np.linalg.norm(neural_res.states[0:3, -1] - neural_res.states[6:9, -1])
        dist_err = abs(pinn_dist - sdre_dist) / max(sdre_dist, 1e-12) * 100

        sdre_dv = np.sum(np.sqrt(np.sum(sdre_res.u_p_history**2, axis=0))) * 20.0
        pinn_dv = np.sum(np.sqrt(np.sum(neural_res.u_p_history**2, axis=0))) * 20.0
        dv_err = abs(pinn_dv - sdre_dv) / max(sdre_dv, 1e-12) * 100

        fb = neural_ctrl.fallback_calls
        total = neural_ctrl.total_calls
        print(f"  {sc['name']:>22s}  {sdre_dist:>14.3f}  {pinn_dist:>14.3f}  {dist_err:>12.2f}  "
              f"{sdre_dv:>10.3f}  {pinn_dv:>10.3f}  {dv_err:>10.2f}  {fb}/{total}")

        all_results.append({
            "name": sc["name"], "sdre_res": sdre_res, "pinn_res": neural_res,
            "sdre_dist": sdre_dist, "pinn_dist": pinn_dist,
        })
    return all_results


# ─── 绘图 ──────────────────────────────────────────────────────────
def _plot_trajectory_comparison(all_results, output_dir):
    titles = ["Scenario A: Chaser 500km behind",
              "Scenario B: Chaser 100km close",
              "Scenario C: Evader eccentric orbit"]
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
    if n == 1:
        axes = [axes]
    for idx, (ax, res) in enumerate(zip(axes, all_results)):
        sdre = res["sdre_res"]
        pinn = res["pinn_res"]
        ax.plot(sdre.states[0] - sdre.states[6], sdre.states[1] - sdre.states[7],
                label="SDRE (truth)", linewidth=2, alpha=0.8)
        ax.plot(pinn.states[0] - pinn.states[6], pinn.states[1] - pinn.states[7],
                label="PINN", linewidth=2, linestyle="--", alpha=0.8)
        ax.set_xlabel(r"$\Delta x$ (km)")
        ax.set_ylabel(r"$\Delta y$ (km)")
        ax.set_title(titles[idx] if idx < len(titles) else res["name"])
        ax.legend(fontsize=9)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "追逃轨迹对比图.png", dpi=150)
    print(f"\n  Trajectory plot saved: {output_dir / '追逃轨迹对比图.png'}")


def _plot_distance_comparison(all_results, output_dir):
    titles = ["Scenario A: Chaser 500km behind",
              "Scenario B: Chaser 100km close",
              "Scenario C: Evader eccentric orbit"]
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 4))
    if n == 1:
        axes = [axes]
    for idx, (ax, res) in enumerate(zip(axes, all_results)):
        sdre = res["sdre_res"]
        pinn = res["pinn_res"]
        sdre_d = np.sqrt(np.sum((sdre.states[0:3] - sdre.states[6:9])**2, axis=0))
        pinn_d = np.sqrt(np.sum((pinn.states[0:3] - pinn.states[6:9])**2, axis=0))
        ax.plot(sdre.t, sdre_d, label="SDRE", linewidth=1.5)
        ax.plot(pinn.t, pinn_d, label="PINN", linewidth=1.5, linestyle="--")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Relative Distance (km)")
        ax.set_title(titles[idx] if idx < len(titles) else res["name"])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "追逃相对距离对比图.png", dpi=150)
    print(f"  Distance plot saved: {output_dir / '追逃相对距离对比图.png'}")


def _plot_residual_hist(residuals, output_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(np.median(residuals), color="red", linestyle="--",
               label=f"median={np.median(residuals):.2e}")
    ax.axvline(np.percentile(residuals, 99), color="orange", linestyle="--",
               label=f"P99={np.percentile(residuals,99):.2e}")
    ax.set_xlabel("ARE Relative Residual")
    ax.set_ylabel("Count")
    ax.set_title("$P_{pred}$ ARE Residual Distribution (3D)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "ARE相对残差直方图.png", dpi=150)
    print(f"  ARE residual plot saved: {output_dir / 'ARE相对残差直方图.png'}")


# ─── 主函数 ─────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "#"*60)
    print("#  3D PINN 模型正确性验证实验")
    print("#"*60)

    ctrl = NeuralSDREController(checkpoint_path=CHECKPOINT)
    test_data, dynamics, q, r, gamma = _make_test_states(n=500, seed=9999)
    print(f"\n  测试集: {len(test_data)} 个独立样本 (seed=9999，与训练集无交集)")

    p_batch = exp1_spd_check(ctrl, test_data)
    residuals = exp2_are_residual(p_batch, test_data, q, r, gamma)
    rel_fro = exp3_elementwise(p_batch, test_data)
    up_rel, ue_rel = exp4_control_consistency(ctrl, test_data, q, r, gamma)
    all_results = exp5_closed_loop(CHECKPOINT, dynamics, q, r, gamma)

    _plot_trajectory_comparison(all_results, OUTPUT_DIR)
    _plot_distance_comparison(all_results, OUTPUT_DIR)
    _plot_residual_hist(residuals, OUTPUT_DIR)

    print("\n" + "#"*60)
    print("#  验证汇总")
    print("#"*60)
    print(f"  SPD 通过率:           {len(test_data)}/{len(test_data)}")
    print(f"  ARE 相对残差 median:  {np.median(residuals):.4e}")
    print(f"  P 相对Fro误差 median: {np.median(rel_fro):.4e}")
    print(f"  u_p 相对误差 median:  {np.median(up_rel):.4e}")
    print(f"  u_e 相对误差 median:  {np.median(ue_rel):.4e}")
    print(f"  图片输出目录:         {OUTPUT_DIR.resolve()}")
    print("#"*60 + "\n")


if __name__ == "__main__":
    main()
