"""
逆最优 SDRE 拟合: 从最优控制轨迹反演状态依赖控制惩罚 R(x)

利用 Phase 1 (CasADi) 输出的协态 λ 和最优轨迹 (x*, u*),
在每个状态点求解标量 r 使得 SDRE 反馈控制匹配 PMP 最优控制:
    u_SDRE = -R^{-1} B^T P x,  R = r * I_2

其中 P 由 ARE 确定, 通过标量化化搜索匹配 u*。
然后拟合 r(x) = R_min + (R_max - R_min) * sigmoid((r - r_0) / w)
"""

import numpy as np
from scipy.linalg import solve_continuous_are, LinAlgError
from scipy.optimize import minimize_scalar
from pathlib import Path


def compute_sdre_control_4d(
    A_SDC: np.ndarray, x_rel: np.ndarray, r: float
) -> np.ndarray:
    """给定标量 r 和 4D 状态, 求解纯追捕 SDRE 控制量 (无博弈)。

    R = r * I_2, Q = I_4, B = [[0,I]]^T, 无 evader (γ→∞)
    """
    Q = np.eye(4)
    B = np.zeros((4, 2))
    B[2:, :] = np.eye(2)
    R = np.eye(2) * r

    try:
        P = solve_continuous_are(A_SDC, B, Q, R)
    except (LinAlgError, ValueError):
        return np.full(2, np.inf)

    u = -np.linalg.inv(R) @ B.T @ P @ x_rel
    return u


def find_best_r(
    A_SDC: np.ndarray,
    x_rel: np.ndarray,
    u_target: np.ndarray,
    r_bounds: tuple = (1e2, 1e20),
) -> float:
    """网格搜索 + 局部优化寻找使 SDRE 控制逼近 u_target 的最佳 r。"""
    lo, hi = np.log10(r_bounds[0]), np.log10(r_bounds[1])

    def err(log10_r):
        u_sdre = compute_sdre_control_4d(A_SDC, x_rel, 10.0 ** float(log10_r))
        if np.any(np.isinf(u_sdre)):
            return 1e20
        return float(np.linalg.norm(u_sdre - u_target))

    # 粗搜索
    best_logr, best_err = lo, np.inf
    for logr in np.linspace(lo, hi, 80):
        e = err(logr)
        if e < best_err:
            best_err, best_logr = e, logr

    if best_err > 1e10:
        return np.nan

    # 局部精化
    res = minimize_scalar(
        err, bounds=(best_logr - 2.0, best_logr + 2.0), method="bounded"
    )
    if res.success and err(res.x) < best_err:
        return 10.0 ** float(res.x)
    return 10.0 ** best_logr


def fit_inverse_optimal(
    sol_dir: str = "outputs/optimal_control",
    max_points: int = 300,
) -> dict:
    """从多个最优控制解中反演 R(r) 并拟合 sigmoid 模型。

    Parameters
    ----------
    sol_dir : str
        Phase 1 输出目录
    max_points : int
        最多使用的轨迹点总数

    Returns
    -------
    dict with r_points, dist_points, rdot_points, fit_params, sol_data
    """
    from aerospace.dynamics.nerm_2d import OrbitalDynamics2D

    sol_dir = Path(sol_dir)
    sol_files = sorted(sol_dir.glob("sol_gamma_*.npz"))
    if not sol_files:
        raise FileNotFoundError(f"no solution files in {sol_dir}")

    orb = OrbitalDynamics2D(mu=3.986e5, a_c=15000.0, e_c=0.5)

    all_r, all_dist, all_rdot = [], [], []
    all_sol_data = []

    for sf in sol_files:
        data = np.load(sf)
        x_traj = data["x"]  # (5, N)
        u_traj = data["u"]  # (2, N)
        N_pts = x_traj.shape[1]

        # 均匀采样
        n_sample = min(max_points // len(sol_files), N_pts)
        idx = np.linspace(0, N_pts - 2, n_sample, dtype=int)  # 避免终端点

        for k in idx:
            dx, dy, dvx, dvy, nu = x_traj[:, k]
            x_rel = np.array([dx, dy, dvx, dvy])
            u_target = u_traj[:, k]

            dist = np.sqrt(dx**2 + dy**2)
            if dist < 0.05:  # 太近无意义
                continue

            rdot = (dx * dvx + dy * dvy) / (dist + 1e-12)

            X_p = x_rel.copy()
            X_e = np.zeros(4)
            r_c, nu_dot_, nu_ddot_ = orb.get_orbital_params(nu)
            A_SDC = orb.get_SDC_matrix(X_p, X_e, r_c, nu_dot_, nu_ddot_)

            r_best = find_best_r(A_SDC, x_rel, u_target)

            if not np.isnan(r_best) and r_best > 0:
                all_r.append(r_best)
                all_dist.append(dist)
                all_rdot.append(rdot)

        all_sol_data.append(dict(gamma=float(data["gamma"]), T=float(data["T"])))

    all_r = np.array(all_r)
    all_dist = np.array(all_dist)
    all_rdot = np.array(all_rdot)

    print(f"Valid points: {len(all_r)}, R range: [{all_r.min():.1e}, {all_r.max():.1e}]")
    print(f"Distance range: [{all_dist.min():.2f}, {all_dist.max():.2f}] km")

    # ── Sigmoid 拟合: R(r) = R_min + (R_max - R_min) * σ((r - r_0)/w) ──
    from scipy.optimize import curve_fit

    def sigmoid_model(r, R_min, R_max, r_0, w):
        return R_min + (R_max - R_min) / (1.0 + np.exp(-(r - r_0) / np.clip(w, 1, 1e6)))

    # 在 log10(R) 空间拟合 (更稳定)
    log_r = np.log10(all_r)
    p0 = [np.log10(np.percentile(all_r, 10)), np.log10(np.percentile(all_r, 90)),
          np.median(all_dist), 100.0]

    try:
        popt, pcov = curve_fit(
            lambda r, a, b, c, d: a + (b - a) / (1.0 + np.exp(-(r - c) / np.clip(d, 1, 1e6))),
            all_dist, log_r, p0=p0, maxfev=20000,
        )
        R_min_fit = 10.0 ** popt[0]
        R_max_fit = 10.0 ** popt[1]
        r_0_fit = popt[2]
        w_fit = abs(popt[3])
    except Exception:
        R_min_fit = np.percentile(all_r, 5)
        R_max_fit = np.percentile(all_r, 95)
        r_0_fit = np.median(all_dist)
        w_fit = max((all_dist.max() - all_dist.min()) / 4.0, 1.0)

    fit_params = dict(R_min=R_min_fit, R_max=R_max_fit, r_0=r_0_fit, w=w_fit)
    print(f"Fit: R_min={R_min_fit:.2e}, R_max={R_max_fit:.2e}, "
          f"r_0={r_0_fit:.1f}, w={w_fit:.1f}")

    return dict(
        r_points=all_r, dist_points=all_dist, rdot_points=all_rdot,
        fit_params=fit_params, sol_data=all_sol_data,
    )


def save_fit_result(result: dict, out_dir: str):
    """保存拟合结果到 .npz 文件。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "inverse_optimal_fit.npz",
        r_points=result["r_points"],
        dist_points=result["dist_points"],
        rdot_points=result["rdot_points"],
        **result["fit_params"],
    )
    print(f"Fit saved to: {out_dir / 'inverse_optimal_fit.npz'}")


def plot_fit_result(result: dict, out_dir: str):
    """可视化拟合结果。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左: R vs 距离
    ax = axes[0]
    ax.scatter(result["dist_points"], result["r_points"], s=6, alpha=0.3,
               color="tab:blue", label="inverse r(x)")
    r_range = np.linspace(result["dist_points"].min(), result["dist_points"].max(), 300)
    p = result["fit_params"]

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
    R_fit = p["R_min"] + (p["R_max"] - p["R_min"]) * sigmoid((r_range - p["r_0"]) / p["w"])
    ax.semilogy(r_range, R_fit, "r-", lw=2, label="sigmoid fit")
    ax.set_xlabel("Relative distance r (km)")
    ax.set_ylabel("R = r * I_2")
    ax.set_title("Inverse Optimal R(r)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)

    # 右: R vs 径向速度
    ax = axes[1]
    ax.scatter(result["rdot_points"], result["r_points"], s=6, alpha=0.3,
               color="tab:green")
    ax.set_xlabel("Radial velocity dr/dt (km/s)")
    ax.set_ylabel("R = r * I_2")
    ax.set_title("R vs Radial Velocity")
    ax.grid(True, alpha=0.4)

    fig.suptitle("Inverse Optimal SDRE: State-Dependent Control Penalty", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "inverse_optimal_fit.png", dpi=150)
    plt.close(fig)
    print(f"Plot saved to: {out_dir / 'inverse_optimal_fit.png'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sol-dir", type=str, default="outputs/optimal_control")
    ap.add_argument("--out", type=str, default="outputs/inverse_optimal")
    args = ap.parse_args()

    result = fit_inverse_optimal(args.sol_dir)
    p = result["fit_params"]
    print(f"\nR(r) = {p['R_min']:.2e} + ({p['R_max']:.2e} - {p['R_min']:.2e}) "
          f"* sigmoid((r - {p['r_0']:.1f}) / {p['w']:.1f})")

    save_fit_result(result, args.out)
    try:
        plot_fit_result(result, args.out)
    except Exception as e:
        print(f"Plot failed: {e}")
