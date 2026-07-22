"""
E0: 滤波有效性验证

比较 SDC-EKF / Jacobian-EKF / UKF 的一步预测精度与一致性。
扫描 dt={10,5,2}s、UKF 参数、有限差分步长。
N=100 配对种子，所有滤波器对同一种子使用相同初始条件与噪声序列。
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from aerospace.dynamics.nerm import OrbitalDynamics
from aerospace.control.sdre import SDREGameController
from aerospace.estimation.ekf import RelativeStateEKF
from aerospace.estimation.navigators import (
    SDCEKFNavigator, JacobianEKFNavigator, SquareRootUKFNavigator,
    BaseNavigator,
)
from aerospace.paths import DATA_DIR, ensure_figures_dir

# ── 配置 ────────────────────────────────────────────────────────────────────
DT_VALUES = [10, 5, 2]
N_SEEDS = 100
MASTER_SEED = 42
T_END_FRAC = 0.1  # 每个种子仿真时长 = T_orbit * T_END_FRAC
CAPTURE_DIST = 0.1  # km

# Jacobian-EKF 参数扫描
JAC_EPS_VALUES = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
JAC_STATE_SCALES = np.array([10.0, 10.0, 10.0, 0.01, 0.01, 0.01])

# UKF 参数扫描
UKF_PARAM_SETS = [
    {"alpha": 1.0, "beta": 2.0, "kappa": 0.0},
    {"alpha": 0.5, "beta": 2.0, "kappa": 0.0},
    {"alpha": 0.1, "beta": 2.0, "kappa": 0.0},
    {"alpha": 1.0, "beta": 2.0, "kappa": -3.0},
]

# 传感器噪声
SIGMA_ANG_DEG = 0.008
DEG2RAD = np.pi / 180.0

# ── 工具 ────────────────────────────────────────────────────────────────────

def _ensure_posdef(P: np.ndarray) -> bool:
    """检查协方差是否正定。"""
    try:
        eigvals = np.linalg.eigvalsh(P)
        return bool(np.all(eigvals > 1e-14))
    except Exception:
        return False


def _wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


# ── 单次一步预测评估 ───────────────────────────────────────────────────────

def _eval_one_step(
    dynamics: OrbitalDynamics,
    X_p_true: np.ndarray, X_e_true: np.ndarray, nu: float,
    u_p: np.ndarray, u_e: np.ndarray, dt: float,
    navigator: BaseNavigator,
    rng: np.random.Generator,
    z_noise: np.ndarray | None = None,  # 配对噪声：所有滤波器共享同一噪声样本
) -> dict:
    """对单个导航器执行一步预测+更新，返回诊断指标。"""
    r_c, nu_dot, nu_ddot = dynamics.get_orbital_params(nu)
    x_true_rel = X_p_true - X_e_true

    # ── 真实状态传播（RK45，作为 ground truth）─────────────────────────
    state0 = np.concatenate([X_p_true, X_e_true, [nu]])
    from scipy.integrate import solve_ivp
    sol = solve_ivp(dynamics.dynamics_13d, [0, dt], state0,
                    args=(u_p, u_e), method="RK45", rtol=1e-8, atol=1e-10)
    state1 = sol.y[:, -1]
    X_p_prop = state1[0:6]
    X_e_prop = state1[6:12]
    x_true_prop = X_p_prop - X_e_prop
    nu_prop = state1[12]
    r_c_prop, nu_dot_prop, nu_ddot_prop = dynamics.get_orbital_params(nu_prop)

    # ── 构造导航器上下文 ───────────────────────────────────────────────
    B = np.zeros((6, 3)); B[3:, :] = np.eye(3)
    du = u_p - u_e

    def A_SDC_fn(x_rel):
        X_e_est = X_p_true - x_rel
        return dynamics.get_SDC_matrix(X_p_true, X_e_est, r_c, nu_dot, nu_ddot)

    def f_discrete(x_rel):
        A = A_SDC_fn(x_rel)
        return x_rel + dt * (A @ x_rel + B @ du)

    A_SDC_nominal = dynamics.get_SDC_matrix(
        X_p_true, X_p_true - navigator.x, r_c, nu_dot, nu_ddot)

    # ── 导航器预测 ─────────────────────────────────────────────────────
    try:
        x_priori, P_priori = navigator.predict(
            A_SDC_nominal, B, u_p, u_e, dt,
            A_SDC_fn=A_SDC_fn, f_discrete=f_discrete)
    except Exception as e:
        return {"error": f"predict failed: {e}"}

    pred_err = x_priori - x_true_prop
    pred_err_pos = float(np.linalg.norm(pred_err[:3])) * 1000  # m
    pred_err_vel = float(np.linalg.norm(pred_err[3:])) * 1000  # m/s

    cov_posdef = _ensure_posdef(P_priori)

    # ── NEES（预测）─────────────────────────────────────────────────────
    try:
        nees_val = float(pred_err @ np.linalg.solve(P_priori, pred_err))
    except np.linalg.LinAlgError:
        nees_val = np.nan

    # ── 生成带噪声测量（配对：z_meas 由外部传入，各滤波器共用）─────
    z_true = RelativeStateEKF.measure(
        X_p_prop, X_e_prop, angle_only=navigator.angles_only)
    if z_noise is not None:
        z_meas_eff = z_true + z_noise
    else:
        z_meas_eff = z_true + rng.multivariate_normal(
            np.zeros(navigator.R.shape[0]), navigator.R)

    # ── 测量更新 ────────────────────────────────────────────────────────
    try:
        innov = navigator.update(x_priori, P_priori, z_meas_eff)
    except Exception as e:
        return {"error": f"update failed: {e}", "pred_err_pos": pred_err_pos,
                "pred_err_vel": pred_err_vel, "cov_posdef": cov_posdef,
                "nees": nees_val}

    # ── NIS ─────────────────────────────────────────────────────────────
    # 优先使用导航器缓存的新息协方差（UKF 用 P_yy，EKF 用 HPH^T+R）
    S_innov = getattr(navigator, '_last_S', None)
    if S_innov is None:
        z_pred = navigator._measure_from_rel_state(x_priori)
        H = RelativeStateEKF.meas_jacobian(x_priori, angle_only=navigator.angles_only)
        S_innov = H @ P_priori @ H.T + navigator.R
    try:
        nis_val = float(innov @ np.linalg.solve(S_innov, innov))
    except np.linalg.LinAlgError:
        nis_val = np.nan

    # ── 3σ 覆盖 ─────────────────────────────────────────────────────────
    # 位置各分量是否在 3σ 界内
    sigma_pos = np.sqrt(np.maximum(np.diag(P_priori)[:3], 0))
    within_3sigma = bool(np.all(np.abs(pred_err[:3]) < 3 * sigma_pos))

    # ── 角度 wrap 表现 ──────────────────────────────────────────────────
    ia, ie = navigator._angle_indices
    angle_innov_max = max(abs(innov[ia]), abs(innov[ie])) if len(innov) > 0 else 0.0

    # ── 后验 NEES ─────────────────────────────────────────────────────
    x_post_err = navigator.x - x_true_prop
    try:
        nees_post = float(x_post_err @ np.linalg.solve(navigator.P, x_post_err))
    except np.linalg.LinAlgError:
        nees_post = np.nan

    return {
        "pred_err_pos": pred_err_pos,
        "pred_err_vel": pred_err_vel,
        "nees": nees_val,
        "nees_post": nees_post,
        "nis": nis_val,
        "cov_posdef": cov_posdef,
        "within_3sigma": within_3sigma,
        "angle_innov_max": float(angle_innov_max),
        "P_trace": float(np.trace(P_priori)),
        "innov": innov,
    }


# ── 主实验 ─────────────────────────────────────────────────────────────────

def run_e0(output_dir: Path | None = None):
    """E0 主入口。"""
    if output_dir is None:
        output_dir = DATA_DIR / "e0_filter_validation"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    orb = OrbitalDynamics(mu=3.986e5, a_c=15000.0, e_c=0.5)
    ctrl = SDREGameController(Q=np.eye(6), R=np.eye(3) * 1e13, gamma=np.sqrt(2))

    sigma_ang = SIGMA_ANG_DEG * DEG2RAD
    R_meas = np.diag([sigma_ang**2, sigma_ang**2])
    Q_proc = np.diag([5e-4, 5e-4, 5e-4, 5e-8, 5e-8, 5e-8])
    P0_pos = (30.0 * sigma_ang) ** 2  # 30 km initial dist
    P0_vel = (0.02 * sigma_ang) ** 2
    P0 = np.diag([P0_pos, P0_pos, P0_pos, P0_vel, P0_vel, P0_vel])

    master_rng = np.random.default_rng(MASTER_SEED)

    # ── 生成配对种子数据 ──────────────────────────────────────────────
    seeds_data = []
    for seed_idx in range(N_SEEDS):
        rng_seed = master_rng.integers(0, 2**31 - 1)
        nu0 = float(master_rng.uniform(0, 2 * np.pi))
        # 随机相对状态（距离 ~30 km）
        direction = master_rng.normal(0, 1, 3)
        direction /= np.linalg.norm(direction)
        dist = float(master_rng.uniform(20, 40))
        pos_rel = direction * dist  # km
        vel_rel = master_rng.normal(0, 0.01, 3)  # ~10 m/s
        x_rel0 = np.concatenate([pos_rel, vel_rel])
        X_p0 = np.concatenate([pos_rel, vel_rel])
        X_e0 = np.zeros(6)
        # 随机控制
        u_p0 = master_rng.normal(0, 1e-4, 3)  # ~0.1 mm/s²
        u_e0 = np.zeros(3)
        seeds_data.append({
            "seed_idx": seed_idx,
            "rng_seed": rng_seed,
            "nu0": nu0,
            "X_p0": X_p0,
            "X_e0": X_e0,
            "x_rel0": x_rel0,
            "u_p0": u_p0,
            "u_e0": u_e0,
        })

    # ── 总行数估算 ─────────────────────────────────────────────────────
    n_configs = (len(DT_VALUES) * (1  # SDC
                  + len(JAC_EPS_VALUES)  # Jacobian
                  + len(UKF_PARAM_SETS)))  # UKF
    print(f"E0 实验: {N_SEEDS} seeds × {n_configs} configs = {N_SEEDS * n_configs} 次评估")
    print(f"  dt values: {DT_VALUES}")
    print(f"  Jacobian eps: {JAC_EPS_VALUES}")
    print(f"  UKF param sets: {len(UKF_PARAM_SETS)}")

    csv_path = output_dir / "e0_results.csv"
    fieldnames = [
        "seed", "dt", "filter", "params",
        "pred_err_pos_m", "pred_err_vel_m_s",
        "nees", "nees_post", "nis",
        "cov_posdef", "within_3sigma", "angle_innov_max",
        "P_trace", "error",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for dt in DT_VALUES:
            print(f"\n--- dt = {dt} s ---")
            for sdata in seeds_data:
                seed_idx = sdata["seed_idx"]
                X_p0 = sdata["X_p0"]
                X_e0 = sdata["X_e0"]
                x_rel0 = sdata["x_rel0"]
                nu0 = sdata["nu0"]
                u_p0 = sdata["u_p0"]
                u_e0 = sdata["u_e0"]

                # 预生成配对测量噪声；测量真值在传播后的 t_{k+1} 计算。
                pair_rng = np.random.default_rng(sdata["rng_seed"])
                z_noise_paired = pair_rng.multivariate_normal(
                    np.zeros(R_meas.shape[0]), R_meas)

                # ── SDC-EKF ──────────────────────────────────────────
                nav_sdc = SDCEKFNavigator(x0=x_rel0, P0=P0, Q=Q_proc, R=R_meas,
                                          angles_only=True)
                r = _eval_one_step(orb, X_p0, X_e0, nu0, u_p0, u_e0, dt,
                                   nav_sdc, pair_rng, z_noise=z_noise_paired)
                writer.writerow({
                    "seed": seed_idx, "dt": dt, "filter": "sdc",
                    "params": "default",
                    "pred_err_pos_m": r.get("pred_err_pos", np.nan),
                    "pred_err_vel_m_s": r.get("pred_err_vel", np.nan),
                    "nees": r.get("nees", np.nan),
                    "nees_post": r.get("nees_post", np.nan),
                    "nis": r.get("nis", np.nan),
                    "cov_posdef": r.get("cov_posdef", False),
                    "within_3sigma": r.get("within_3sigma", False),
                    "angle_innov_max": r.get("angle_innov_max", np.nan),
                    "P_trace": r.get("P_trace", np.nan),
                    "error": r.get("error", ""),
                })

                # ── Jacobian-EKF（扫描 eps）─────────────────────────
                for eps in JAC_EPS_VALUES:
                    nav_jac = JacobianEKFNavigator(
                        x0=x_rel0, P0=P0, Q=Q_proc, R=R_meas,
                        angles_only=True, eps=eps, state_scales=JAC_STATE_SCALES)
                    r = _eval_one_step(orb, X_p0, X_e0, nu0, u_p0, u_e0, dt,
                                       nav_jac, pair_rng, z_noise=z_noise_paired)
                    writer.writerow({
                        "seed": seed_idx, "dt": dt, "filter": "jacobian",
                        "params": f"eps={eps}",
                        "pred_err_pos_m": r.get("pred_err_pos", np.nan),
                        "pred_err_vel_m_s": r.get("pred_err_vel", np.nan),
                        "nees": r.get("nees", np.nan),
                        "nees_post": r.get("nees_post", np.nan),
                        "nis": r.get("nis", np.nan),
                        "cov_posdef": r.get("cov_posdef", False),
                        "within_3sigma": r.get("within_3sigma", False),
                        "angle_innov_max": r.get("angle_innov_max", np.nan),
                        "P_trace": r.get("P_trace", np.nan),
                        "error": r.get("error", ""),
                    })

                # ── UKF（扫描参数）─────────────────────────────────
                for ukf_p in UKF_PARAM_SETS:
                    try:
                        nav_ukf = SquareRootUKFNavigator(
                            x0=x_rel0, P0=P0, Q=Q_proc, R=R_meas,
                            angles_only=True, **ukf_p)
                    except Exception as e:
                        writer.writerow({
                            "seed": seed_idx, "dt": dt, "filter": "ukf",
                            "params": str(ukf_p), "error": f"init: {e}",
                        })
                        continue
                    r = _eval_one_step(orb, X_p0, X_e0, nu0, u_p0, u_e0, dt,
                                       nav_ukf, pair_rng, z_noise=z_noise_paired)
                    writer.writerow({
                        "seed": seed_idx, "dt": dt, "filter": "ukf",
                        "params": str(ukf_p),
                        "pred_err_pos_m": r.get("pred_err_pos", np.nan),
                        "pred_err_vel_m_s": r.get("pred_err_vel", np.nan),
                        "nees": r.get("nees", np.nan),
                        "nees_post": r.get("nees_post", np.nan),
                        "nis": r.get("nis", np.nan),
                        "cov_posdef": r.get("cov_posdef", False),
                        "within_3sigma": r.get("within_3sigma", False),
                        "angle_innov_max": r.get("angle_innov_max", np.nan),
                        "P_trace": r.get("P_trace", np.nan),
                        "error": r.get("error", ""),
                    })

            print(f"  dt={dt}: {N_SEEDS} seeds complete")

    print(f"\nE0 结果已保存至 {csv_path}")
    return csv_path


if __name__ == "__main__":
    run_e0()
