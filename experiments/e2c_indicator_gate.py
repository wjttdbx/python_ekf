"""
E2c-A: 仅测角 Flow-Jacobian EKF 一致性失配的在线指标门控

逐步记录离线标签（NEES）和在线候选指标（NIS、线性化差距、P 条件数）。
原子写、可恢复、不覆盖 E2/E2b。
"""

from __future__ import annotations
import argparse, json, os, sys, warnings, time
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.stats import chi2

from aerospace.dynamics.nerm import OrbitalDynamics
from aerospace.control.sdre import SDREGameController
from aerospace.simulation.nerm_ekf_sdre import EKFSDRESimulation
from aerospace.estimation.navigators import (
    JacobianEKFNavigator, SquareRootUKFNavigator,
    _sqrt_of_semidef, _circular_mean, _wrap_angle_scalar, _ensure_symmetric,
)
from aerospace.estimation.ekf import RelativeStateEKF
from aerospace.paths import DATA_DIR

warnings.filterwarnings("ignore")
DEG2RAD = np.pi / 180.0

from experiments.e2_flow_isolated_n10 import (
    make_initial_state, los_p0, generate_x0_errors, FixedNoiseRNG,
    generate_paired_noise_sequences, seed_from_str,
    MU, A_C, E_C,
    SIGMA_THETA_VALUES, GEOMETRY_LABELS, N_SEEDS, MASTER_SEED,
    T_ORBIT_FACTOR, CAPTURE_DIST, V_THRESH, SUSTAIN_DUR,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 标签与指标参数
# ═══════════════════════════════════════════════════════════════════════════════

CHI2_NEES_99 = chi2.ppf(0.99, 6)       # 16.81
CHI2_NIS_99 = chi2.ppf(0.99, 2)         # 9.21
LABEL_CONSECUTIVE = 10   # 连续超出步数，确认失配
NIS_WINDOW = 20           # NIS 滑动窗口步数
LINEARIZATION_WINDOW = 20
UKF_ALPHA = 1.0
UKF_BETA = 2.0
UKF_KAPPA = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 在线指标计算工具
# ═══════════════════════════════════════════════════════════════════════════════

def _ukf_sigma_points(x: np.ndarray, S_upper: np.ndarray,
                       alpha: float, beta: float, kappa: float
                       ) -> tuple[np.ndarray, float, float, np.ndarray]:
    """生成 UKF sigma 点及权重。S_upper 为 P 的上三角 Cholesky 因子。"""
    n = len(x)
    lam = alpha**2 * (n + kappa) - n
    gamma = np.sqrt(n + lam)
    Wm0 = lam / (n + lam)
    Wc0 = lam / (n + lam) + (1.0 - alpha**2 + beta)
    Wi = 1.0 / (2.0 * (n + lam))

    chi = np.zeros((2 * n + 1, n))
    chi[0] = x
    for i in range(n):
        chi[i + 1] = x + gamma * S_upper[i, :]
        chi[i + 1 + n] = x - gamma * S_upper[i, :]
    return chi, Wm0, Wi, np.array([Wc0] + [Wi] * (2 * n))


def _compute_linearization_gaps(x_prior: np.ndarray, P_prior: np.ndarray,
                                 R_meas: np.ndarray,
                                 ) -> dict:
    """从 EKF 先验状态计算 UKF-vs-EKF 测量线性化差距。

    不修改 EKF 状态，纯粹作为在线诊断指标。
    """
    n = 6
    m = R_meas.shape[0]
    result = {}

    # ── EKF 测量预测 ──────────────────────────────────────────────
    H = RelativeStateEKF.meas_jacobian(x_prior, angle_only=True)
    z_ekf = np.array([
        np.arctan2(x_prior[1], x_prior[0]),
        np.arcsin(np.clip(x_prior[2] / (np.linalg.norm(x_prior[:3]) + 1e-12), -1, 1)),
    ])
    S_ekf = H @ P_prior @ H.T + R_meas

    # ── UKF 测量预测 ──────────────────────────────────────────────
    try:
        from scipy.linalg import cholesky
        S_prior = cholesky(P_prior, lower=False)
    except Exception:
        S_prior = _sqrt_of_semidef(P_prior)

    chi_prior, Wm0, Wi, Wc = _ukf_sigma_points(
        x_prior, S_prior, UKF_ALPHA, UKF_BETA, UKF_KAPPA)

    zeta = np.zeros((2 * n + 1, m))
    for i in range(2 * n + 1):
        x_rel = chi_prior[i]
        rho = np.linalg.norm(x_rel[:3]) + 1e-12
        az = np.arctan2(x_rel[1], x_rel[0])
        el = np.arcsin(np.clip(x_rel[2] / rho, -1, 1))
        zeta[i] = np.array([az, el])

    # 圆周角加权均值
    z_ukf = np.zeros(m)
    z_ukf[0] = float(np.arctan2(
        Wm0 * np.sin(zeta[0, 0]) + Wi * np.sum(np.sin(zeta[1:, 0])),
        Wm0 * np.cos(zeta[0, 0]) + Wi * np.sum(np.cos(zeta[1:, 0]))))
    z_ukf[1] = Wm0 * zeta[0, 1] + Wi * np.sum(zeta[1:, 1])

    # Wrap sigma points for covariance
    zeta_w = zeta.copy()
    for s in range(2 * n + 1):
        zeta_w[s, 0] = _wrap_angle_scalar(zeta[s, 0] - z_ukf[0]) + z_ukf[0]

    # UKF 测量协方差
    S_ukf = np.zeros((m, m))
    S_ukf += Wc[0] * np.outer(zeta_w[0] - z_ukf, zeta_w[0] - z_ukf)
    for i in range(1, 2 * n + 1):
        S_ukf += Wi * np.outer(zeta_w[i] - z_ukf, zeta_w[i] - z_ukf)

    # ── 归一化差距 ────────────────────────────────────────────────
    trace_S_ekf = max(np.trace(S_ekf), 1e-30)
    frob_S_ekf = max(np.linalg.norm(S_ekf, 'fro'), 1e-30)

    result["lin_gap_mean"] = float(np.linalg.norm(z_ukf - z_ekf) / np.sqrt(trace_S_ekf))
    result["lin_gap_cov"] = float(np.linalg.norm(S_ukf - S_ekf, 'fro') / frob_S_ekf)

    # 额外：预测测量范数差异（带符号的方位角差异）
    az_diff = _wrap_angle_scalar(float(z_ukf[0] - z_ekf[0]))
    result["lin_gap_az"] = float(abs(az_diff) / np.sqrt(max(S_ekf[0, 0], 1e-30)))
    result["lin_gap_el"] = float(abs(float(z_ukf[1] - z_ekf[1])) / np.sqrt(max(S_ekf[1, 1], 1e-30)))

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 单条轨迹逐步采集
# ═══════════════════════════════════════════════════════════════════════════════

def _run_diagnostic_trajectory(job: dict) -> dict:
    """运行单条 Jacobian EKF 轨迹，逐步记录标签和候选指标。"""
    geom = job["geom"]
    seed = job["seed"]
    sigma_deg = job["sigma_deg"]
    sigma_rad = job["sigma_rad"]
    noise_seq = np.array(job["noise_seq"])
    P0 = np.array(job["P0"])
    x0_err = np.array(job["x0_err"])
    X_p0 = np.array(job["X_p0"])
    X_e0 = np.array(job["X_e0"])
    nu0 = job["nu0"]
    dt = job["dt"]
    t_end = job["t_end"]

    sigma_ang_eff = max(sigma_rad, 1e-12)
    R_meas = np.diag([sigma_ang_eff ** 2, sigma_ang_eff ** 2])
    Q_proc = np.zeros((6, 6))

    x_est0 = (X_p0 - X_e0) + x0_err
    nav = JacobianEKFNavigator(x0=x_est0, P0=P0, Q=Q_proc, R=R_meas,
                                angles_only=True)

    orb = OrbitalDynamics(mu=MU, a_c=A_C, e_c=E_C)
    ctrl = SDREGameController(Q=np.eye(6), R=np.eye(3) * 1e13, gamma=np.sqrt(2))

    n_steps_expected = int(t_end / dt) + 2
    noise_scaled = np.zeros((n_steps_expected, R_meas.shape[0]))
    for i in range(min(len(noise_seq), n_steps_expected)):
        noise_scaled[i] = noise_seq[i] * sigma_ang_eff
    rng = FixedNoiseRNG(noise_scaled, R_meas.shape[0])

    # ── 状态初始化 ──────────────────────────────────────────────
    state = np.zeros(13)
    state[0:6] = X_p0
    state[6:12] = X_e0
    state[12] = nu0
    t = 0.0
    N = int(t_end / dt)
    B_ctrl = np.zeros((6, 3))
    B_ctrl[3:, :] = np.eye(3)

    # ── 逐步记录缓冲区 ──────────────────────────────────────────
    log = {
        "t": np.zeros(N + 1),
        "nis": np.full(N + 1, np.nan),
        "nees": np.full(N + 1, np.nan),
        "P_eig_min": np.full(N + 1, np.nan),
        "P_cond": np.full(N + 1, np.nan),
        "lin_gap_mean": np.full(N + 1, np.nan),
        "lin_gap_cov": np.full(N + 1, np.nan),
        "lin_gap_az": np.full(N + 1, np.nan),
        "lin_gap_el": np.full(N + 1, np.nan),
        "innov_norm": np.full(N + 1, np.nan),
        "dist_km": np.full(N + 1, np.nan),
        "err_pos_m": np.full(N + 1, np.nan),
        "err_vel_ms": np.full(N + 1, np.nan),
    }

    # ── Step 0 ──────────────────────────────────────────────────
    x_true_rel0 = state[0:6] - state[6:12]
    err0 = nav.x - x_true_rel0
    try:
        nees0 = float(err0 @ np.linalg.solve(nav.P, err0))
    except np.linalg.LinAlgError:
        nees0 = np.nan
    log["t"][0] = 0.0
    log["nees"][0] = nees0
    log["err_pos_m"][0] = float(np.linalg.norm(err0[:3])) * 1000.0
    log["err_vel_ms"][0] = float(np.linalg.norm(err0[3:])) * 1000.0
    log["dist_km"][0] = float(np.linalg.norm(x_true_rel0[:3]))
    eigvals = np.linalg.eigvalsh(nav.P)
    log["P_eig_min"][0] = float(np.min(eigvals))
    log["P_cond"][0] = float(np.max(eigvals) / max(np.min(eigvals), 1e-300))

    # ── 主循环 ──────────────────────────────────────────────────
    for k in range(N):
        nu = float(state[12])
        r_c, nu_dot, nu_ddot = orb.get_orbital_params(nu)

        # ── 控制 ───────────────────────────────────────────────
        x_ctrl = nav.x
        X_p_true = state[0:6]
        X_e_est = X_p_true - nav.x
        A_SDC_ctrl = orb.get_SDC_matrix(X_p_true, X_e_est, r_c, nu_dot, nu_ddot)
        u_p, u_e = ctrl.compute_control(A_SDC_ctrl, x_ctrl, t=t, solve_are=(k % 1 == 0))
        u_e = np.zeros(3)
        du = u_p - u_e

        # ── 预测 ────────────────────────────────────────────────
        def flow_discrete(x_rel):
            est_state = np.concatenate([X_p_true, X_p_true - x_rel, [nu]])
            dt_local = dt
            k1 = orb.dynamics_13d(0.0, est_state, u_p, u_e)
            k2 = orb.dynamics_13d(0.5 * dt_local, est_state + 0.5 * dt_local * k1, u_p, u_e)
            k3 = orb.dynamics_13d(0.5 * dt_local, est_state + 0.5 * dt_local * k2, u_p, u_e)
            k4 = orb.dynamics_13d(dt_local, est_state + dt_local * k3, u_p, u_e)
            prop = est_state + (dt_local / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            return prop[0:6] - prop[6:12]

        def A_SDC_fn(x_rel):
            X_e_est_local = X_p_true - x_rel
            return orb.get_SDC_matrix(X_p_true, X_e_est_local, r_c, nu_dot, nu_ddot)

        A_SDC_nominal = A_SDC_fn(nav.x)
        x_priori, P_priori = nav.predict(A_SDC_nominal, B_ctrl, u_p, u_e, dt,
                                          f_discrete=flow_discrete, A_SDC_fn=A_SDC_fn)

        # ── 线性化差距（在线指标，只用先验和测量模型）─────────
        lin_gaps = _compute_linearization_gaps(x_priori, P_priori, R_meas)

        # ── 真实状态传播 ────────────────────────────────────────
        from scipy.integrate import solve_ivp
        sol = solve_ivp(orb.dynamics_13d, [t, t + dt], state,
                        args=(u_p, u_e), method="RK45", rtol=1e-8, atol=1e-10)
        state = sol.y[:, -1]
        t += dt

        # ── 测量 + 更新 ──────────────────────────────────────────
        z_true = np.array([
            np.arctan2(state[1] - state[7], state[0] - state[6]),
            np.arcsin(np.clip((state[2] - state[8]) /
                     (np.linalg.norm(state[0:3] - state[6:9]) + 1e-12), -1, 1)),
        ])
        z_meas = z_true + rng.multivariate_normal(np.zeros(2), R_meas)
        innov = nav.update(x_priori, P_priori, z_meas)

        # ── 标签与指标 ──────────────────────────────────────────
        S = getattr(nav, '_last_S', None)
        nis = np.nan
        if S is not None:
            try:
                nis = float(innov @ np.linalg.solve(S, innov))
            except np.linalg.LinAlgError:
                pass

        x_true_rel = state[0:6] - state[6:12]
        err = nav.x - x_true_rel
        nees = np.nan
        try:
            nees = float(err @ np.linalg.solve(nav.P, err))
        except np.linalg.LinAlgError:
            pass

        eigvals = np.linalg.eigvalsh(nav.P)
        P_eig_min = float(np.min(eigvals))
        P_cond = float(np.max(eigvals) / max(P_eig_min, 1e-300))

        idx = k + 1
        log["t"][idx] = t
        log["nis"][idx] = nis
        log["nees"][idx] = nees
        log["P_eig_min"][idx] = P_eig_min
        log["P_cond"][idx] = P_cond
        log["lin_gap_mean"][idx] = lin_gaps["lin_gap_mean"]
        log["lin_gap_cov"][idx] = lin_gaps["lin_gap_cov"]
        log["lin_gap_az"][idx] = lin_gaps["lin_gap_az"]
        log["lin_gap_el"][idx] = lin_gaps["lin_gap_el"]
        log["innov_norm"][idx] = float(np.linalg.norm(innov))
        log["err_pos_m"][idx] = float(np.linalg.norm(err[:3])) * 1000.0
        log["err_vel_ms"][idx] = float(np.linalg.norm(err[3:])) * 1000.0
        log["dist_km"][idx] = float(np.linalg.norm(x_true_rel[:3]))

    return {
        "geom": geom, "seed": seed, "sigma_deg": sigma_deg,
        "success": True,
    }, {k: v for k, v in log.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# Worker (module-level for pickle)
# ═══════════════════════════════════════════════════════════════════════════════

def _pool_init() -> None:
    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[var] = "1"


# ═══════════════════════════════════════════════════════════════════════════════
# 主运行器
# ═══════════════════════════════════════════════════════════════════════════════

def run_e2c_collection(output_dir: Path, seeds_override: int | None = None,
                        geoms_override: list[str] | None = None,
                        n_workers: int | None = None):
    if n_workers is None:
        n_workers = min(cpu_count(), 8)
    n_seeds_eff = seeds_override if seeds_override is not None else N_SEEDS
    geoms = geoms_override if geoms_override is not None else GEOMETRY_LABELS

    print("=" * 60)
    print("E2c-A: Online indicator data collection")
    print(f"Geometries: {geoms}, Seeds: {n_seeds_eff}")
    print(f"Noise levels: {SIGMA_THETA_VALUES}")
    print(f"Workers: {n_workers}")
    print("=" * 60)

    orb = OrbitalDynamics(mu=MU, a_c=A_C, e_c=E_C)
    t_end = T_ORBIT_FACTOR * orb.T_orbit
    dt = 10.0
    n_steps = int(t_end / dt) + 2
    m_dim = 2
    noise_seqs = generate_paired_noise_sequences(n_seeds_eff, n_steps, m_dim)

    # ── 原子输出目录 ──────────────────────────────────────────
    atoms_dir = output_dir / "_atoms"
    atoms_dir.mkdir(parents=True, exist_ok=True)

    # ── 构建 job 列表 ──────────────────────────────────────────
    jobs = []
    for geom in geoms:
        X_p0, X_e0 = make_initial_state(geom, dist_km=30.0)
        P0 = los_p0(X_p0 - X_e0)
        x0_errs = generate_x0_errors(n_seeds_eff, P0, seed_offset=seed_from_str(geom))
        for seed in range(n_seeds_eff):
            for sigma_deg in SIGMA_THETA_VALUES:
                atom_path = atoms_dir / f"{geom}_seed{seed}_sigma{sigma_deg}.npz"
                if atom_path.exists():
                    continue
                sigma_rad = sigma_deg * DEG2RAD
                jobs.append({
                    "geom": geom, "seed": seed,
                    "sigma_deg": sigma_deg, "sigma_rad": sigma_rad,
                    "noise_seq": noise_seqs[seed].tolist(),
                    "P0": P0.tolist(), "x0_err": x0_errs[seed].tolist(),
                    "X_p0": X_p0.tolist(), "X_e0": X_e0.tolist(),
                    "nu0": 0.0, "dt": dt, "t_end": t_end,
                    "_atom_path": str(atom_path),
                })

    total_jobs = len(geoms) * len(SIGMA_THETA_VALUES) * n_seeds_eff
    already_done = total_jobs - len(jobs)
    if already_done > 0:
        print(f"Resuming: {already_done}/{total_jobs} already complete")
    print(f"Queued: {len(jobs)}/{total_jobs}")

    if not jobs:
        print("All jobs complete, skipping collection.")
        return

    with Pool(processes=n_workers, initializer=_pool_init) as pool:
        for i, (meta, log) in enumerate(pool.imap_unordered(_run_diagnostic_trajectory, jobs)):
            ap = atoms_dir / f"{meta['geom']}_seed{meta['seed']}_sigma{meta['sigma_deg']}.npz"
            np.savez_compressed(ap, **log)
            count = already_done + i + 1
            if count % 10 == 0 or count == total_jobs:
                print(f"  [{count}/{total_jobs}] {meta['geom']} "
                      f"sigma={meta['sigma_deg']} seed={meta['seed']}",
                      flush=True)

    print(f"\nCollection complete: {total_jobs} trajectories saved to {atoms_dir}")


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--geom", type=str, default=None,
                        help="Comma-separated geometry filter")
    parser.add_argument("--seeds", type=int, default=None,
                        help="Override seed count")
    args = parser.parse_args()

    output_dir = DATA_DIR / "e2c_indicator_gate"
    output_dir.mkdir(parents=True, exist_ok=True)

    geoms_override = args.geom.split(",") if args.geom else None
    seeds_eff = args.seeds if args.seeds is not None else (1 if args.smoke else N_SEEDS)

    if args.smoke:
        geoms_override = ["crosstrack", "radial"]
        seeds_eff = 2
        print(f"SMOKE MODE: {geoms_override}, seeds 0-1")

    run_e2c_collection(output_dir, seeds_override=seeds_eff,
                        geoms_override=geoms_override,
                        n_workers=args.workers)
