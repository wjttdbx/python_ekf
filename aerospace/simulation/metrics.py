"""
全时域仿真后处理指标提取

对完整历史轨迹（无早停）做事件检测，提取：
  T_FP, v_FP, T_soft, T_sustain, 驻留成功标志,
  出球/再入次数, ΔV, 控制能量, 估计误差统计。
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.stats import chi2


@dataclass
class CaptureMetrics:
    """全时域捕获与软着陆指标。

    所有时间单位为秒，所有 "None" 表示对应事件在仿真时长内未发生。
    """
    # ── 首次穿入 ──
    T_FP: float | None = None          # 首次进入 capture_dist 的时间 (s)
    v_FP: float | None = None          # T_FP 时刻的相对速度范数 (km/s)
    radial_closure_rate_FP: float | None = None  # T_FP 时刻径向闭合速度 (km/s)

    # ── 软到达 ──
    T_soft: float | None = None        # 首次同时满足 ρ≤capture_dist 且 ||v||≤v_thresh
    v_at_soft: float | None = None     # T_soft 时刻的相对速度范数

    # ── 持续驻留 ──
    T_sustain: float | None = None     # 首次连续满足 ρ≤capture_dist 且 ||v||≤v_thresh
                                       #   持续时间 ≥ sustain_dur 的起始时刻
    sustained: bool = False            # 是否在仿真时间内达到持续驻留
    sustain_end: float | None = None   # 持续驻留区间结束时间（首次 violation 或仿真结束）

    # ── 球内行为（首次进入 capture_dist 后）────────────────────────
    max_dist_after_FP: float | None = None   # 首次进入后的最大距离 (km)
    min_dist_after_FP: float | None = None   # 首次进入后的最小距离 (km)
    exit_count: int = 0                       # 出球 (>capture_dist) 后再次进入的次数
    time_in_sphere_frac: float = 0.0          # 首次进入后待在球内的时间占比

    # ── 控制预算（全时域）───────────────────────────────────────────
    total_delta_v: float = 0.0          # 累计 ΔV = ∫||u_p|| dt (km/s)
    peak_accel: float = 0.0             # 峰值加速度 ||u_p|| (km/s²)
    control_energy: float = 0.0         # 控制能量积分 ∫||u_p||² dt (km²/s³)

    # ── 估计精度（全时域）───────────────────────────────────────────
    rmse_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))   # (3,) m
    rmse_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))   # (3,) m/s
    rmse_dist: float = 0.0              # ||Δr|| RMSE (m)
    final_pos_err: float = 0.0          # 终点 ||Δr|| (m)
    final_vel_err: float = 0.0          # 终点 ||Δv|| (m/s)
    est_dist_bias: float = 0.0          # 估计距离偏差均值（带符号，km）
    est_radial_vel_bias: float = 0.0    # 估计径向速度偏差均值（带符号，km/s）

    # ── 滤波器一致性 ────────────────────────────────────────────────
    nis_mean: float | None = None       # 平均 NIS
    nis_inlier_frac: float | None = None  # 3σ 卡方界内占比
    nees_mean: float | None = None      # 平均 NEES（仅当有真值时）
    nees_inlier_frac: float | None = None

    # ── 控制器数值诊断 ──────────────────────────────────────────────
    are_fallback_count: int = 0
    controller_mean_step_ms: float = 0.0


def compute_metrics(result, dt: float,
                    capture_dist: float = 0.1,
                    v_thresh: float = 1e-4,   # 0.1 m/s = 1e-4 km/s
                    sustain_dur: float = 600.0,
                    compute_consistency: bool = True) -> CaptureMetrics:
    """从全时域仿真结果提取所有捕获/软着陆指标。

    Parameters
    ----------
    result : EKFSDRESimResult  全时域仿真结果（early_stop=False）
    dt : float                 仿真步长 (s)
    capture_dist : float       距离阈值 (km)，默认 0.1 km = 100 m
    v_thresh : float           速度阈值 (km/s)，默认 1e-4 km/s = 0.1 m/s
    sustain_dur : float        连续驻留要求 (s)，默认 600 s
    compute_consistency : bool 是否计算 NIS/NEES

    Returns
    -------
    CaptureMetrics
    """
    m = CaptureMetrics()
    dist = result.dist_history          # (N,) km
    t_arr = result.t                    # (N,) s
    u_p = result.u_p_history            # (3, N) km/s²
    err = result.ekf_err_history        # (6, N) km, km/s
    n_steps = len(t_arr)

    if n_steps < 2:
        return m

    # ── 控制预算 ────────────────────────────────────────────────────
    u_norm = np.linalg.norm(u_p, axis=0)
    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    m.total_delta_v = float(_trapz(u_norm, t_arr))
    m.peak_accel = float(np.max(u_norm))
    m.control_energy = float(_trapz(u_norm**2, t_arr))

    # ── 估计精度 ────────────────────────────────────────────────────
    pos_err_m = err[:3, :] * 1000.0     # m
    vel_err_ms = err[3:, :] * 1000.0    # m/s → 实际 err 是 km/s，*1000 = m/s
    m.rmse_pos = np.sqrt(np.mean(pos_err_m**2, axis=1))
    m.rmse_vel = np.sqrt(np.mean(vel_err_ms**2, axis=1))
    m.rmse_dist = float(np.sqrt(np.mean(np.sum(pos_err_m**2, axis=0))))

    # 终点误差
    m.final_pos_err = float(np.linalg.norm(pos_err_m[:, -1]))
    m.final_vel_err = float(np.linalg.norm(vel_err_ms[:, -1]))

    # 估计偏差（带符号）
    est_dist = np.array([np.linalg.norm(result.x_est_history[:3, i])
                         for i in range(n_steps)])
    true_rel_pos_norm = np.linalg.norm(
        result.states[0:3, :] - result.states[6:9, :], axis=0)
    m.est_dist_bias = float(np.mean(est_dist - true_rel_pos_norm))
    # 径向速度偏差
    true_rel_vel = result.states[3:6, :] - result.states[9:12, :]
    radial_unit = np.zeros_like(true_rel_vel)
    for i in range(n_steps):
        r_vec = result.states[0:3, i] - result.states[6:9, i]
        r_norm = np.linalg.norm(r_vec)
        if r_norm > 1e-12:
            radial_unit[:, i] = r_vec / r_norm
    true_radial_vel = np.sum(true_rel_vel * radial_unit, axis=0)
    est_radial_vel = np.array([
        float(np.dot(result.x_est_history[3:, i], radial_unit[:, i]))
        for i in range(n_steps)])
    m.est_radial_vel_bias = float(np.mean(est_radial_vel - true_radial_vel))

    # 一致性统计与是否进入终端球无关，必须对失败/删失试验同样计算。
    if compute_consistency:
        nis_hist = getattr(result, 'nis_history', None)
        if nis_hist is not None and len(nis_hist) > 0 and not np.all(np.isnan(nis_hist)):
            valid_nis = nis_hist[~np.isnan(nis_hist)]
            m.nis_mean = float(np.mean(valid_nis))
            m_dim = result.innov_history.shape[0] if hasattr(result, 'innov_history') else 2
            threshold = chi2.ppf(0.997, m_dim)
            m.nis_inlier_frac = float(np.mean(valid_nis < threshold))

        nees_hist = getattr(result, 'nees_history', None)
        if nees_hist is not None and len(nees_hist) > 0 and not np.all(np.isnan(nees_hist)):
            valid_nees = nees_hist[~np.isnan(nees_hist)]
            m.nees_mean = float(np.mean(valid_nees))
            threshold = chi2.ppf(0.997, 6)
            m.nees_inlier_frac = float(np.mean(valid_nees < threshold))

    # ── 首次穿入 (T_FP) ─────────────────────────────────────────────
    fp_mask = dist <= capture_dist
    fp_indices = np.where(fp_mask)[0]
    if len(fp_indices) == 0:
        return m  # 从未进入球内

    i_fp = fp_indices[0]
    m.T_FP = float(t_arr[i_fp])
    rel_vel_fp = result.states[3:6, i_fp] - result.states[9:12, i_fp]
    m.v_FP = float(np.linalg.norm(rel_vel_fp))
    r_vec_fp = result.states[0:3, i_fp] - result.states[6:9, i_fp]
    r_norm_fp = np.linalg.norm(r_vec_fp)
    if r_norm_fp > 1e-12:
        m.radial_closure_rate_FP = -float(np.dot(rel_vel_fp, r_vec_fp) / r_norm_fp)

    # ── 球内行为 ────────────────────────────────────────────────────
    post_fp = slice(i_fp, n_steps)
    dist_post = dist[post_fp]
    if len(dist_post) > 0:
        m.max_dist_after_FP = float(np.max(dist_post))
        m.min_dist_after_FP = float(np.min(dist_post))

    # 出球再入次数
    inside = dist <= capture_dist
    transitions = np.diff(inside.astype(int))
    # 进入: 0→1, 离开: 1→0
    entries = np.sum(transitions == 1)
    m.exit_count = max(0, int(entries - 1))  # 首次进入不计为"再入"

    # 球内时间占比
    if n_steps > i_fp:
        m.time_in_sphere_frac = float(np.mean(inside[post_fp]))

    # ── 软到达 (T_soft) ─────────────────────────────────────────────
    vel_norm = np.array([np.linalg.norm(
        result.states[3:6, i] - result.states[9:12, i]) for i in range(n_steps)])
    soft_mask = (dist <= capture_dist) & (vel_norm <= v_thresh)
    soft_indices = np.where(soft_mask)[0]
    if len(soft_indices) > 0:
        i_soft = soft_indices[0]
        m.T_soft = float(t_arr[i_soft])
        m.v_at_soft = float(vel_norm[i_soft])

    # ── 持续驻留 (T_sustain) ────────────────────────────────────────
    if len(soft_indices) > 0:
        # 按连续采样段检查真实时间跨度，避免 N 个点仅覆盖 (N-1)dt。
        run_start = int(soft_indices[0])
        run_end = run_start
        soft_runs: list[tuple[int, int]] = []
        for sample_index in soft_indices[1:]:
            sample_index = int(sample_index)
            if sample_index == run_end + 1:
                run_end = sample_index
            else:
                soft_runs.append((run_start, run_end))
                run_start = sample_index
                run_end = sample_index
        soft_runs.append((run_start, run_end))

        for run_start, run_end in soft_runs:
            if t_arr[run_end] - t_arr[run_start] >= sustain_dur:
                m.sustained = True
                m.T_sustain = float(t_arr[run_start])
                m.sustain_end = float(t_arr[run_end])
                break


    return m
