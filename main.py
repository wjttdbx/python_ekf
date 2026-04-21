"""
NERM + EKF + SDRE 航天器追逃博弈仿真 — 主入口

追踪星初始状态: X_p(0) = [500, 500, 500, 0.01, 0.01, 0.01]^T  (km, km/s)
逃逸星初始状态: X_e(0) = [0, 0, 0, 0, 0, 0]^T
测量噪声: 距离 ~10 m, 角度 ~1e-4 deg
"""


import numpy as np

from aerospace.dynamics.nerm import OrbitalDynamics
from aerospace.control.sdre import SDREGameController
from aerospace.estimation.ekf import RelativeStateEKF
from aerospace.simulation.nerm_ekf_sdre import EKFSDRESimulation
from aerospace.visualization.ekf_plots import plot_single_simulation, plot_comparison


# ─── 常量 ────────────────────────────────────────────────────────────────────
DEG2RAD = np.pi / 180.0


def create_ekf(noisy: bool, x0: np.ndarray, initial_dist: float = 866.0) -> RelativeStateEKF:
    """创建 EKF 实例。

    Parameters
    ----------
    noisy : bool  是否添加测量噪声
    x0 : (6,) ndarray  初始相对状态估计
    initial_dist : float  初始相对距离 (km)，用于计算初始协方差

    Returns
    -------
    RelativeStateEKF
    """
    # 仅测角传感器，精度 0.008°
    sigma_ang = (0.008 * DEG2RAD) if noisy else 0.0  # 0.008 deg → rad

    # 测量噪声：距离项设为极大值（退化为纯测角）
    R_meas = np.diag([
        1e10,  # 距离测量不可信
        max(sigma_ang**2, 1e-30),
        max(sigma_ang**2, 1e-30)
    ])

    # 过程噪声（保持原值）
    Q_proc = np.diag([5e-4, 5e-4, 5e-4, 5e-8, 5e-8, 5e-8]) if noisy else np.zeros((6, 6))

    # 初始协方差：位置误差 = 初始距离 × σ_ang，速度误差 = 1 km/s × σ_ang
    if noisy:
        sigma_pos = initial_dist * sigma_ang  # km
        sigma_vel = 1.0 * sigma_ang           # km/s
        P0 = np.diag([sigma_pos**2, sigma_pos**2, sigma_pos**2,
                      sigma_vel**2, sigma_vel**2, sigma_vel**2])
    else:
        P0 = np.diag([1.0, 1.0, 1.0, 1e-4, 1e-4, 1e-4])

    return RelativeStateEKF(x0=x0, P0=P0, Q=Q_proc, R=R_meas)


def run_simulation(noisy: bool = True):
    """运行单次仿真。

    Parameters
    ----------
    noisy : bool  是否添加测量噪声

    Returns
    -------
    EKFSDRESimResult
    """
    # 轨道动力学
    orb = OrbitalDynamics(mu=3.986e5, a_c=15000.0, e_c=0.5)

    # 初始状态
    X_p0 = np.array([500.0, 500.0, 500.0, 0.01, 0.01, 0.01])
    X_e0 = np.array([0.0,   0.0,   0.0,   0.0,  0.0,  0.0 ])
    nu0  = 0.0

    # SDRE 控制器
    Q_ctrl = np.eye(6)
    R_ctrl = np.eye(3) * 1e13
    ctrl = SDREGameController(Q=Q_ctrl, R=R_ctrl, gamma=np.sqrt(2))

    # EKF 估计器
    x0_est = X_p0 - X_e0  # 初始估计 = 真值（无先验误差）
    initial_dist = float(np.linalg.norm(x0_est[:3]))
    ekf = create_ekf(noisy=noisy, x0=x0_est, initial_dist=initial_dist)

    # 随机数生成器（有噪声时使用）
    rng = np.random.default_rng(42) if noisy else None

    # 仿真器
    dt = 10.0  # 时间步长 (s)
    sim = EKFSDRESimulation(
        dynamics=orb,
        controller=ctrl,
        ekf=ekf,
        X_p0=X_p0,
        X_e0=X_e0,
        nu0=nu0,
        dt=dt,
        are_interval=1,  # 每步求解 ARE
        rng=rng
    )

    # 运行仿真（10 个轨道周期，与 main_nerm_sdre.py 对齐）
    result = sim.run(t_end=10.0 * orb.T_orbit)
    return result, orb


if __name__ == "__main__":
    print("=== 有噪声仿真 (EKF) ===")
    result_noisy, orb = run_simulation(noisy=True)
    plot_single_simulation(
        result_noisy, orb,
        title="NERM+EKF+SDRE — With Measurement Noise",
        out_path="nerm_ekf_sdre_trajectory.png"
    )

    print("\n=== 全知仿真 (理想) ===")
    result_clean, _ = run_simulation(noisy=False)
    plot_single_simulation(
        result_clean, orb,
        title="NERM+SDRE — Ideal (No Noise)",
        out_path="nerm_sdre_ideal.png"
    )

    print("\n=== 生成对比图 ===")
    plot_comparison(result_noisy, result_clean, orb, out_path="nerm_comparison.png")
