
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from aerospace.dynamics.nerm import OrbitalDynamics
from aerospace.control.sdre import SDREGameController
from aerospace.estimation.ekf import RelativeStateEKF
from aerospace.simulation.nerm_ekf_sdre import EKFSDRESimulation

# 复用 run_scenarios.py 中的参数
DEG2RAD = np.pi / 180.0
MU = 3.986e5
SIGMA_ANG = 0.008 * DEG2RAD
REF_DIST  = 3000.0
REF_SIGMA_POS = 10.0
REF_SIGMA_VEL = 1e-3
SIGMA_DIST = 1e10  # 改回 1e10 以观察纯测角下的收敛状况
GAMMA = np.sqrt(2)

def create_ekf(x0, initial_dist):
    scale = initial_dist / REF_DIST
    sigma_pos = REF_SIGMA_POS * scale
    sigma_vel = REF_SIGMA_VEL * scale
    P0 = np.diag([sigma_pos**2] * 3 + [sigma_vel**2] * 3)
    R_meas = np.diag([SIGMA_DIST, SIGMA_ANG**2, SIGMA_ANG**2])
    Q_proc = np.diag([5e-4, 5e-4, 5e-4, 5e-8, 5e-8, 5e-8])
    return RelativeStateEKF(x0=x0, P0=P0, Q=Q_proc, R=R_meas)

def analyze_initial_convergence(duration_min=5.0):
    # 模拟场景 1
    a_c, e_c = 6871.0, 0.1
    orb = OrbitalDynamics(mu=MU, a_c=a_c, e_c=e_c)
    ctrl = SDREGameController(Q=np.eye(6), R=np.eye(3) * 1e13, gamma=GAMMA)
    
    # 场景 1 的近似初始状态 (从之前的日志推断)
    # 实际上我们只需要一个相对状态来观察收敛
    x_rel0 = np.array([50.0, 80.0, 0.0, 0.0, 0.0, 0.0]) # 约 100km 初始距离
    initial_dist = np.linalg.norm(x_rel0[:3])
    
    # 生成初始估计误差
    rng = np.random.default_rng(42)
    scale = initial_dist / REF_DIST
    noise = rng.standard_normal(6) * np.array([REF_SIGMA_POS*scale]*3 + [REF_SIGMA_VEL*scale]*3)
    x0_est = x_rel0 + noise
    
    ekf = create_ekf(x0_est, initial_dist)
    
    # 构建仿真 (只跑一小段时间)
    # X_p0 = x_rel0, X_e0 = 0 (简化处理，只看相对收敛)
    sim = EKFSDRESimulation(
        dynamics=orb, controller=ctrl, ekf=ekf,
        X_p0=x_rel0, X_e0=np.zeros(6), nu0=0.0,
        dt=1.0, rng=rng # dt 设小一点看细节
    )
    
    t_end = duration_min * 60.0
    result = sim.run(t_end=t_end)
    
    # 绘图
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # 1. 位置误差
    axes[0].plot(result.t / 60.0, result.ekf_err_history[0] * 1000, label='Error x (Radial)')
    axes[0].plot(result.t / 60.0, result.ekf_err_history[1] * 1000, label='Error y (Along-track)')
    axes[0].set_ylabel('Position Error (m)')
    axes[0].grid(True)
    axes[0].legend()
    axes[0].set_title(f'EKF Initial Convergence (First {duration_min} min)')

    # 2. 速度误差
    axes[1].plot(result.t / 60.0, result.ekf_err_history[3], label='Error vx')
    axes[1].plot(result.t / 60.0, result.ekf_err_history[4], label='Error vy')
    axes[1].set_ylabel('Velocity Error (km/s)')
    axes[1].set_xlabel('Time (min)')
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    out_path = "outputs/figures/initial_convergence_5min.png"
    plt.savefig(out_path)
    print(f"Analysis plot saved to {out_path}")

if __name__ == "__main__":
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    analyze_initial_convergence(5.0)
