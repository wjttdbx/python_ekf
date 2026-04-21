"""
多场景批量仿真脚本

场景1：同平面同轨道
场景2：不同平面同轨道
场景3：同平面不同轨道（低）
场景4：同平面不同轨道（高）
场景5：不同平面不同轨道（高）

5个追逃场景，ECI初始状态转换为LVLH坐标后仿真。
噪声参数按初始距离比例缩放（基准：3000km对应10km位置误差）。
"""
import zhplot
import numpy as np
from pathlib import Path

from aerospace.dynamics.nerm import OrbitalDynamics
from aerospace.control.sdre import SDREGameController
from aerospace.estimation.ekf import RelativeStateEKF
from aerospace.simulation.nerm_ekf_sdre import EKFSDRESimulation
from aerospace.visualization.ekf_plots import plot_single_simulation
from aerospace.shared.coord_transform import keplerian_to_eci, eci_to_lvlh_state

# 常量定义
DEG2RAD = np.pi / 180.0
MU = 3.986e5                    # 地球引力常数 (km^3/s^2)
SIGMA_ANG = 0.008 * DEG2RAD      # 测角精度 (rad)
REF_DIST  = 3000.0              # 基准距离 (km)
REF_SIGMA_POS = 10.0            # 基准位置误差 (km) @ REF_DIST
REF_SIGMA_VEL = 1e-3            # 基准速度误差 (km/s) @ REF_DIST (对应 1 m/s)
SIGMA_DIST = 1e10                   # 协方差 —— 距离测量不可信，设置极大值退化为纯测角
GAMMA = np.sqrt(2)               # 控制逃方机动参数

# ─── 场景定义（ECI 绝对状态，单位：km 和 km/s）──────────────────────────────────
# 参考轨道（chief）统一取目标航天器轨道：a=6871, e=0.1, i=0, Omega=0, omega=0

SCENARIOS = {
    "scenario_1": {
        "name": "场景1：同平面同轨道",
        "chief_orbit": {"a": 6871.0, "e": 0.1, "i": 0.0, "Omega": 0.0, "omega": 0.0},
        "X_e0_eci": np.array([-3284.62, -6329.16,    0.0,      6.79452, -2.76059,  0.0     ]), # 逃逸者初始ECI状态
        "X_p0_eci": np.array([-3377.37, -6290.69,    0.0,      6.74444, -2.85544,  0.0     ]), # 追击者初始ECI状态
    },
    "scenario_2": {
        "name": "场景2：不同平面同轨道",
        "chief_orbit": {"a": 6871.0, "e": 0.1, "i": 0.0, "Omega": 0.0, "omega": 0.0},
        "X_e0_eci": np.array([-7457.23,  1167.26,    0.0,     -1.18375, -6.79734,  0.0     ]),
        "X_p0_eci": np.array([-7439.72,  1263.48,  -22.0542,  -1.28183, -6.7803,   0.118351]),
    },
    "scenario_3": {
        "name": "场景3：同平面不同轨道（低轨道）",
        "chief_orbit": {"a": 6871.0, "e": 0.1, "i": 0.0, "Omega": 0.0, "omega": 0.0},
        "X_e0_eci": np.array([ 719.856,  6691.72,    0.0,     -7.611,    1.58426,  0.0     ]),
        "X_p0_eci": np.array([ 786.174,  6616.76,    0.0,     -7.63541,  1.67614,  0.0     ]),
    },
    "scenario_4": {
        "name": "场景4：同平面不同轨道（高轨道）",
        "chief_orbit": {"a": 6871.0, "e": 0.1, "i": 0.0, "Omega": 0.0, "omega": 0.0},
        "X_e0_eci": np.array([ 3668.07, -5287.73,    0.0,      6.2898,   5.12867,  0.0     ]),
        "X_p0_eci": np.array([ 3637.51, -5383.14,    0.0,      6.31566,  5.02984,  0.0     ]),
    },
    "scenario_5": {
        "name": "场景5：不同平面不同轨道（高轨道）",
        "chief_orbit": {"a": 6871.0, "e": 0.1, "i": 0.0, "Omega": 0.0, "omega": 0.0},
        "X_e0_eci": np.array([-7368.1,  -1596.62,    0.0,      1.62119, -6.7158,   0.0     ]),
        "X_p0_eci": np.array([-7447.48, -1541.82,   26.9125,   1.54551, -6.7007,   0.116961]),
    },
}


def eci_to_lvlh_scenario(cfg: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """
    将场景从 ECI 绝对状态转换为 LVLH 相对坐标。
    返回: (追击者初始LVLH状态, 逃逸者初始LVLH状态, 初始真近点角nu0)
    """
    orb = cfg["chief_orbit"]
    a  = orb["a"]
    e  = orb["e"]
    i  = orb["i"]     * DEG2RAD
    Om = orb["Omega"] * DEG2RAD
    om = orb["omega"] * DEG2RAD

    # 从目标（逃逸者）ECI 位置反推真近点角 nu0 (假设 i=0, Omega=0, omega=0 时可以通过 arctan2(y,x) 获取)
    r_e_eci = cfg["X_e0_eci"][:3]
    nu0 = float(np.arctan2(r_e_eci[1], r_e_eci[0]))

    # 计算参考点（Chief）在 ECI 下的状态
    r_chief, v_chief = keplerian_to_eci(a, e, i, Om, om, nu0, MU)

    # 转换逃逸者和追击者的绝对状态为 LVLH 坐标下的相对状态
    X_e0 = eci_to_lvlh_state(cfg["X_e0_eci"][:3], cfg["X_e0_eci"][3:], r_chief, v_chief)
    X_p0 = eci_to_lvlh_state(cfg["X_p0_eci"][:3], cfg["X_p0_eci"][3:], r_chief, v_chief)
    return X_p0, X_e0, nu0


def create_ekf(noisy: bool, x0: np.ndarray, initial_dist: float) -> RelativeStateEKF:
    """
    创建 EKF 实例，噪声参数根据初始距离进行比例缩放。
    
    逻辑：
    位置误差标准差 = REF_SIGMA_POS * (当前距离 / 基准距离)
    速度误差标准差 = REF_SIGMA_VEL * (当前距离 / 基准距离)
    """
    if noisy:
        # 有噪声模式：根据距离动态计算 P0 (初始协方差)
        scale = initial_dist / REF_DIST
        sigma_pos = REF_SIGMA_POS * scale   # 位置误差标准差 (km)
        sigma_vel = REF_SIGMA_VEL * scale   # 速度误差标准差 (km/s)
        P0 = np.diag([sigma_pos**2] * 3 + [sigma_vel**2] * 3)
        # 测量噪声 R：距离量测权重设得很小（1e10），主要依靠角度
        R_meas = np.diag([SIGMA_DIST, SIGMA_ANG**2, SIGMA_ANG**2])
        # 过程噪声 Q
        Q_proc = np.diag([5e-4, 5e-4, 5e-4, 5e-8, 5e-8, 5e-8])
    else:
        # 理想/准确模式：赋予极小的噪声协方差
        P0     = np.diag([1.0, 1.0, 1.0, 1e-4, 1e-4, 1e-4])
        R_meas = np.diag([SIGMA_DIST, 1e-30, 1e-30])
        Q_proc = np.zeros((6, 6))

    return RelativeStateEKF(x0=x0, P0=P0, Q=Q_proc, R=R_meas)


def run_scenario(key: str, cfg: dict, noisy: bool, out_dir: Path, seed: int = 42):
    """运行单个追逃场景仿真，并绘制/保存结果。"""
    # 1. 坐标转换
    X_p0, X_e0, nu0 = eci_to_lvlh_scenario(cfg)
    x_rel0 = X_p0 - X_e0
    initial_dist = float(np.linalg.norm(x_rel0[:3]))

    print(f"\n{'='*60}")
    print(f"场景: {cfg['name']}  |  初始距离: {initial_dist:.1f} km  |  {'有噪声 (EKF+SDRE)' if noisy else '理想测量 (Ideal SDRE)'}")

    # 2. 动力学、控制器初始化
    orb  = OrbitalDynamics(mu=MU, a_c=cfg["chief_orbit"]["a"], e_c=cfg["chief_orbit"]["e"])
    ctrl = SDREGameController(Q=np.eye(6), R=np.eye(3) * 1e13, gamma=GAMMA)

    # 3. 估计初值与 EKF 设置
    if noisy:
        # 为 EKF 估计初值引入一次性随机噪声
        rng_init  = np.random.default_rng(seed)
        scale     = initial_dist / REF_DIST
        sigma_pos = REF_SIGMA_POS * scale
        sigma_vel = REF_SIGMA_VEL * scale
        noise     = rng_init.standard_normal(6) * np.array([sigma_pos]*3 + [sigma_vel]*3)
        x0_est    = x_rel0 + noise
    else:
        # 理想模式下估计值直接取真值
        x0_est = x_rel0.copy()

    ekf = create_ekf(noisy=noisy, x0=x0_est, initial_dist=initial_dist)
    # 为仿真过程中的测量噪声准备随机数生成器
    rng = np.random.default_rng(seed + 1) if noisy else None

    # 4. 构建并运行仿真引擎
    sim = EKFSDRESimulation(
        dynamics=orb, controller=ctrl, ekf=ekf,
        X_p0=X_p0, X_e0=X_e0, nu0=nu0,
        dt=10.0, are_interval=1, rng=rng,
    )

    # 运行 5 个轨道周期
    result = sim.run(t_end=5.0 * orb.T_orbit)

    # 5. 绘图与保存
    tag   = "noisy" if noisy else "ideal"
    title = f"{cfg['name']} — {'EKF+SDRE' if noisy else 'Ideal SDRE'}"
    plot_single_simulation(result, orb, title=title, out_path=str(out_dir / f"{key}_{tag}"))
    return result


if __name__ == "__main__":
    # 输出根目录
    root_out = Path("outputs/figures")

    # 遍历所有定义的场景
    for key, cfg in SCENARIOS.items():
        out_dir = root_out / key
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # 针对每个场景运行“噪声估计”和“理想真值”两种版本，便于对比分析
        run_scenario(key, cfg, noisy=True,  out_dir=out_dir)
        run_scenario(key, cfg, noisy=False, out_dir=out_dir)

    print("\n所有场景批量仿真完成。")

