"""
SDRE 航天器追逃博弈仿真 — 主入口
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from aerospace.paths import FIGURES_DIR, ensure_figures_dir
from aerospace.control.sdre import SDREGameController
from aerospace.dynamics.nerm import OrbitalDynamics
from aerospace.simulation.nerm_sdre import SDRESimulation
from aerospace.shared.coord_transform import (
    lvlh_to_eci,
    chief_eci_pos,
    plot_eci_trajectory,
)

def plot_results(result, save_dir: str = None,
                 orbital: OrbitalDynamics | None = None):
    """绘制 SDRE 仿真结果并保存。

    Parameters
    ----------
    orbital : OrbitalDynamics, optional  提供时额外输出 ECI 绝对轨迹图
    """
    import os
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 1. 绘制 3D 相对轨迹
    # x_rel = X_p - X_e
    x_rel = result.states[0:3, :] - result.states[6:9, :]
    x, y, z = x_rel[0], x_rel[1], x_rel[2]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    ax.plot(x, y, z, "b-", linewidth=1.2, label="Relative Trajectory (SDRE)")
    ax.plot([x[0]], [y[0]], [z[0]], "go", markersize=8, label="Start")
    ax.plot([x[-1]], [y[-1]], [z[-1]], "r^", markersize=8, label="End")
    ax.plot([0], [0], [0], "ks", markersize=8, label="Target (Origin)")
    
    ax.set_xlabel("Radial x (km)")
    ax.set_ylabel("Along-track y (km)")
    ax.set_zlabel("Cross-track z (km)")
    ax.set_title("SDRE Pursuit-Evasion Relative Trajectory (LVLH)")
    ax.legend()
    
    plt.tight_layout()
    if save_dir:
        fig.savefig(f"{save_dir}/sdre_trajectory_3d.png", dpi=150)
    plt.show()
    
    # 2. 绘制相对距离曲线
    distance = np.sqrt(x**2 + y**2 + z**2)
    t_min = result.t / 60.0
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_min, distance, "b-", linewidth=1.5)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Relative Distance (km)")
    ax.set_title("SDRE Pursuer-Evader Relative Distance")
    ax.grid(True, alpha=0.5)
    
    plt.tight_layout()
    if save_dir:
        fig.savefig(f"{save_dir}/sdre_relative_distance.png", dpi=150)
    plt.show()

    # 3. ECI 轨迹图（直接从 13D 状态提取 X_p, X_e, ν）
    if orbital is not None:
        X_p_lvlh = result.states[0:3, :]    # (3, N) 追踪者 LVLH 位置 (km)
        X_e_lvlh = result.states[6:9, :]    # (3, N) 逃逸者 LVLH 位置 (km)
        nu_arr   = result.states[12, :]     # (N,)   真近点角 (rad)

        p_eci = lvlh_to_eci(X_p_lvlh, nu_arr, orbital.a_c, orbital.e_c)
        e_eci = lvlh_to_eci(X_e_lvlh, nu_arr, orbital.a_c, orbital.e_c)
        c_eci = chief_eci_pos(nu_arr, orbital.a_c, orbital.e_c)

        save_path = f"{save_dir}/sdre_eci_trajectory.png" if save_dir else None
        plot_eci_trajectory(
            pursuer_eci=p_eci, evader_eci=e_eci, chief_eci=c_eci,
            earth_radius=6371.0,
            title="NERM+SDRE: Pursuer & Evader Trajectories (ECI)",
            unit_label="km", save_path=save_path,
        )

def print_P_summary(result, controller) -> None:
    """打印仿真过程中 P 矩阵的统计摘要。"""
    P_hist = result.P_history  # (N, 6, 6)
    N = P_hist.shape[0]

    # 各时间步的条件数和最大特征值
    cond_hist    = np.array([np.linalg.cond(P_hist[i]) for i in range(N)])
    eigmax_hist  = np.array([np.linalg.eigvalsh(P_hist[i]).max() for i in range(N)])
    eigmin_hist  = np.array([np.linalg.eigvalsh(P_hist[i]).min() for i in range(N)])

    print("\n" + "="*60)
    print("  P 矩阵统计摘要")
    print("="*60)
    print(f"  仿真步数:          {N}")
    print(f"  条件数  均值/最大:  {cond_hist.mean():.4e} / {cond_hist.max():.4e}")
    print(f"  最大特征值 均值/最大:{eigmax_hist.mean():.4e} / {eigmax_hist.max():.4e}")
    print(f"  最小特征值 均值/最小:{eigmin_hist.mean():.4e} / {eigmin_hist.min():.4e}")

    # 打印初始、中间、末尾三个时刻的完整 P 矩阵
    for idx, label in [(0, "初始时刻"), (N // 2, "中间时刻"), (N - 1, "末尾时刻")]:
        t_val = result.t[idx]
        print(f"\n  [{label}  t = {t_val:.1f} s]")
        P = P_hist[idx]
        for row in P:
            row_str = "  ".join(f"{v:+12.4e}" for v in row)
            print(f"    [ {row_str} ]")
        eigs = np.linalg.eigvalsh(P)
        print(f"    特征值: {eigs}")
        print(f"    条件数: {np.linalg.cond(P):.4e}")

    # 最终时刻完整 P（来自控制器缓存，与上面末尾一致，提供对照）
    controller.print_P_matrix("最终时刻（控制器缓存）")


def main():
    # 初始化动力学模型
    dynamics = OrbitalDynamics(mu=3.986e5, a_c=15000.0, e_c=0.5)
    
    # 初始化控制器
    # verbose=True: 每 50 步打印一次 P 矩阵关键信息
    Q = np.eye(6)
    R = np.eye(3) * 1e13
    gamma = np.sqrt(2)
    controller = SDREGameController(Q, R, gamma, verbose=True, verbose_interval=50)
    
    # 初始化仿真器
    # 追踪星初始状态 X_p(0) = [500, 500, 500, 0.01, 0.01, 0.01]^T
    X_p0 = np.array([500.0, 500.0, 500.0, 0.01, 0.01, 0.01])
    # 逃逸星初始状态 X_e(0) = [0, 0, 0, 0, 0, 0]^T
    X_e0 = np.zeros(6)
    
    # 仿真器设置
    dt = 10.0
    sim = SDRESimulation(
        dynamics=dynamics,
        controller=controller,
        X_p0=X_p0,
        X_e0=X_e0,
        nu0=0.0,
        dt=dt
    )
    
    # 运行仿真
    start = time.perf_counter()
    result = sim.run()
    end = time.perf_counter()
    print(f"仿真时间: {end - start:.4f}s")
    times = np.array(controller.step_times)
    print(f"单步控制计算时间  均值: {times.mean()*1e3:.4f}ms  最大: {times.max()*1e3:.4f}ms  总计: {times.sum():.4f}s")

    # 打印 P 矩阵摘要
    print_P_summary(result, controller)

    # 可视化（图片统一写入 outputs/figures/）
    ensure_figures_dir()
    plot_results(result, save_dir=str(FIGURES_DIR), orbital=dynamics)

if __name__ == "__main__":
    main()
