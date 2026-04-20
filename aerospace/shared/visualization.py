"""
追逃博弈仿真可视化

包含三个主要图表:
  1. LVLH 坐标系下的 3D 追逃轨迹动画
  2. 两星相对距离随时间变化曲线
  3. 双方推力加速度随时间变化曲线
"""

import numpy as np
import matplotlib
from aerospace.shared.coord_transform import (
    lvlh_to_eci,
    chief_eci_pos,
    plot_eci_trajectory as _plot_eci,
)

# 优先使用 TkAgg 交互式后端；不可用时回退到 Agg（仅保存文件）
_INTERACTIVE = False
try:
    matplotlib.use("TkAgg")
    # matplotlib.use() 不会立刻验证后端可用性，需要主动触发加载
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: F401
    _INTERACTIVE = True
except Exception:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from aerospace.simulation.cw_lqdg import SimResult

plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "SimHei"],
    "axes.unicode_minus": False,
})


def plot_3d_trajectory(result: SimResult, save_path: str | None = None):
    """绘制 LVLH 坐标系下的 3D 追逃轨迹。

    坐标轴单位转换为 km 便于观察。
    """
    x = result.states[0] / 1000  # m -> km
    y = result.states[1] / 1000
    z = result.states[2] / 1000

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(x, y, z, "b-", linewidth=0.8, label="Relative Trajectory")
    ax.plot([x[0]], [y[0]], [z[0]], "go", markersize=10, label="Start")
    ax.plot([x[-1]], [y[-1]], [z[-1]], "r^", markersize=10, label="End")
    ax.plot([0], [0], [0], "ks", markersize=10, label="Target (Origin)")

    ax.set_xlabel("Radial x (km)")
    ax.set_ylabel("Along-track y (km)")
    ax.set_zlabel("Cross-track z (km)")
    ax.set_title("Pursuit-Evasion Relative Trajectory (LVLH)")
    ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def animate_3d_trajectory(
    result: SimResult,
    interval: int = 20,
    skip: int = 10,
    save_path: str | None = None,
):
    """3D 追逃轨迹动画。

    Parameters
    ----------
    interval : int   帧间隔 (ms)
    skip     : int   每隔 skip 个数据点取一帧，控制动画速度
    save_path: str   若指定则保存为 gif
    """
    x = result.states[0] / 1000
    y = result.states[1] / 1000
    z = result.states[2] / 1000

    idx = np.arange(0, len(x), skip)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    margin = 0.5
    ax.set_xlim(min(x.min(), 0) - margin, max(x.max(), 0) + margin)
    ax.set_ylim(min(y.min(), 0) - margin, max(y.max(), 0) + margin)
    ax.set_zlim(min(z.min(), 0) - margin, max(z.max(), 0) + margin)

    ax.set_xlabel("Radial x (km)")
    ax.set_ylabel("Along-track y (km)")
    ax.set_zlabel("Cross-track z (km)")
    ax.set_title("Pursuit-Evasion Trajectory Animation")

    ax.plot([0], [0], [0], "ks", markersize=10, label="Target")
    (line,) = ax.plot([], [], [], "b-", linewidth=0.8, label="Trajectory")
    (point,) = ax.plot([], [], [], "ro", markersize=6, label="Chaser")
    ax.legend()

    def update(frame):
        i = idx[frame]
        line.set_data(x[: i + 1], y[: i + 1])
        line.set_3d_properties(z[: i + 1])
        point.set_data([x[i]], [y[i]])
        point.set_3d_properties([z[i]])
        return line, point

    anim = FuncAnimation(
        fig, update, frames=len(idx), interval=interval, blit=False
    )

    if save_path:
        anim.save(save_path, writer="pillow", fps=30)
    plt.show()


def plot_relative_distance(result: SimResult, save_path: str | None = None):
    """绘制两星相对距离随时间变化曲线。"""
    pos = result.states[:3]  # (3, N)
    distance = np.linalg.norm(pos, axis=0) / 1000  # m -> km
    t_min = result.t / 60  # s -> min

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_min, distance, "b-", linewidth=1.2)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Relative Distance (km)")
    ax.set_title("Pursuer-Evader Relative Distance")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_thrust_profiles(result: SimResult, save_path: str | None = None):
    """绘制双方推力加速度幅值随时间变化曲线。"""
    t_min = result.t / 60

    u_p_norm = np.linalg.norm(result.u_p_history, axis=0) * 1000  # m/s² -> mm/s²
    u_e_norm = np.linalg.norm(result.u_e_history, axis=0) * 1000

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_min, u_p_norm, "r-", linewidth=1.2, label="Pursuer ||u_p||")
    ax.plot(t_min, u_e_norm, "b--", linewidth=1.2, label="Evader ||u_e||")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Thrust Acceleration (mm/s²)")
    ax.set_title("Thrust Acceleration Profiles")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_eci_trajectory_cw(
    result: SimResult,
    a_m: float,
    n: float,
    e: float = 0.0,
    save_dir: str | None = None,
) -> None:
    """绘制 CW+LQDG 仿真结果在 ECI 坐标系中的绝对轨迹。

    Parameters
    ----------
    result : SimResult  仿真结果（须包含 X_p_history / X_e_history）
    a_m    : float      参考轨道半长轴 (m)
    n      : float      轨道角速度 (rad/s)，用于计算 ν(t) = n·t
    e      : float      轨道偏心率（圆轨道默认 0）
    save_dir : str      保存目录（None=不保存）
    """
    R_EARTH_M = 6.371e6  # m

    # 圆轨道：ν(t) = n · t
    nu = n * result.t

    # LVLH 绝对位置 → ECI（单位 m，绘图时转换为 km）
    p_eci = lvlh_to_eci(result.X_p_history, nu, a_m, e) / 1e3   # km
    e_eci = lvlh_to_eci(result.X_e_history, nu, a_m, e) / 1e3
    c_eci = chief_eci_pos(nu, a_m / 1e3, e)                       # km（直接用 km 单位的 a）

    save_path = f"{save_dir}/cw_lqdg_eci_trajectory.png" if save_dir else None
    _plot_eci(
        pursuer_eci=p_eci,
        evader_eci=e_eci,
        chief_eci=c_eci,
        earth_radius=R_EARTH_M / 1e3,
        title="CW+LQDG: Pursuer & Evader Trajectories (ECI)",
        unit_label="km",
        save_path=save_path,
    )


def plot_all(result: SimResult, save_dir: str | None = None,
             orbital_params: dict | None = None):
    """一次性生成所有静态图表。

    Parameters
    ----------
    orbital_params : dict, optional
        包含 ``a`` (m) 和 ``n`` (rad/s) 的字典，提供时额外生成 ECI 轨迹图。
    """
    prefix = f"{save_dir}/" if save_dir else None

    plot_3d_trajectory(
        result, save_path=f"{prefix}trajectory_3d.png" if prefix else None
    )
    plot_relative_distance(
        result, save_path=f"{prefix}relative_distance.png" if prefix else None
    )
    plot_thrust_profiles(
        result, save_path=f"{prefix}thrust_profiles.png" if prefix else None
    )

    if orbital_params is not None:
        plot_eci_trajectory_cw(
            result,
            a_m=orbital_params["a"],
            n=orbital_params["n"],
            e=orbital_params.get("e", 0.0),
            save_dir=save_dir,
        )
