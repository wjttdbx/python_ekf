"""
坐标变换工具：LVLH ↔ ECI

假设参考轨道为赤道平面轨道（倾角 i = 0，Ω = 0，ω = 0）。
在此假设下，LVLH → ECI 的旋转矩阵仅由真近点角 ν 决定：

    R(ν) = [[ cos(ν), -sin(ν), 0 ],
             [ sin(ν),  cos(ν), 0 ],
             [    0,       0,   1 ]]

LVLH 轴约定：
  x̂_r（径向）：从地心指向参考星
  x̂_θ（沿迹）：垂直于 x̂_r，在轨道面内，指向运动方向
  x̂_h（法向）：垂直于轨道面（= x̂_r × x̂_θ）
"""

import numpy as np


# ─────────────────────────── 轨道几何 ───────────────────────────

def r_chief(nu: np.ndarray, a: float, e: float) -> np.ndarray:
    """主星（参考轨道）半径序列。

    Parameters
    ----------
    nu : (N,) array  真近点角 (rad)
    a  : float       半长轴（与期望的位置单位相同）
    e  : float       轨道偏心率

    Returns
    -------
    (N,) array  各时刻轨道半径
    """
    return a * (1.0 - e**2) / (1.0 + e * np.cos(nu))


def chief_eci_pos(nu: np.ndarray, a: float, e: float) -> np.ndarray:
    """主星在 ECI 坐标系中的位置序列（赤道平面轨道）。

    Returns
    -------
    (3, N) array  ECI 位置
    """
    nu = np.asarray(nu, dtype=float)
    rc = r_chief(nu, a, e)
    return np.array([rc * np.cos(nu), rc * np.sin(nu), np.zeros_like(nu)])


# ─────────────────────────── 坐标变换 ───────────────────────────

def lvlh_to_eci(pos_lvlh: np.ndarray, nu: np.ndarray,
                a: float, e: float) -> np.ndarray:
    """将 LVLH 相对位置序列批量变换为 ECI 绝对位置。

    Parameters
    ----------
    pos_lvlh : (3, N)  各时刻在 LVLH 系中相对主星的位置（与 a 同单位）
    nu       : (N,)    真近点角序列 (rad)
    a        : float   半长轴
    e        : float   偏心率

    Returns
    -------
    (3, N) array  ECI 绝对位置
    """
    nu = np.asarray(nu, dtype=float)
    pos_lvlh = np.asarray(pos_lvlh, dtype=float)

    cn, sn = np.cos(nu), np.sin(nu)

    # 向量化旋转：R(ν) @ pos_lvlh[:, k]
    x_eci = cn * pos_lvlh[0] - sn * pos_lvlh[1]
    y_eci = sn * pos_lvlh[0] + cn * pos_lvlh[1]
    z_eci = pos_lvlh[2]

    return chief_eci_pos(nu, a, e) + np.array([x_eci, y_eci, z_eci])


# ─────────────────────────── 可视化辅助 ───────────────────────────

def draw_earth(ax, radius: float,
               color: str = "royalblue", alpha: float = 0.20) -> None:
    """在 3D 轴上绘制半透明地球球体。

    Parameters
    ----------
    ax     : Axes3D
    radius : float  地球半径（与轨道坐标同单位）
    """
    u = np.linspace(0, 2 * np.pi, 36)
    v = np.linspace(0, np.pi, 18)
    xs = radius * np.outer(np.cos(u), np.sin(v))
    ys = radius * np.outer(np.sin(u), np.sin(v))
    zs = radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color=color, alpha=alpha, linewidth=0, zorder=0)


def plot_eci_trajectory(
    pursuer_eci: np.ndarray,
    evader_eci: np.ndarray,
    chief_eci: np.ndarray,
    earth_radius: float,
    title: str = "ECI Trajectories",
    unit_label: str = "km",
    save_path: str | None = None,
) -> None:
    """绘制 ECI 坐标系中的追逃轨迹（通用函数，被各 main_*.py 调用）。

    Parameters
    ----------
    pursuer_eci  : (3, N)  追踪者 ECI 位置
    evader_eci   : (3, N)  逃逸者 ECI 位置
    chief_eci    : (3, N)  参考主星 ECI 位置（参考轨道）
    earth_radius : float   地球半径（与位置单位相同）
    title        : str     图标题
    unit_label   : str     坐标轴单位标签
    save_path    : str     保存路径（None=不保存）
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    # 地球
    draw_earth(ax, earth_radius)

    # 参考轨道
    ax.plot(chief_eci[0], chief_eci[1], chief_eci[2],
            "k--", linewidth=0.8, alpha=0.5, label="Chief orbit")

    # 追踪者
    ax.plot(pursuer_eci[0], pursuer_eci[1], pursuer_eci[2],
            "b-", linewidth=1.2, label="Pursuer")
    ax.plot([pursuer_eci[0, 0]], [pursuer_eci[1, 0]], [pursuer_eci[2, 0]],
            "bs", markersize=7)
    ax.plot([pursuer_eci[0, -1]], [pursuer_eci[1, -1]], [pursuer_eci[2, -1]],
            "b^", markersize=7)

    # 逃逸者
    ax.plot(evader_eci[0], evader_eci[1], evader_eci[2],
            "r-", linewidth=1.2, label="Evader")
    ax.plot([evader_eci[0, 0]], [evader_eci[1, 0]], [evader_eci[2, 0]],
            "rs", markersize=7)
    ax.plot([evader_eci[0, -1]], [evader_eci[1, -1]], [evader_eci[2, -1]],
            "r^", markersize=7)

    ax.set_xlabel(f"X ({unit_label})")
    ax.set_ylabel(f"Y ({unit_label})")
    ax.set_zlabel(f"Z ({unit_label})")
    ax.set_title(title)
    ax.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


# ─────────────────────────── ECI → LVLH 转换 ───────────────────────────

def eci_to_lvlh_state(r_eci: np.ndarray, v_eci: np.ndarray,
                      r_chief_eci: np.ndarray, v_chief_eci: np.ndarray) -> np.ndarray:
    """将 ECI 绝对状态转换为 LVLH 相对状态。

    Parameters
    ----------
    r_eci       : (3,) 目标星 ECI 位置 (km)
    v_eci       : (3,) 目标星 ECI 速度 (km/s)
    r_chief_eci : (3,) 参考星 ECI 位置 (km)
    v_chief_eci : (3,) 参考星 ECI 速度 (km/s)

    Returns
    -------
    (6,) LVLH 相对状态 [x, y, z, vx, vy, vz]
    """
    # LVLH 基向量
    r_hat = r_chief_eci / np.linalg.norm(r_chief_eci)
    h_vec = np.cross(r_chief_eci, v_chief_eci)
    h_hat = h_vec / np.linalg.norm(h_vec)
    y_hat = np.cross(h_hat, r_hat)

    # 旋转矩阵 ECI → LVLH（行向量为基向量）
    R = np.array([r_hat, y_hat, h_hat])

    # 相对位置
    dr = r_eci - r_chief_eci
    pos_lvlh = R @ dr

    # 角速度 omega = h / |r|^2
    omega = h_vec / np.linalg.norm(r_chief_eci)**2

    # 相对速度（去除参考系旋转）
    dv = v_eci - v_chief_eci
    vel_lvlh = R @ dv - np.cross(omega, pos_lvlh)

    return np.concatenate([pos_lvlh, vel_lvlh])


def keplerian_to_eci(a: float, e: float, i: float,
                     Omega: float, omega: float, nu: float,
                     mu: float = 3.986e5) -> tuple[np.ndarray, np.ndarray]:
    """从开普勒轨道根数计算 ECI 位置和速度。

    Parameters
    ----------
    a     : 半长轴 (km)
    e     : 偏心率
    i     : 倾角 (rad)
    Omega : 升交点赤经 (rad)
    omega : 近地点幅角 (rad)
    nu    : 真近点角 (rad)
    mu    : 引力常数 (km^3/s^2)

    Returns
    -------
    r_eci : (3,) ECI 位置 (km)
    v_eci : (3,) ECI 速度 (km/s)
    """
    p = a * (1 - e**2)
    r = p / (1 + e * np.cos(nu))

    # 轨道面内位置和速度
    r_orb = np.array([r * np.cos(nu), r * np.sin(nu), 0.0])
    v_orb = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    # 旋转矩阵：轨道面 → ECI（3-1-3 欧拉角：Omega, i, omega）
    cO, sO = np.cos(Omega), np.sin(Omega)
    ci, si = np.cos(i),     np.sin(i)
    co, so = np.cos(omega), np.sin(omega)

    R = np.array([
        [cO*co - sO*so*ci,  -cO*so - sO*co*ci,  sO*si],
        [sO*co + cO*so*ci,  -sO*so + cO*co*ci, -cO*si],
        [so*si,              co*si,              ci   ],
    ])

    return R @ r_orb, R @ v_orb
