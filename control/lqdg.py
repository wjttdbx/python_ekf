"""
线性二次型微分博弈 (LQDG) 控制器

零和博弈框架：
  - 追踪者 (Pursuer/Chaser) 最小化代价函数 J
  - 逃逸者 (Evader)           最大化代价函数 J

代价函数:
  J = ∫₀^∞ (xᵀQx + uₚᵀRₚuₚ − uₑᵀRₑuₑ) dt

最优控制律由博弈代数黎卡提方程 (GARE) 的解 P 给出:
  uₚ* = −Rₚ⁻¹ BₚᵀP x   (追踪者)
  uₑ* =  Rₑ⁻¹ BₑᵀP x   (逃逸者)
"""

import numpy as np
from scipy.linalg import solve_continuous_are, LinAlgError


def solve_game_riccati(
    A: np.ndarray,
    Bp: np.ndarray,
    Be: np.ndarray,
    Q: np.ndarray,
    Rp: np.ndarray,
    Re: np.ndarray,
) -> np.ndarray:
    """求解博弈代数黎卡提方程 (GARE)。

    GARE:
        AᵀP + PA − P·S·P + Q = 0
    其中:
        S = Bₚ Rₚ⁻¹ BₚT − Bₑ Rₑ⁻¹ BₑT

    由于 Bp 和 Be 具有相同的列空间结构 (Be = -Bp = -B)，S 可以分解为:
        S = B · M · Bᵀ,    M = Rₚ⁻¹ − Rₑ⁻¹
    当 M 正定时，GARE 等价于标准 CARE:
        AᵀP + PA − P·B·R_eff⁻¹·Bᵀ·P + Q = 0
    其中 R_eff = M⁻¹ = (Rₚ⁻¹ − Rₑ⁻¹)⁻¹。

    Parameters
    ----------
    A  : (n, n)  系统矩阵
    Bp : (n, m)  追踪者输入矩阵 (= B)
    Be : (n, m)  逃逸者输入矩阵 (= -B)
    Q  : (n, n)  状态权重矩阵 (半正定)
    Rp : (m, m)  追踪者控制权重矩阵 (正定)
    Re : (m, m)  逃逸者控制权重矩阵 (正定)

    Returns
    -------
    P : (n, n) GARE 的正定对称解

    Raises
    ------
    ValueError
        当 M = Rₚ⁻¹ − Rₑ⁻¹ 不正定（逃逸者机动惩罚 Re 相对 Rp 过小）或 GARE 无解时
    """
    Rp_inv = np.linalg.inv(Rp)
    Re_inv = np.linalg.inv(Re)

    # M = Rₚ⁻¹ − Rₑ⁻¹，鞍点解存在的必要条件是 M 正定
    # 物理含义: 追踪者的控制能力优势必须大于逃逸者 (Rp < Re)
    M = Rp_inv - Re_inv

    eigvals_M = np.linalg.eigvalsh(M)
    if np.any(eigvals_M <= 0):
        raise ValueError(
            f"矩阵 M = Rₚ⁻¹ − Rₑ⁻¹ 不正定 "
            f"(最小特征值 = {eigvals_M.min():.6e})。\n"
            "逃逸者机动惩罚 Re 相对于追踪者惩罚 Rp 过小，GARE 鞍点解不存在。\n"
            "请增大 Re 或减小 Rp。"
        )

    # 等效 CARE: R_eff = M⁻¹，使得 B · R_eff⁻¹ · Bᵀ = B · M · Bᵀ = S
    R_eff = np.linalg.inv(M)

    try:
        P = solve_continuous_are(A, Bp, Q, R_eff)
    except LinAlgError as e:
        raise ValueError(
            f"GARE 求解失败: {e}\n"
            "可能原因: (A, B) 不可镇定或权重矩阵设置不合理。"
        ) from e

    return P


class ChaserController:
    """追踪者最优控制器。

    最优控制律: uₚ* = −Rₚ⁻¹ BₚᵀP x

    Parameters
    ----------
    Bp : (n, m) 追踪者输入矩阵
    Rp : (m, m) 追踪者控制权重矩阵
    P  : (n, n) GARE 解矩阵
    """

    def __init__(self, Bp: np.ndarray, Rp: np.ndarray, P: np.ndarray):
        self.gain = np.linalg.inv(Rp) @ Bp.T @ P  # Rₚ⁻¹ BₚᵀP

    def compute_control(self, x: np.ndarray) -> np.ndarray:
        """计算追踪者最优推力加速度。

        Parameters
        ----------
        x : (n,) 当前相对状态向量

        Returns
        -------
        u_p : (m,) 追踪者推力加速度 (m/s²)
        """
        return -self.gain @ x


class EvaderController:
    """逃逸者最优控制器。

    最优控制律: uₑ* = Rₑ⁻¹ BₑᵀP x

    Parameters
    ----------
    Be : (n, m) 逃逸者输入矩阵
    Re : (m, m) 逃逸者控制权重矩阵
    P  : (n, n) GARE 解矩阵
    """

    def __init__(self, Be: np.ndarray, Re: np.ndarray, P: np.ndarray):
        self.gain = np.linalg.inv(Re) @ Be.T @ P  # Rₑ⁻¹ BₑᵀP

    def compute_control(self, x: np.ndarray) -> np.ndarray:
        """计算逃逸者最优推力加速度。

        Parameters
        ----------
        x : (n,) 当前相对状态向量

        Returns
        -------
        u_e : (m,) 逃逸者推力加速度 (m/s²)
        """
        return self.gain @ x
