"""
线性化相对运动方程 (LERM) 动力学模型

将用户给出的非线性相对运动方程 (NERM) 在 r_d ≈ r_c 处一阶线性化，
得到以瞬时真近点角速度 ν̇(t) 和角加速度 ν̈(t) 为时变参数的线性方程：

    ẍ =  3ν̇²x + ν̈y  + 2ν̇ẏ  + u_x
    ÿ = -ν̈x          - 2ν̇ẋ  + u_y
    z̈ = -ν̇²z                  + u_z

等价状态空间形式：ẋ_rel = A_LERM(ν̇, ν̈) · x_rel + B · (u_p - u_e)

线性化的物理依据（引力项处理）：
    μ/r_c² - μ(r_c+x)/r_d³  ≈  2ν̇²·x    （小偏差 x,y,z ≪ r_c）
    -μy/r_d³                ≈  -ν̇²·y → 与 ν̇²y 合并后消零
    -μz/r_d³                ≈  -ν̇²·z

参数 ν̇, ν̈, r_c 由开普勒轨道力学完全确定，是时间的已知函数，
不是系统独立自由度。系统自由度始终为 6D（x_rel）。

圆参考轨道极限：ν̈=0，ν̇=n（常数）→ 退化为 CW 方程。
"""

import numpy as np

from aerospace.dynamics.nerm import OrbitalDynamics


class LERMDynamics:
    """线性化相对运动方程（LERM）动力学模型。

    在每个控制步冻结 ν̇, ν̈（由当前真近点角 ν 计算），构建时变 A_LERM 矩阵。
    SDRE 控制器以此矩阵为输入求解 ARE，得到时变增益。

    Parameters
    ----------
    orbital : OrbitalDynamics
        开普勒轨道参数计算器，提供 r_c, ν̇, ν̈
    """

    def __init__(self, orbital: OrbitalDynamics):
        self.orbital = orbital
        self.T_orbit = orbital.T_orbit

        # 输入矩阵 B：推力加速度直接作用在速度分量上（与 CW 相同）
        self._B = np.zeros((6, 3))
        self._B[3:, :] = np.eye(3)

    def get_A_matrix(self, nu_dot: float, nu_ddot: float) -> np.ndarray:
        """构建时变系统矩阵 A_LERM(ν̇, ν̈)。

        矩阵结构（行索引对应 [x, y, z, vx, vy, vz] 的导数）：

            ⎡ 0      0      0      1      0     0  ⎤
            ⎢ 0      0      0      0      1     0  ⎥
            ⎢ 0      0      0      0      0     1  ⎥
            ⎢ 3ν̇²   ν̈     0      0      2ν̇   0  ⎥
            ⎢ -ν̈    0      0      -2ν̇   0     0  ⎥
            ⎣ 0      0      -ν̇²   0      0     0  ⎦

        圆轨道极限（ν̈=0，ν̇=n）：退化为 CW A 矩阵，与 cw_dynamics.py 一致。

        Parameters
        ----------
        nu_dot : float
            真近点角速度 ν̇ (rad/s)
        nu_ddot : float
            真近点角加速度 ν̈ (rad/s²)

        Returns
        -------
        A : (6, 6) ndarray
            时变系统矩阵
        """
        nd  = nu_dot
        ndd = nu_ddot

        A = np.array([
            # x        y     z      vx     vy    vz
            [0.,       0.,   0.,    1.,    0.,   0.  ],  # ẋ  = vx
            [0.,       0.,   0.,    0.,    1.,   0.  ],  # ẏ  = vy
            [0.,       0.,   0.,    0.,    0.,   1.  ],  # ż  = vz
            [3*nd**2,  ndd,  0.,    0.,   2*nd,  0.  ],  # v̇x = 3ν̇²x + ν̈y + 2ν̇vy
            [-ndd,     0.,   0.,  -2*nd,   0.,   0.  ],  # v̇y = -ν̈x  - 2ν̇vx
            [0.,       0.,  -nd**2, 0.,    0.,   0.  ],  # v̇z = -ν̇²z
        ])
        return A

    def get_B_matrix(self) -> np.ndarray:
        """返回输入矩阵 B（推力加速度作用在速度分量上）。

        Returns
        -------
        B : (6, 3) ndarray
        """
        return self._B.copy()

    def ode_rhs(self, x_rel: np.ndarray, u_p: np.ndarray, u_e: np.ndarray,
                nu_dot: float, nu_ddot: float) -> np.ndarray:
        """计算 6D 相对运动 ODE 右端项。

        dx_rel/dt = A_LERM(ν̇, ν̈) · x_rel + B · (u_p - u_e)

        ν̇, ν̈ 在 ZOH 步内冻结（由步开始时的 ν 计算）。

        Parameters
        ----------
        x_rel   : (6,) ndarray  当前相对状态 [x, y, z, vx, vy, vz]
        u_p     : (3,) ndarray  追踪者推力加速度
        u_e     : (3,) ndarray  逃逸者推力加速度
        nu_dot  : float         当前步冻结的 ν̇ (rad/s)
        nu_ddot : float         当前步冻结的 ν̈ (rad/s²)

        Returns
        -------
        dx : (6,) ndarray  状态时间导数
        """
        A = self.get_A_matrix(nu_dot, nu_ddot)
        return A @ x_rel + self._B @ (u_p - u_e)
