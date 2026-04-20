"""
Clohessy-Wiltshire (CW) 相对运动动力学模型

CW 方程描述了在近圆轨道上，追踪航天器相对于目标航天器的线性化相对运动。
坐标系采用 LVLH（Local Vertical Local Horizontal）坐标系：
  - x: 径向方向（指向远离地心方向）
  - y: 沿迹方向（沿轨道运动方向）
  - z: 法向方向（轨道面法线方向，构成右手系）
"""

import numpy as np


class CWDynamics:
    """基于 Clohessy-Wiltshire 方程的相对运动动力学模型。

    Parameters
    ----------
    n : float
        目标航天器的轨道平均角速度 (rad/s)，满足 n = sqrt(mu / a^3)，
        其中 mu 为地球引力常数，a 为轨道半长轴。
    """

    def __init__(self, n: float):
        if n <= 0:
            raise ValueError("轨道角速度 n 必须为正数")
        self.n = n

    def get_state_space(self) -> tuple[np.ndarray, np.ndarray]:
        """返回 CW 方程的连续时间状态空间矩阵 (A, B)。

        状态向量 x = [x, y, z, vx, vy, vz]^T
          - x, y, z:   相对位置分量 (m)
          - vx, vy, vz: 相对速度分量 (m/s)

        控制输入 u = [ax, ay, az]^T 为推力加速度 (m/s^2)

        动力学方程 dx/dt = A @ x + B @ u

        Returns
        -------
        A : ndarray, shape (6, 6)
            系统矩阵（自由漂移动力学）
        B : ndarray, shape (6, 3)
            输入矩阵（推力加速度作用通道）
        """
        n = self.n
        n2 = n * n

        A = np.array([
            # x    y    z    vx   vy   vz
            [0.,   0.,  0.,  1.,  0.,  0.],   # dx/dt  = vx
            [0.,   0.,  0.,  0.,  1.,  0.],   # dy/dt  = vy
            [0.,   0.,  0.,  0.,  0.,  1.],   # dz/dt  = vz
            [3*n2, 0.,  0.,  0.,  2*n, 0.],   # dvx/dt = 3n²x + 2n·vy        (径向：引力梯度 + 科氏力)
            [0.,   0.,  0., -2*n, 0.,  0.],   # dvy/dt = -2n·vx               (沿迹：科氏力耦合)
            [0.,   0., -n2, 0.,  0.,  0.],    # dvz/dt = -n²z                 (法向：简谐振动)
        ])

        # 推力加速度直接作用在速度分量上
        B = np.array([
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [1., 0., 0.],   # ax -> dvx
            [0., 1., 0.],   # ay -> dvy
            [0., 0., 1.],   # az -> dvz
        ])

        return A, B
