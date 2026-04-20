"""
2D NERM 航天器追逃博弈动力学模型

将 3D NERM 简化为面内（径向 x + 沿迹 y）2D 模型。
z 方向（法向）与面内完全解耦，设 z=0, vz=0 后面内运动自洽。

状态向量：
- 单星: [x, y, vx, vy]  (4D)
- 闭环: [X_p(4), X_e(4), nu(1)]  (9D)
"""

import numpy as np


class OrbitalDynamics2D:
    """2D 面内非线性椭圆轨道相对运动动力学模型。

    Parameters
    ----------
    mu : float
        中心天体引力常数 (km^3/s^2)
    a_c : float
        参考轨道半长轴 (km)
    e_c : float
        参考轨道偏心率
    """

    def __init__(self, mu: float = 3.986e5, a_c: float = 15000.0, e_c: float = 0.5):
        self.mu = mu
        self.a_c = a_c
        self.e_c = e_c
        self.T_orbit = 2 * np.pi * np.sqrt(a_c**3 / mu)

    def get_orbital_params(self, nu: float) -> tuple[float, float, float]:
        """计算参考轨道在给定真近点角下的参数。

        Returns
        -------
        r_c, nu_dot, nu_ddot
        """
        r_c = (self.a_c * (1 - self.e_c**2)) / (1 + self.e_c * np.cos(nu))
        nu_dot = np.sqrt(self.mu * self.a_c * (1 - self.e_c**2)) / (r_c**2)
        r_c_dot = np.sqrt(self.mu / (self.a_c * (1 - self.e_c**2))) * self.e_c * np.sin(nu)
        nu_ddot = -(2 * r_c_dot * nu_dot) / r_c
        return r_c, nu_dot, nu_ddot

    def get_SDC_matrix(
        self,
        X_p: np.ndarray,
        X_e: np.ndarray,
        r_c: float,
        nu_dot: float,
        nu_ddot: float,
    ) -> np.ndarray:
        """构建 2D 面内状态依赖系数 (SDC) 矩阵 A_SDC (4x4)。

        Parameters
        ----------
        X_p : (4,) ndarray  追踪星 [x, y, vx, vy]
        X_e : (4,) ndarray  逃逸星 [x, y, vx, vy]
        """
        x_p, y_p = X_p[0], X_p[1]
        x_e, y_e = X_e[0], X_e[1]

        x_rel = x_p - x_e
        y_rel = y_p - y_e

        # 2D 地心距离（z = 0）
        r_p = np.sqrt((r_c + x_p) ** 2 + y_p**2)
        r_e = np.sqrt((r_c + x_e) ** 2 + y_e**2)

        r2_rel = x_rel**2 + y_rel**2 + 1e-6

        b_x = -(self.mu * (r_c + x_p)) / (r_p**3) + (self.mu * (r_c + x_e)) / (r_e**3)
        b_y = -(self.mu * y_p) / (r_p**3) + (self.mu * y_e) / (r_e**3)

        A = np.zeros((4, 4))

        # 位置→速度
        A[0, 2] = 1.0
        A[1, 3] = 1.0

        # 加速度块（引力 + 离心 + 科氏）
        A[2, 0] = nu_dot**2 + (b_x * x_rel) / r2_rel
        A[2, 1] = nu_ddot + (b_x * y_rel) / r2_rel
        A[3, 0] = -nu_ddot + (b_y * x_rel) / r2_rel
        A[3, 1] = nu_dot**2 + (b_y * y_rel) / r2_rel

        # 科里奥利
        A[2, 3] = 2 * nu_dot
        A[3, 2] = -2 * nu_dot

        return A

    def dynamics_9d(
        self, t: float, state: np.ndarray, u_p: np.ndarray, u_e: np.ndarray
    ) -> np.ndarray:
        """9 维系统 ODE 右端项。

        state = [X_p(4), X_e(4), nu(1)]
        u_p, u_e : (2,) 面内推力加速度 [u_x, u_y]
        """
        X_p = state[0:4]
        X_e = state[4:8]
        nu = state[8]

        r_c, nu_dot, nu_ddot = self.get_orbital_params(nu)

        x_p, y_p, vx_p, vy_p = X_p
        x_e, y_e, vx_e, vy_e = X_e

        r_p = np.sqrt((r_c + x_p) ** 2 + y_p**2)
        r_e = np.sqrt((r_c + x_e) ** 2 + y_e**2)

        dstate = np.zeros(9)

        # 位置导数
        dstate[0] = vx_p
        dstate[1] = vy_p
        dstate[4] = vx_e
        dstate[5] = vy_e

        # 追踪星加速度
        dstate[2] = (
            2 * nu_dot * vy_p
            + nu_ddot * y_p
            + nu_dot**2 * x_p
            - (self.mu * (r_c + x_p)) / (r_p**3)
            + self.mu / (r_c**2)
            + u_p[0]
        )
        dstate[3] = (
            -2 * nu_dot * vx_p
            - nu_ddot * x_p
            + nu_dot**2 * y_p
            - (self.mu * y_p) / (r_p**3)
            + u_p[1]
        )

        # 逃逸星加速度
        dstate[6] = (
            2 * nu_dot * vy_e
            + nu_ddot * y_e
            + nu_dot**2 * x_e
            - (self.mu * (r_c + x_e)) / (r_e**3)
            + self.mu / (r_c**2)
            + u_e[0]
        )
        dstate[7] = (
            -2 * nu_dot * vx_e
            - nu_ddot * x_e
            + nu_dot**2 * y_e
            - (self.mu * y_e) / (r_e**3)
            + u_e[1]
        )

        # 真近点角
        dstate[8] = nu_dot

        return dstate
