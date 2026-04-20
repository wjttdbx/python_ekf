"""
SDRE 航天器追逃博弈动力学模型

基于状态依赖里卡蒂方程（SDRE）的非线性航天器追逃博弈动力学。
假设中心天体为地球，参考轨道为大椭圆轨道。
追踪星（Pursuer）和逃逸星（Evader）的运动均在参考轨道的 LVLH 坐标系下描述。
"""

from __future__ import annotations

import math
import numpy as np

class OrbitalDynamics:
    """非线性大椭圆轨道相对运动动力学模型。

    Parameters
    ----------
    mu : float
        中心天体引力常数 (km^3/s^2)，默认为地球 3.986e5
    a_c : float
        参考轨道半长轴 (km)，默认 15000 km
    e_c : float
        参考轨道偏心率，默认 0.5
    """

    def __init__(self, mu: float = 3.986e5, a_c: float = 15000.0, e_c: float = 0.5):
        self.mu = mu
        self.a_c = a_c
        self.e_c = e_c
        
        # 预先计算轨道周期
        self.T_orbit = 2 * np.pi * math.sqrt(a_c**3 / mu)

    def get_orbital_params(self, nu: float) -> tuple[float, float, float]:
        """计算参考轨道在给定真近点角下的参数。

        Parameters
        ----------
        nu : float
            真近点角 (rad)

        Returns
        -------
        r_c : float
            参考轨道地心距离 (km)
        nu_dot : float
            真近点角速度 (rad/s)
        nu_ddot : float
            真近点角加速度 (rad/s^2)
        """
        # 参考轨道半径
        r_c = (self.a_c * (1 - self.e_c**2)) / (1 + self.e_c * np.cos(nu))
        
        # 真近点角速度
        nu_dot = np.sqrt(self.mu * self.a_c * (1 - self.e_c**2)) / (r_c**2)
        
        # 半径变化率 r_c_dot
        r_c_dot = np.sqrt(self.mu / (self.a_c * (1 - self.e_c**2))) * self.e_c * np.sin(nu)
        
        # 真近点角加速度
        nu_ddot = - (2 * r_c_dot * nu_dot) / r_c
        
        return r_c, nu_dot, nu_ddot

    def get_SDC_matrix(self, X_p: np.ndarray, X_e: np.ndarray, 
                       r_c: float, nu_dot: float, nu_ddot: float) -> np.ndarray:
        """构建状态依赖系数 (SDC) 矩阵 A_SDC。

        Parameters
        ----------
        X_p : (6,) ndarray
            追踪星绝对状态 [x, y, z, vx, vy, vz]
        X_e : (6,) ndarray
            逃逸星绝对状态 [x, y, z, vx, vy, vz]
        r_c : float
            参考轨道地心距离
        nu_dot : float
            真近点角速度
        nu_ddot : float
            真近点角加速度

        Returns
        -------
        A_SDC : (6, 6) ndarray
            状态依赖系数矩阵
        """
        x_p, y_p, z_p = X_p[0], X_p[1], X_p[2]
        x_e, y_e, z_e = X_e[0], X_e[1], X_e[2]
        
        # 相对状态
        x_rel = x_p - x_e
        y_rel = y_p - y_e
        z_rel = z_p - z_e
        
        # 追踪星和逃逸星的地心距离
        r_p = np.sqrt((r_c + x_p)**2 + y_p**2 + z_p**2)
        r_e = np.sqrt((r_c + x_e)**2 + y_e**2 + z_e**2)
        
        # 相对距离的平方（加上极小值防止除零）
        r2_rel = x_rel**2 + y_rel**2 + z_rel**2 + 1e-6
        
        # 非线性引力差值项
        b_x = - (self.mu * (r_c + x_p)) / (r_p**3) + (self.mu * (r_c + x_e)) / (r_e**3)
        b_y = - (self.mu * y_p) / (r_p**3) + (self.mu * y_e) / (r_e**3)
        b_z = - (self.mu * z_p) / (r_p**3) + (self.mu * z_e) / (r_e**3)
        
        # 组装 SDC 矩阵
        A_SDC = np.zeros((6, 6))
        
        # 右上角 A12: I_3x3
        A_SDC[0, 3] = 1.0
        A_SDC[1, 4] = 1.0
        A_SDC[2, 5] = 1.0
        
        # 左下角 A21
        A_SDC[3, 0] = nu_dot**2 + (b_x * x_rel) / r2_rel
        A_SDC[3, 1] = nu_ddot + (b_x * y_rel) / r2_rel
        A_SDC[3, 2] = (b_x * z_rel) / r2_rel
        
        A_SDC[4, 0] = -nu_ddot + (b_y * x_rel) / r2_rel
        A_SDC[4, 1] = nu_dot**2 + (b_y * y_rel) / r2_rel
        A_SDC[4, 2] = (b_y * z_rel) / r2_rel
        
        A_SDC[5, 0] = (b_z * x_rel) / r2_rel
        A_SDC[5, 1] = (b_z * y_rel) / r2_rel
        A_SDC[5, 2] = (b_z * z_rel) / r2_rel
        
        # 右下角 A22: 科里奥利力项
        A_SDC[3, 4] = 2 * nu_dot
        A_SDC[4, 3] = -2 * nu_dot
        
        return A_SDC

    def dynamics_13d(self, t: float, state: np.ndarray, u_p: np.ndarray, u_e: np.ndarray) -> np.ndarray:
        """13 维系统的常微分方程右端项。
        
        状态向量 state = [X_p(6), X_e(6), nu(1)]，共 13 维
        u_p: 追踪星的控制输入 [ux_p, uy_p, uz_p]
        u_e: 逃逸星的控制输入 [ux_e, uy_e, uz_e]

        Parameters
        ----------
        t : float
            当前时间 (s)
        state : (13,) ndarray
            当前状态向量
        u_p : (3,) ndarray
            追踪星推力加速度 (km/s^2)
        u_e : (3,) ndarray
            逃逸星推力加速度 (km/s^2)

        Returns
        -------
        dstate : (13,) ndarray
            状态的时间导数
        """
        X_p = state[0:6]
        X_e = state[6:12]
        nu = state[12]
        
        r_c, nu_dot, nu_ddot = self.get_orbital_params(nu)
        
        dstate = np.zeros_like(state)
        
        # 提取速度和位置
        x_p, y_p, z_p, vx_p, vy_p, vz_p = X_p
        x_e, y_e, z_e, vx_e, vy_e, vz_e = X_e
        
        # 速度导数
        dstate[0:3] = X_p[3:6]
        dstate[6:9] = X_e[3:6]
        
        # 追踪星和逃逸星的地心距离
        r_p = np.sqrt((r_c + x_p)**2 + y_p**2 + z_p**2)
        r_e = np.sqrt((r_c + x_e)**2 + y_e**2 + z_e**2)
        
        # 加速度导数 (X_p)
        dstate[3] = 2 * nu_dot * vy_p + nu_ddot * y_p + nu_dot**2 * x_p - (self.mu * (r_c + x_p)) / (r_p**3) + self.mu / (r_c**2) + u_p[0]
        dstate[4] = -2 * nu_dot * vx_p - nu_ddot * x_p + nu_dot**2 * y_p - (self.mu * y_p) / (r_p**3) + u_p[1]
        dstate[5] = - (self.mu * z_p) / (r_p**3) + u_p[2]
        
        # 加速度导数 (X_e)
        dstate[9] = 2 * nu_dot * vy_e + nu_ddot * y_e + nu_dot**2 * x_e - (self.mu * (r_c + x_e)) / (r_e**3) + self.mu / (r_c**2) + u_e[0]
        dstate[10] = -2 * nu_dot * vx_e - nu_ddot * x_e + nu_dot**2 * y_e - (self.mu * y_e) / (r_e**3) + u_e[1]
        dstate[11] = - (self.mu * z_e) / (r_e**3) + u_e[2]
        
        # 真近点角导数
        dstate[12] = nu_dot
        
        return dstate
