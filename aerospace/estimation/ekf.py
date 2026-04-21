"""
Extended Kalman Filter for Relative State Estimation

基于距离-方位角-仰角测量的相对状态估计器。
"""


import numpy as np


class RelativeStateEKF:
    """相对状态扩展卡尔曼滤波器。

    测量模型：距离-方位角-仰角 [ρ, az, el]
    状态：相对位置和速度 [dx, dy, dz, dvx, dvy, dvz]

    Parameters
    ----------
    x0 : (6,) ndarray  初始相对状态估计
    P0 : (6, 6) ndarray  初始协方差矩阵
    Q  : (6, 6) ndarray  过程噪声协方差矩阵
    R  : (3, 3) ndarray  测量噪声协方差矩阵
    """

    def __init__(self, x0: np.ndarray, P0: np.ndarray,
                 Q: np.ndarray, R: np.ndarray):
        self.x = x0.copy()
        self.P = P0.copy()
        self.Q = Q
        self.R = R

    # ── 静态工具方法 ──────────────────────────────────────────────────────────

    @staticmethod
    def measure(X_p: np.ndarray, X_e: np.ndarray) -> np.ndarray:
        """从绝对状态计算 [ρ, az, el] 测量值 (km, rad, rad)。"""
        dx = X_p[:3] - X_e[:3]
        rho = np.linalg.norm(dx)
        az = np.arctan2(dx[1], dx[0])
        el = np.arcsin(np.clip(dx[2] / (rho + 1e-12), -1, 1))
        return np.array([rho, az, el])

    @staticmethod
    def wrap_angle(a: np.ndarray) -> np.ndarray:
        """将角度归一化到 [-π, π]。"""
        return (a + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def meas_jacobian(x_rel: np.ndarray) -> np.ndarray:
        """测量方程对相对状态 [dx,dy,dz,dvx,dvy,dvz] 的雅可比矩阵 H (3×6)。"""
        dx, dy, dz = x_rel[0], x_rel[1], x_rel[2]
        rho = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-12
        rho_xy = np.sqrt(dx**2 + dy**2) + 1e-12

        H = np.zeros((3, 6))
        H[0, 0] = dx / rho
        H[0, 1] = dy / rho
        H[0, 2] = dz / rho
        H[1, 0] = -dy / rho_xy**2
        H[1, 1] =  dx / rho_xy**2
        H[2, 0] = -dx * dz / (rho**2 * rho_xy)
        H[2, 1] = -dy * dz / (rho**2 * rho_xy)
        H[2, 2] =  rho_xy / rho**2
        return H

    # ── 核心滤波步骤 ──────────────────────────────────────────────────────────

    def predict(self, A: np.ndarray, B: np.ndarray,
                u_p: np.ndarray, u_e: np.ndarray, dt: float) -> tuple:
        """EKF 预测步，离散化传播（对齐 C++ F*X / F*P*F^T+Q 结构）。

        Returns
        -------
        x_priori : (6,) ndarray
        P_priori : (6, 6) ndarray
        """
        F = np.eye(6) + A * dt                          # 一阶离散化转移矩阵
        x_priori = F @ self.x + dt * B @ (u_p - u_e)
        P_priori = F @ self.P @ F.T + self.Q
        return x_priori, P_priori

    def update(self, x_priori: np.ndarray, P_priori: np.ndarray,
               z_meas: np.ndarray) -> np.ndarray:
        """EKF 更新步，在预测状态处线性化（对齐 C++ H(X_priori) 结构）。

        Returns
        -------
        y_innov : (3,) ndarray
        """
        rho_p = np.linalg.norm(x_priori[:3]) + 1e-12
        az_p  = np.arctan2(x_priori[1], x_priori[0])
        el_p  = np.arcsin(np.clip(x_priori[2] / rho_p, -1, 1))
        z_pred = np.array([rho_p, az_p, el_p])

        y_innov = z_meas - z_pred
        y_innov[1:3] = self.wrap_angle(y_innov[1:3])

        H = self.meas_jacobian(x_priori)
        S = H @ P_priori @ H.T + self.R
        K = P_priori @ H.T @ np.linalg.inv(S)

        self.x = x_priori + K @ y_innov
        self.P = (np.eye(6) - K @ H) @ P_priori
        return y_innov

    def step(self, A: np.ndarray, B: np.ndarray,
             u_p: np.ndarray, u_e: np.ndarray, dt: float,
             z_meas: np.ndarray) -> np.ndarray:
        """预测 + 更新一步。

        Returns
        -------
        y_innov : (3,) ndarray
        """
        x_priori, P_priori = self.predict(A, B, u_p, u_e, dt)
        return self.update(x_priori, P_priori, z_meas)
