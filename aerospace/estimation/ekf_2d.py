"""
2D Extended Kalman Filter for Relative State Estimation

基于距离-方位角测量的面内相对状态估计器。
"""

import numpy as np


class RelativeStateEKF2D:
    """2D 面内相对状态扩展卡尔曼滤波器。

    测量模型（range_angle）: [ρ, θ]  — 距离 + 方位角
    测量模型（angle_only）:  [θ]     — 仅方位角

    状态：相对位置和速度 [dx, dy, dvx, dvy]

    Parameters
    ----------
    x0 : (4,) ndarray  初始相对状态估计
    P0 : (4, 4) ndarray  初始协方差矩阵
    Q  : (4, 4) ndarray  过程噪声协方差矩阵
    R  : (1, 1) | (2, 2) ndarray  测量噪声协方差矩阵
    """

    def __init__(self, x0: np.ndarray, P0: np.ndarray,
                 Q: np.ndarray, R: np.ndarray):
        self.x = x0.copy()
        self.P = P0.copy()
        self.Q = Q
        self.R = R

    # ── 静态工具方法 ──────────────────────────────────────────────────────────

    @staticmethod
    def measure(X_p: np.ndarray, X_e: np.ndarray, angle_only: bool = False) -> np.ndarray:
        """从 2D 绝对状态计算测量值。

        angle_only=False: [ρ, θ]  (km, rad)
        angle_only=True:  [θ]     (rad)
        """
        dx = X_p[:2] - X_e[:2]
        rho = np.linalg.norm(dx)
        theta = np.arctan2(dx[1], dx[0])
        return np.array([theta]) if angle_only else np.array([rho, theta])

    @staticmethod
    def wrap_angle(a: np.ndarray) -> np.ndarray:
        """将角度归一化到 [-π, π]。"""
        return (a + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def meas_jacobian(x_rel: np.ndarray, angle_only: bool = False) -> np.ndarray:
        """测量方程雅可比矩阵。angle_only=True 返回 1×4，否则 2×4。"""
        dx, dy = x_rel[0], x_rel[1]
        rho2 = dx**2 + dy**2 + 1e-12

        if angle_only:
            H = np.zeros((1, 4))
            H[0, 0] = -dy / rho2
            H[0, 1] = dx / rho2
        else:
            rho = np.sqrt(rho2)
            H = np.zeros((2, 4))
            H[0, 0] = dx / rho
            H[0, 1] = dy / rho
            H[1, 0] = -dy / rho2
            H[1, 1] = dx / rho2
        return H

    # ── 核心滤波步骤 ──────────────────────────────────────────────────────────

    def predict(self, A: np.ndarray, B: np.ndarray,
                u_p: np.ndarray, u_e: np.ndarray, dt: float) -> tuple:
        """EKF 预测步。

        Returns
        -------
        x_priori : (4,) ndarray
        P_priori : (4, 4) ndarray
        """
        F = np.eye(4) + A * dt
        x_priori = F @ self.x + dt * B @ (u_p - u_e)
        P_priori = F @ self.P @ F.T + self.Q
        return x_priori, P_priori

    def update(self, x_priori: np.ndarray, P_priori: np.ndarray,
               z_meas: np.ndarray) -> np.ndarray:
        """EKF 更新步。R 为 1×1 时自动切换仅测角模式，2×2 时为 [ρ, θ] 模式。"""
        angle_only = (self.R.shape[0] == 1)

        rho_p = np.linalg.norm(x_priori[:2]) + 1e-12
        theta_p = np.arctan2(x_priori[1], x_priori[0])
        z_pred = np.array([theta_p]) if angle_only else np.array([rho_p, theta_p])

        y_innov = z_meas - z_pred
        y_innov[-1] = self.wrap_angle(y_innov[-1])

        H = self.meas_jacobian(x_priori, angle_only=angle_only)
        S = H @ P_priori @ H.T + self.R
        K = P_priori @ H.T @ np.linalg.inv(S)

        self.x = x_priori + K @ y_innov
        self.P = (np.eye(4) - K @ H) @ P_priori
        return y_innov

    def step(self, A: np.ndarray, B: np.ndarray,
             u_p: np.ndarray, u_e: np.ndarray, dt: float,
             z_meas: np.ndarray) -> np.ndarray:
        """预测 + 更新一步。

        Returns
        -------
        y_innov : ndarray
        """
        x_priori, P_priori = self.predict(A, B, u_p, u_e, dt)
        return self.update(x_priori, P_priori, z_meas)
