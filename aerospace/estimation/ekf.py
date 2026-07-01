"""
Extended Kalman Filter for Relative State Estimation

基于距离-方位角-仰角测量的相对状态估计器。
"""


import numpy as np


class RelativeStateEKF:
    """相对状态扩展卡尔曼滤波器。

    测量模型：距离-方位角-仰角 [ρ, az, el] 或 方位角-仰角 [az, el]
    状态：相对位置和速度 [dx, dy, dz, dvx, dvy, dvz]

    Parameters
    ----------
    x0 : (6,) ndarray  初始相对状态估计
    P0 : (6, 6) ndarray  初始协方差矩阵
    Q  : (6, 6) ndarray  过程噪声协方差矩阵
    R  : (m, m) ndarray  测量噪声协方差矩阵 (m=3 或 m=2)
    angles_only : bool 是否仅使用角度测量 (仅 [az, el])
    """

    def __init__(self, x0: np.ndarray, P0: np.ndarray,
                 Q: np.ndarray, R: np.ndarray, angles_only: bool = False,
                 use_doppler: bool = False):
        self.x = x0.copy()
        self.P = P0.copy()
        self.Q = Q
        self.R = R
        self.angles_only = angles_only
        self.use_doppler = use_doppler

    @property
    def _angle_indices(self) -> tuple[int, int]:
        """返回 (az, el) 在测量向量中的索引。"""
        return (0, 1) if self.angles_only else (1, 2)

    # ── 静态工具方法 ──────────────────────────────────────────────────────────

    @staticmethod
    def measure_range_rate(x_rel: np.ndarray) -> float:
        """从相对状态计算距变率 (Doppler)。

        ρ̇ = (r · v) / ρ，其中 r = x_rel[:3], v = x_rel[3:].
        """
        r = x_rel[:3]
        v = x_rel[3:]
        rho = np.linalg.norm(r) + 1e-12
        return float(np.dot(r, v) / rho)

    @staticmethod
    def measure(X_p: np.ndarray, X_e: np.ndarray, angle_only: bool = False,
                use_doppler: bool = False) -> np.ndarray:
        """从绝对状态计算测量值。
        angle_only=False: [ρ, az, el]  (km, rad, rad)
        angle_only=True:  [az, el]     (rad, rad)  — 对应 Source/MotionModel.cpp mode=2
        """
        dx = X_p[:3] - X_e[:3]
        rho = np.linalg.norm(dx)
        az = np.arctan2(dx[1], dx[0])
        el = np.arcsin(np.clip(dx[2] / (rho + 1e-12), -1, 1))
        z = np.array([az, el]) if angle_only else np.array([rho, az, el])
        if use_doppler:
            rho_dot = RelativeStateEKF.measure_range_rate(
                np.concatenate([X_p[:3] - X_e[:3], X_p[3:] - X_e[3:]]))
            z = np.append(z, rho_dot)
        return z

    @staticmethod
    def wrap_angle(a: np.ndarray) -> np.ndarray:
        """将角度归一化到 [-π, π]。"""
        return (a + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def meas_jacobian(x_rel: np.ndarray, angle_only: bool = False,
                      use_doppler: bool = False) -> np.ndarray:
        """测量方程雅可比矩阵。

        angle_only=True  → 基础 2×6 ([az, el])
        angle_only=False → 基础 3×6 ([ρ, az, el])
        use_doppler=True → 追加距变率 ∂ρ̇/∂x 行
        """
        dx, dy, dz = x_rel[0], x_rel[1], x_rel[2]
        dvx, dvy, dvz = x_rel[3], x_rel[4], x_rel[5]
        rho = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-12
        rho_xy = np.sqrt(dx**2 + dy**2) + 1e-12

        if angle_only:
            n_rows = 2 + (1 if use_doppler else 0)
            H = np.zeros((n_rows, 6))
            H[0, 0] = -dy / rho_xy**2
            H[0, 1] =  dx / rho_xy**2
            H[1, 0] = -dx * dz / (rho**2 * rho_xy)
            H[1, 1] = -dy * dz / (rho**2 * rho_xy)
            H[1, 2] =  rho_xy / rho**2
        else:
            n_rows = 3 + (1 if use_doppler else 0)
            H = np.zeros((n_rows, 6))
            H[0, 0] = dx / rho
            H[0, 1] = dy / rho
            H[0, 2] = dz / rho
            H[1, 0] = -dy / rho_xy**2
            H[1, 1] =  dx / rho_xy**2
            H[2, 0] = -dx * dz / (rho**2 * rho_xy)
            H[2, 1] = -dy * dz / (rho**2 * rho_xy)
            H[2, 2] =  rho_xy / rho**2

        if use_doppler:
            rho_dot = float(np.dot(x_rel[:3], x_rel[3:]) / rho)
            # ∂ρ̇/∂r_i = (v_i − ρ̇·r_i/ρ) / ρ
            for i in range(3):
                H[-1, i] = (x_rel[3 + i] - rho_dot * x_rel[i] / rho) / rho
            # ∂ρ̇/∂v_i = r_i / ρ
            for i in range(3):
                H[-1, 3 + i] = x_rel[i] / rho
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
        """EKF 更新步。角度分量自动归一化到 [-π, π]。"""
        rho_p  = np.linalg.norm(x_priori[:3]) + 1e-12
        az_p   = np.arctan2(x_priori[1], x_priori[0])
        el_p   = np.arcsin(np.clip(x_priori[2] / rho_p, -1, 1))
        z_pred = self._build_measurement_prediction(rho_p, az_p, el_p, x_priori)

        y_innov = z_meas - z_pred
        ia, ie = self._angle_indices
        y_innov[ia] = self.wrap_angle(np.atleast_1d(y_innov[ia]))[0]
        y_innov[ie] = self.wrap_angle(np.atleast_1d(y_innov[ie]))[0]

        H = self.meas_jacobian(x_priori, angle_only=self.angles_only,
                               use_doppler=self.use_doppler)
        S = H @ P_priori @ H.T + self.R
        K = P_priori @ H.T @ np.linalg.inv(S)

        self.x = x_priori + K @ y_innov
        self.P = (np.eye(6) - K @ H) @ P_priori
        return y_innov

    def _build_measurement_prediction(self, rho_p: float, az_p: float,
                                       el_p: float, x_rel: np.ndarray) -> np.ndarray:
        """组装预测测量向量，与 measure() 结构一致。x_rel 为预测后的先验状态。"""
        if self.angles_only:
            z = np.array([az_p, el_p])
        else:
            z = np.array([rho_p, az_p, el_p])
        if self.use_doppler:
            rho_dot_p = float(np.dot(x_rel[:3], x_rel[3:]) / (rho_p + 1e-12))
            z = np.append(z, rho_dot_p)
        return z

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
