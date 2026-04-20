"""
NERM + EKF + SDRE 航天器追逃博弈仿真

追踪星初始状态: X_p(0) = [500, 500, 500, 0.01, 0.01, 0.01]^T  (km, km/s)
逃逸星初始状态: X_e(0) = [0, 0, 0, 0, 0, 0]^T
测量噪声: 距离 ~10 km, 角度 ~1e-4 deg
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are, LinAlgError

from dynamics.nerm import OrbitalDynamics
from control.sdre import SDREGameController

import zhplot

# ─── 常量 ────────────────────────────────────────────────────────────────────
DEG2RAD = np.pi / 180.0

# ─── 测量函数 ─────────────────────────────────────────────────────────────────

def measure(X_p: np.ndarray, X_e: np.ndarray) -> np.ndarray:
    """从相对位置计算距离-方位角-仰角测量值。"""
    dx = X_p[:3] - X_e[:3]
    rho = np.linalg.norm(dx)
    az  = np.arctan2(dx[1], dx[0])
    el  = np.arcsin(np.clip(dx[2] / (rho + 1e-12), -1, 1))
    return np.array([rho, az, el])


def wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def meas_jacobian(x_rel: np.ndarray) -> np.ndarray:
    """测量方程对相对状态 [dx,dy,dz,dvx,dvy,dvz] 的雅可比矩阵 H (3×6)。"""
    dx, dy, dz = x_rel[0], x_rel[1], x_rel[2]
    rho = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-12
    rho_xy = np.sqrt(dx**2 + dy**2) + 1e-12

    H = np.zeros((3, 6))
    # d(rho)/d(pos)
    H[0, 0] = dx / rho
    H[0, 1] = dy / rho
    H[0, 2] = dz / rho
    # d(az)/d(pos)
    H[1, 0] = -dy / rho_xy**2
    H[1, 1] =  dx / rho_xy**2
    # d(el)/d(pos)
    H[2, 0] = -dx * dz / (rho**2 * rho_xy)
    H[2, 1] = -dy * dz / (rho**2 * rho_xy)
    H[2, 2] =  rho_xy / rho**2
    return H

# ─── RK4 积分 ─────────────────────────────────────────────────────────────────

def rk4_step(f, t: float, y: np.ndarray, dt: float, *args) -> np.ndarray:
    k1 = f(t,          y,          *args)
    k2 = f(t + dt/2,   y + dt/2*k1, *args)
    k3 = f(t + dt/2,   y + dt/2*k2, *args)
    k4 = f(t + dt,     y + dt*k3,   *args)
    return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

# ─── 主仿真 ───────────────────────────────────────────────────────────────────

def run_simulation():
    rng = np.random.default_rng(42)

    # 轨道动力学
    orb = OrbitalDynamics(mu=3.986e5, a_c=15000.0, e_c=0.5)
    T_orbit = orb.T_orbit

    # 仿真参数
    dt = 5.0                          # 时间步长 (s)
    t_end = 2.0 * T_orbit             # 仿真时长 (2 个轨道周期)
    N = int(t_end / dt)
    are_interval = 5                  # 每 5 步重解一次 ARE

    # 初始状态
    X_p0 = np.array([500.0, 500.0, 500.0, 0.01, 0.01, 0.01])
    X_e0 = np.array([0.0,   0.0,   0.0,   0.0,  0.0,  0.0 ])
    nu0  = 0.0

    state_true = np.concatenate([X_p0, X_e0, [nu0]])  # 13D

    # 推力限幅
    u_p_max = 1.2e-3   # km/s²
    u_e_max = 2.0e-4   # km/s²
    evader_ratio = 0.35

    # 测量噪声标准差
    sigma_rho = 10.0 / 1000.0          # 10 km → 0.01 km (统一单位 km)
    sigma_ang = 1e-4 * DEG2RAD         # 1e-4 deg → rad

    R_meas = np.diag([sigma_rho**2, sigma_ang**2, sigma_ang**2])

    # 过程噪声
    Q_proc = np.diag([5e-4, 5e-4, 5e-4, 5e-8, 5e-8, 5e-8])

    # EKF 初始化（相对状态 x_rel = X_p - X_e）
    x_est = X_p0 - X_e0          # 初始估计 = 真值（无先验误差）
    P_ekf = np.diag([1.0, 1.0, 1.0, 1e-4, 1e-4, 1e-4])

    # SDRE 控制器
    Q_ctrl = np.diag([1.0, 1.0, 1.0, 10.0, 10.0, 10.0])
    R_ctrl = np.eye(3) * 1e13
    ctrl = SDREGameController(Q=Q_ctrl, R=R_ctrl, gamma=np.sqrt(2))

    # 记录历史
    hist_t      = np.zeros(N + 1)
    hist_pos_p  = np.zeros((N + 1, 3))
    hist_pos_e  = np.zeros((N + 1, 3))
    hist_dist   = np.zeros(N + 1)
    hist_ekf_err = np.zeros(N + 1)
    hist_nu     = np.zeros(N + 1)   # 真近点角，用于惯性系转换

    def record(k, t, st, x_est_):
        hist_t[k]       = t
        hist_pos_p[k]   = st[0:3]
        hist_pos_e[k]   = st[6:9]
        hist_nu[k]      = st[12]
        hist_dist[k]    = np.linalg.norm(st[0:3] - st[6:9])
        hist_ekf_err[k] = np.linalg.norm(x_est_[:3] - (st[0:3] - st[6:9]))

    t = 0.0
    record(0, t, state_true, x_est)

    u_p = np.zeros(3)
    u_e = np.zeros(3)

    for k in range(N):
        nu = state_true[12]
        r_c, nu_dot, nu_ddot = orb.get_orbital_params(nu)

        # ── SDRE 控制（基于 EKF 估计）──────────────────────────────────────
        X_p_est = x_est + state_true[6:12]   # 估计追踪星绝对状态（近似）
        X_e_est = state_true[6:12]            # 逃逸星绝对状态（真值，控制器视角）
        A_SDC = orb.get_SDC_matrix(X_p_est, X_e_est, r_c, nu_dot, nu_ddot)

        solve_now = (k % are_interval == 0)
        u_p_raw, u_e_raw = ctrl.compute_control(A_SDC, x_est, t=t, solve_are=solve_now)

        # 推力限幅
        norm_p = np.linalg.norm(u_p_raw)
        u_p = u_p_raw * min(1.0, u_p_max / (norm_p + 1e-15))

        norm_e = np.linalg.norm(u_e_raw)
        u_e = u_e_raw * evader_ratio * min(1.0, u_e_max / (norm_e + 1e-15))

        # ── 真实状态传播（RK4）────────────────────────────────────────────
        state_true = rk4_step(orb.dynamics_13d, t, state_true, dt, u_p, u_e)
        t += dt

        # ── EKF 预测步 ────────────────────────────────────────────────────
        nu_new = state_true[12]
        r_c2, nu_dot2, nu_ddot2 = orb.get_orbital_params(nu_new)

        # 用估计状态构建 SDC 矩阵作为线性化矩阵
        X_p_pred = x_est + state_true[6:12]
        A_ekf = orb.get_SDC_matrix(X_p_pred, state_true[6:12], r_c2, nu_dot2, nu_ddot2)

        B_ctrl = np.zeros((6, 3))
        B_ctrl[3:, :] = np.eye(3)

        # 线性化传播
        x_pred = x_est + dt * (A_ekf @ x_est + B_ctrl @ (u_p - u_e))
        P_pred = A_ekf @ P_ekf @ A_ekf.T + Q_proc

        # ── EKF 更新步 ────────────────────────────────────────────────────
        X_p_true = state_true[0:6]
        X_e_true = state_true[6:12]

        z_true = measure(X_p_true, X_e_true)
        # 加测量噪声
        noise = rng.multivariate_normal(np.zeros(3), R_meas)
        z_meas = z_true + noise

        # 预测测量（基于估计相对状态）
        x_rel_pred = x_pred
        rho_p = np.linalg.norm(x_rel_pred[:3]) + 1e-12
        az_p  = np.arctan2(x_rel_pred[1], x_rel_pred[0])
        el_p  = np.arcsin(np.clip(x_rel_pred[2] / rho_p, -1, 1))
        z_pred = np.array([rho_p, az_p, el_p])

        y_innov = z_meas - z_pred
        y_innov[1:3] = wrap_angle(y_innov[1:3])

        H = meas_jacobian(x_rel_pred)
        S_innov = H @ P_pred @ H.T + R_meas
        K = P_pred @ H.T @ np.linalg.inv(S_innov)

        x_est = x_pred + K @ y_innov
        P_ekf = (np.eye(6) - K @ H) @ P_pred

        record(k + 1, t, state_true, x_est)

        # 提前终止
        if hist_dist[k + 1] < 0.1:
            N_actual = k + 1
            hist_t       = hist_t[:N_actual + 1]
            hist_pos_p   = hist_pos_p[:N_actual + 1]
            hist_pos_e   = hist_pos_e[:N_actual + 1]
            hist_dist    = hist_dist[:N_actual + 1]
            hist_ekf_err = hist_ekf_err[:N_actual + 1]
            hist_nu      = hist_nu[:N_actual + 1]
            print(f"捕获！t = {t:.1f} s，相对距离 = {hist_dist[-1]*1000:.1f} m")
            break
    else:
        print(f"仿真结束，最终相对距离 = {hist_dist[-1]:.3f} km")

    # ─── LVLH → 惯性系转换 ─────────────────────────────────────────────────
    def lvlh_to_inertial(x_lvlh, nu, r_c):
        """将 LVLH 坐标转换到惯性系（以地心为原点）。

        LVLH 坐标系定义：
        - x: 径向（远离地心）
        - y: 沿轨道方向
        - z: 法向（轨道平面外）

        惯性系：参考轨道在 x-y 平面，z 轴垂直轨道平面
        """
        # 参考轨道在惯性系中的位置（椭圆轨道，近地点在 x 轴）
        r_ref_inertial = np.array([r_c * np.cos(nu), r_c * np.sin(nu), 0.0])

        # LVLH → 惯性系旋转矩阵（绕 z 轴旋转 nu）
        cos_nu = np.cos(nu)
        sin_nu = np.sin(nu)
        R = np.array([
            [cos_nu, -sin_nu, 0],
            [sin_nu,  cos_nu, 0],
            [0,       0,      1]
        ])

        # LVLH 相对位置转到惯性系，再加上参考轨道位置
        return r_ref_inertial + R @ x_lvlh

    # 转换追踪星和逃逸星轨迹到惯性系
    pos_p_inertial = np.zeros_like(hist_pos_p)
    pos_e_inertial = np.zeros_like(hist_pos_e)

    for i in range(len(hist_t)):
        r_c, _, _ = orb.get_orbital_params(hist_nu[i])
        pos_p_inertial[i] = lvlh_to_inertial(hist_pos_p[i], hist_nu[i], r_c)
        pos_e_inertial[i] = lvlh_to_inertial(hist_pos_e[i], hist_nu[i], r_c)

    # ─── 绘图 ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("NERM + EKF + SDRE 航天器追逃博弈", fontsize=14, fontweight="bold")

    # 1. 3D 轨迹
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.plot(hist_pos_p[:, 0], hist_pos_p[:, 1], hist_pos_p[:, 2],
             "b-", lw=1.2, label="追踪星")
    ax1.plot(hist_pos_e[:, 0], hist_pos_e[:, 1], hist_pos_e[:, 2],
             "r-", lw=1.2, label="逃逸星")
    ax1.scatter(*hist_pos_p[0],  c="b", s=40, marker="o")
    ax1.scatter(*hist_pos_e[0],  c="r", s=40, marker="o")
    ax1.scatter(*hist_pos_p[-1], c="b", s=60, marker="*")
    ax1.scatter(*hist_pos_e[-1], c="r", s=60, marker="*")
    ax1.set_xlabel("x (km)"); ax1.set_ylabel("y (km)"); ax1.set_zlabel("z (km)")
    ax1.set_title("3D LVLH 轨迹")
    ax1.legend(fontsize=8)

    # 2. 相对距离
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(hist_t / 3600, hist_dist, "k-", lw=1.5)
    ax2.set_xlabel("时间 (h)"); ax2.set_ylabel("相对距离 (km)")
    ax2.set_title("追逃相对距离")
    ax2.grid(True, alpha=0.4)
    ax2.set_yscale("log")

    # 3. EKF 位置估计误差
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(hist_t / 3600, hist_ekf_err * 1000, "g-", lw=1.2)
    ax3.set_xlabel("时间 (h)"); ax3.set_ylabel("EKF 位置误差 (m)")
    ax3.set_title("EKF 估计误差（位置）")
    ax3.grid(True, alpha=0.4)

    # 4. x-y 平面投影
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(hist_pos_p[:, 0], hist_pos_p[:, 1], "b-", lw=1.2, label="追踪星")
    ax4.plot(hist_pos_e[:, 0], hist_pos_e[:, 1], "r-", lw=1.2, label="逃逸星")
    ax4.scatter(hist_pos_p[0, 0],  hist_pos_p[0, 1],  c="b", s=40, marker="o")
    ax4.scatter(hist_pos_e[0, 0],  hist_pos_e[0, 1],  c="r", s=40, marker="o")
    ax4.scatter(hist_pos_p[-1, 0], hist_pos_p[-1, 1], c="b", s=60, marker="*")
    ax4.scatter(hist_pos_e[-1, 0], hist_pos_e[-1, 1], c="r", s=60, marker="*")
    ax4.set_xlabel("x (km)"); ax4.set_ylabel("y (km)")
    ax4.set_title("x-y 平面投影")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.4)
    ax4.set_aspect("equal")

    plt.tight_layout()
    out_path = "nerm_ekf_sdre_trajectory.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"图像已保存至 {out_path}")


if __name__ == "__main__":
    run_simulation()
