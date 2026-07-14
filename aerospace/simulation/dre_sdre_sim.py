"""
DRE-SDRE闭环仿真引擎

对比 ARE-SDRE (infinite-horizon) vs DRE-SDRE (finite-horizon)
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Optional
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aerospace.dynamics.nerm import OrbitalDynamics
from aerospace.control.dre_sdre import DRESDREController


@dataclass
class DRESimResult:
    """DRE-SDRE仿真结果"""
    t: np.ndarray  # (N,) 时间
    states: np.ndarray  # (13, N) [X_p(6), X_e(6), nu(1)]
    u_p_history: np.ndarray  # (3, N) 追逐者控制
    u_e_history: np.ndarray  # (3, N) 逃逸者控制
    dist_history: np.ndarray  # (N,) 距离
    captured: bool
    soft_captured: bool
    T_capture: float
    wall_time: float
    controller_type: str  # 'DRE' or 'ARE'


class DRESDRESimulation:
    """DRE-SDRE闭环仿真引擎"""

    def __init__(
        self,
        dynamics: OrbitalDynamics,
        controller: DRESDREController,
        X_p0: np.ndarray,
        X_e0: np.ndarray,
        nu0: float,
        dt: float = 20.0,
        capture_dist: float = 0.1,
        soft_speed: float = 1e-4,
        rng: Optional[np.random.Generator] = None,
        controller_type: str = 'DRE',
    ):
        """
        Parameters
        ----------
        dynamics : OrbitalDynamics
            轨道动力学模型
        controller : DRESDREController
            DRE-SDRE控制器
        X_p0 : (6,)
            追逐者初始状态
        X_e0 : (6,)
            逃逸者初始状态
        nu0 : float
            初始真近点角
        dt : float
            控制更新周期（秒）
        capture_dist : float
            捕获距离（km）
        soft_speed : float
            软捕获速度阈值（km/s）
        rng : np.random.Generator
            随机数生成器（保留，未使用）
        controller_type : str
            'DRE' 或 'ARE'
        """
        self.dynamics = dynamics
        self.controller = controller
        self.X_p0 = X_p0.copy()
        self.X_e0 = X_e0.copy()
        self.nu0 = nu0
        self.dt = dt
        self.capture_dist = capture_dist
        self.soft_speed = soft_speed
        self.rng = rng
        self.controller_type = controller_type

    def run(self, t_end: float) -> DRESimResult:
        """运行闭环仿真

        Parameters
        ----------
        t_end : float
            最大仿真时间（秒）

        Returns
        -------
        DRESimResult
        """
        wall_start = time.time()

        # 初始化
        X_p = self.X_p0.copy()
        X_e = self.X_e0.copy()
        nu = self.nu0

        t_hist = [0.0]
        state_hist = [np.concatenate([X_p, X_e, [nu]])]
        u_p_hist = [np.zeros(3)]
        u_e_hist = [np.zeros(3)]
        dist_hist = [float(np.linalg.norm(X_p[:3] - X_e[:3]))]

        captured = False
        soft_captured = False
        t = 0.0
        step = 0
        max_steps = int(t_end / self.dt)

        B = np.vstack([np.zeros((3, 3)), np.eye(3)])

        while step < max_steps:
            # 检查捕获
            dist = float(np.linalg.norm(X_p[:3] - X_e[:3]))
            v_rel = float(np.linalg.norm(X_p[3:] - X_e[3:]))

            if dist < self.capture_dist:
                captured = True
                if v_rel < self.soft_speed:
                    soft_captured = True
                break

            # 获取轨道参数
            r_c, nu_dot, nu_ddot = self.dynamics.get_orbital_params(nu)

            # 构建SDC矩阵
            A_SDC = self.dynamics.get_SDC_matrix(X_p, X_e, r_c, nu_dot, nu_ddot)

            # 计算控制
            x_rel = X_p - X_e

            try:
                if self.controller_type == 'DRE':
                    u_p, u_e = self.controller.compute_control(
                        A_SDC, x_rel, t, recompute=(step % 5 == 0)  # 每5步重新求解DRE
                    )
                else:  # ARE
                    u_p, u_e = self.controller.compute_control_infinite_horizon(
                        A_SDC, x_rel
                    )
            except Exception as e:
                print(f"  Control computation failed at t={t:.1f}s: {e}")
                break

            # 动力学积分
            def dynamics_13d(t_local, state):
                X_p_local = state[:6]
                X_e_local = state[6:12]
                nu_local = state[12]

                r_c_local, nu_dot_local, nu_ddot_local = self.dynamics.get_orbital_params(nu_local)

                def compute_accel(X, u):
                    x, y, z = X[0], X[1], X[2]
                    r = np.sqrt((r_c_local + x)**2 + y**2 + z**2)
                    grav_x = -self.dynamics.mu * (r_c_local + x) / r**3 + self.dynamics.mu / r_c_local**2
                    grav_y = -self.dynamics.mu * y / r**3
                    grav_z = -self.dynamics.mu * z / r**3

                    ax = (2 * nu_dot_local * X[4] + nu_ddot_local * y +
                          nu_dot_local**2 * x + grav_x + u[0])
                    ay = (-2 * nu_dot_local * X[3] - nu_ddot_local * x +
                          nu_dot_local**2 * y + grav_y + u[1])
                    az = grav_z + u[2]

                    return np.array([X[3], X[4], X[5], ax, ay, az])

                dX_p = compute_accel(X_p_local, u_p)
                dX_e = compute_accel(X_e_local, u_e)
                dnu = nu_dot_local

                return np.concatenate([dX_p, dX_e, [dnu]])

            state = np.concatenate([X_p, X_e, [nu]])
            sol = solve_ivp(
                dynamics_13d,
                (t, t + self.dt),
                state,
                method='RK45',
                rtol=1e-8,
                atol=1e-10
            )

            if not sol.success:
                print(f"  Integration failed at t={t:.1f}s")
                break

            state_new = sol.y[:, -1]
            X_p = state_new[:6]
            X_e = state_new[6:12]
            nu = state_new[12]

            t += self.dt
            step += 1

            t_hist.append(t)
            state_hist.append(state_new.copy())
            u_p_hist.append(u_p.copy())
            u_e_hist.append(u_e.copy())
            dist_hist.append(float(np.linalg.norm(X_p[:3] - X_e[:3])))

        wall_time = time.time() - wall_start

        return DRESimResult(
            t=np.array(t_hist),
            states=np.array(state_hist).T,
            u_p_history=np.array(u_p_hist).T,
            u_e_history=np.array(u_e_hist).T,
            dist_history=np.array(dist_hist),
            captured=captured,
            soft_captured=soft_captured,
            T_capture=t if captured else np.nan,
            wall_time=wall_time,
            controller_type=self.controller_type,
        )


if __name__ == "__main__":
    """快速测试：d600初值，DRE vs ARE对比"""
    print("="*80)
    print("DRE-SDRE vs ARE-SDRE Quick Test")
    print("="*80)

    # 初值
    X_p0 = np.array([600.0, 0.0, 0.0, -0.01, 0.0, 0.0])
    X_e0 = np.zeros(6)
    nu0 = 0.0

    # 控制参数
    Q = np.diag([1.65, 1.65, 1.65, 0.35, 0.35, 0.35])
    R = 10**11.699 * np.eye(3)

    # 动力学
    orb = OrbitalDynamics()

    # ARE baseline
    print("\n[1/2] Running ARE-SDRE...")
    controller_are = DRESDREController(Q, R, T_horizon=10000.0)
    sim_are = DRESDRESimulation(
        orb, controller_are, X_p0, X_e0, nu0,
        controller_type='ARE'
    )
    result_are = sim_are.run(t_end=20000)

    print(f"  Captured: {result_are.captured}, Soft: {result_are.soft_captured}")
    print(f"  T = {result_are.T_capture/3600:.3f}h")
    u_norm_are = np.linalg.norm(result_are.u_p_history, axis=0)
    dv_are = float(np.trapezoid(u_norm_are, result_are.t))
    print(f"  Δv = {dv_are:.4f} km/s")
    print(f"  Peak = {np.max(u_norm_are)*1000:.3f} mm/s²")
    print(f"  End = {u_norm_are[-1]*1000:.6f} mm/s²")
    print(f"  Wall time: {result_are.wall_time:.1f}s")

    # DRE
    print("\n[2/2] Running DRE-SDRE...")
    controller_dre = DRESDREController(Q, R, T_horizon=5000.0, receding=True)
    sim_dre = DRESDRESimulation(
        orb, controller_dre, X_p0, X_e0, nu0,
        controller_type='DRE'
    )
    result_dre = sim_dre.run(t_end=20000)

    print(f"  Captured: {result_dre.captured}, Soft: {result_dre.soft_captured}")
    print(f"  T = {result_dre.T_capture/3600:.3f}h")
    u_norm_dre = np.linalg.norm(result_dre.u_p_history, axis=0)
    dv_dre = float(np.trapezoid(u_norm_dre, result_dre.t))
    print(f"  Δv = {dv_dre:.4f} km/s")
    print(f"  Peak = {np.max(u_norm_dre)*1000:.3f} mm/s²")
    print(f"  End = {u_norm_dre[-1]*1000:.6f} mm/s²")
    print(f"  Wall time: {result_dre.wall_time:.1f}s")

    # 对比
    print("\n" + "="*80)
    print("Comparison (DRE / ARE):")
    print("="*80)
    print(f"  Time:  {result_dre.T_capture / result_are.T_capture:.4f}x")
    print(f"  Fuel:  {dv_dre / dv_are:.4f}x")
    print(f"  Peak:  {np.max(u_norm_dre) / np.max(u_norm_are):.4f}x")
    print(f"  End:   {u_norm_dre[-1] / u_norm_are[-1]:.1f}x")
    print("="*80)

    print("\nTest complete!") End:   {u_norm_dre[-1] / u_norm_are[-1]:.1f}x")
    print("="*80)

    print("\nTest complete!")
