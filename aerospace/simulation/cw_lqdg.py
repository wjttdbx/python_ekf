"""
追逃博弈闭环仿真引擎

将 LQDG 控制器与 CW 动力学模型结合，进行连续时间数值积分仿真。
支持推力饱和限制，记录完整的状态与推力历史。
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from aerospace.dynamics.cw import CWDynamics
from aerospace.control.lqdg import ChaserController, EvaderController


@dataclass
class SimResult:
    """仿真结果数据容器。

    Attributes
    ----------
    t          : (N,)    时间序列 (s)
    states     : (6, N)  相对状态历史 x_rel = X_p - X_e  [x,y,z,vx,vy,vz]
    u_p_history: (3, N)  追踪者实际推力加速度历史 (饱和后)
    u_e_history: (3, N)  逃逸者实际推力加速度历史 (饱和后)
    X_p_history: (3, N)  追踪者在 LVLH 系中的绝对位置历史 (m)
    X_e_history: (3, N)  逃逸者在 LVLH 系中的绝对位置历史 (m)
    """
    t: np.ndarray
    states: np.ndarray
    u_p_history: np.ndarray
    u_e_history: np.ndarray
    X_p_history: np.ndarray
    X_e_history: np.ndarray


class GameSimulation:
    """追逃博弈闭环仿真。

    Parameters
    ----------
    cw_dynamics   : CWDynamics       CW 动力学模型
    chaser_ctrl   : ChaserController 追踪者控制器
    evader_ctrl   : EvaderController 逃逸者控制器
    x0            : (6,) ndarray     初始相对状态 [x, y, z, vx, vy, vz] (m, m/s)
    t_span        : tuple            仿真时间区间 (t0, tf) (s)
    dt            : float            输出采样间隔 (s)
    thrust_max_p  : float            追踪者最大推力加速度 (m/s²)
    thrust_max_e  : float            逃逸者最大推力加速度 (m/s²)
    """

    def __init__(
        self,
        cw_dynamics: CWDynamics,
        chaser_ctrl: ChaserController,
        evader_ctrl: EvaderController,
        x0: np.ndarray,
        t_span: tuple[float, float] = (0, 3600),
        dt: float = 1.0,
        thrust_max_p: float = 0.01,
        thrust_max_e: float = 0.005,
        x_e0: np.ndarray | None = None,
    ):
        self.A, self.B = cw_dynamics.get_state_space()
        self.chaser_ctrl = chaser_ctrl
        self.evader_ctrl = evader_ctrl
        self.x0 = np.asarray(x0, dtype=float)
        self.t_span = t_span
        self.dt = dt
        self.thrust_max_p = thrust_max_p
        self.thrust_max_e = thrust_max_e
        # 逃逸者初始 LVLH 6D 状态（默认在主星位置/速度处）
        self.x_e0 = np.zeros(6) if x_e0 is None else np.asarray(x_e0, dtype=float)

    @staticmethod
    def _saturate(u: np.ndarray, u_max: float) -> np.ndarray:
        """推力饱和限制（按向量模截断，保持方向不变）。

        若 ||u|| > u_max，则 u_sat = u_max * u / ||u||
        """
        norm = np.linalg.norm(u)
        if norm > u_max:
            return u_max * u / norm
        return u

    def _dynamics(self, t: float, x: np.ndarray) -> np.ndarray:
        """12D ODE 右端函数：状态 = [x_rel(6), X_e(6)]

        ẋ_rel = A · x_rel + B · (u_p − u_e)
        ẋ_e   = A · X_e   + B · u_e
        """
        x_rel = x[:6]
        X_e   = x[6:]

        u_p = self._saturate(self.chaser_ctrl.compute_control(x_rel), self.thrust_max_p)
        u_e = self._saturate(self.evader_ctrl.compute_control(x_rel), self.thrust_max_e)

        return np.concatenate([
            self.A @ x_rel + self.B @ (u_p - u_e),
            self.A @ X_e   + self.B @ u_e,
        ])

    def run(self) -> SimResult:
        """执行闭环仿真。

        Returns
        -------
        SimResult
            包含时间、状态和推力历史的仿真结果
        """
        t_eval = np.arange(self.t_span[0], self.t_span[1], self.dt)
        if t_eval[-1] < self.t_span[1]:
            t_eval = np.append(t_eval, self.t_span[1])

        x0_12 = np.concatenate([self.x0, self.x_e0])

        sol = solve_ivp(
            self._dynamics,
            self.t_span,
            x0_12,
            method="RK45",
            t_eval=t_eval,
            rtol=1e-10,
            atol=1e-12,
        )

        if not sol.success:
            raise RuntimeError(f"数值积分失败: {sol.message}")

        x_rel_hist = sol.y[:6, :]   # (6, N) 相对状态
        X_e_hist   = sol.y[6:, :]   # (6, N) 逃逸者绝对状态
        X_p_hist   = X_e_hist + x_rel_hist  # (6, N) 追踪者绝对状态

        # 回放计算各时刻的实际推力（含饱和）
        n_steps = sol.t.size
        u_p_history = np.zeros((3, n_steps))
        u_e_history = np.zeros((3, n_steps))

        for i in range(n_steps):
            x_rel = x_rel_hist[:, i]
            u_p = self._saturate(self.chaser_ctrl.compute_control(x_rel), self.thrust_max_p)
            u_e = self._saturate(self.evader_ctrl.compute_control(x_rel), self.thrust_max_e)
            u_p_history[:, i] = u_p
            u_e_history[:, i] = u_e

        return SimResult(
            t=sol.t,
            states=x_rel_hist,
            u_p_history=u_p_history,
            u_e_history=u_e_history,
            X_p_history=X_p_hist[:3, :],   # 仅位置 (3, N)
            X_e_history=X_e_hist[:3, :],   # 仅位置 (3, N)
        )
