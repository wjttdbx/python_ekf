"""
Differential Riccati Equation (DRE) SDRE 控制器

Finite-horizon LQR for time-varying systems:
    min J = x(T)'·S·x(T) + ∫[0,T] (x'Qx + u'Ru) dt
    s.t. dx/dt = A(x)·x + B·u

DRE (反向积分):
    -dP/dt = A'P + PA - PBR⁻¹B'P + Q,  P(T) = S

控制律:
    u*(t) = -R⁻¹B'P(t)·x(t)

关键差异 vs ARE:
- P(t) 是时变的，在接近终端T时增大
- K(t) = R⁻¹B'P(t) 也是时变的
- 即使 x↓，K(t)·x 可以不衰减 → 提供终端制动力
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
from typing import Tuple, Optional


class DRESDREController:
    """DRE-SDRE Finite-Horizon 控制器

    核心思想：
    1. 给定终端时间 T 和当前时刻 t
    2. 反向积分 DRE 从 t 到 T，得到 P(τ) for τ ∈ [t, T]
    3. 使用 P(t) 计算当前控制 u(t) = -K(t)·x(t)

    Parameters
    ----------
    Q : (6, 6)
        状态权重矩阵
    R : (3, 3)
        控制权重矩阵
    S : (6, 6)
        终端状态权重矩阵（默认 = Q）
    T_horizon : float
        时域长度（秒）
    receding : bool
        是否使用receding horizon（每步更新T）
    """

    def __init__(
        self,
        Q: np.ndarray,
        R: np.ndarray,
        S: Optional[np.ndarray] = None,
        T_horizon: float = 10000.0,
        receding: bool = True,
        gamma: float = None,  # 博弈调节因子，None表示非博弈场景
    ):
        self.Q = Q
        self.R = R
        self.S = S if S is not None else Q.copy()
        self.T_horizon = T_horizon
        self.receding = receding
        self.gamma = gamma

        # 验证维度
        assert Q.shape == (6, 6), "Q must be (6,6)"
        assert R.shape == (3, 3), "R must be (3,3)"
        assert self.S.shape == (6, 6), "S must be (6,6)"

        # B矩阵（固定）：推力作用在速度上
        self.B = np.vstack([np.zeros((3, 3)), np.eye(3)])

        # 博弈场景：计算有效R
        if gamma is not None:
            # R_eff = R / (1 - gamma^(-2))
            # 这来自零和博弈：Be = -Bp
            self.R_eff = self.R / (1.0 - self.gamma**(-2))
            self.R_inv = np.linalg.inv(self.R)
            self.R_eff_inv = self.R_inv * (1.0 - self.gamma**(-2))
        else:
            # 非博弈场景：R_eff = R
            self.R_eff = self.R
            self.R_eff_inv = np.linalg.inv(self.R)

        # 缓存：上一次DRE求解结果
        self._P_cache = None
        self._t_cache = None
        self._T_cache = None

    def _solve_are_balanced(self, A: np.ndarray) -> np.ndarray:
        """使用辛平衡求解ARE（借鉴sdre-python的实现）

        辛平衡技术：当||Q||与||S||量级差距大时，通过标量缩放平衡
            α = √(||Q|| / ||S||),  Q' = Q/α,  R' = R_eff/α  →  P = α·P'

        注意：博弈场景下使用 R_eff = R / (1 - gamma^(-2))
        """
        from scipy.linalg import solve_continuous_are, LinAlgError

        # 计算S = B·R_eff⁻¹·B^T（使用有效R）
        S = self.B @ self.R_eff_inv @ self.B.T
        norm_Q = np.linalg.norm(self.Q, "fro")
        norm_S = np.linalg.norm(S, "fro")

        if norm_S < 1e-30:
            raise ValueError("S = B·R_eff⁻¹·B^T 接近零，ARE不可解")

        alpha = np.sqrt(norm_Q / norm_S)
        Q_bal = self.Q / alpha
        R_bal = self.R_eff / alpha  # 使用R_eff

        try:
            P_bar = solve_continuous_are(A, self.B, Q_bal, R_bal)
            return alpha * P_bar
        except (LinAlgError, ValueError):
            pass

        # 第二层fallback：行列缩放
        d = np.empty(6)
        for i in range(6):
            row_max = np.max(np.abs(A[i, :]))
            d[i] = 1.0 / row_max if row_max > 1e-15 else 1.0
        T = np.diag(d)
        T_inv = np.diag(1.0 / d)

        A2 = T_inv @ A @ T
        B2 = T_inv @ self.B
        Q2 = T.T @ Q_bal @ T

        P2_bar = solve_continuous_are(A2, B2, Q2, R_bal)
        return alpha * (T_inv.T @ P2_bar @ T_inv)

    def solve_dre_backward(
        self,
        A: np.ndarray,
        t_current: float,
        T_end: float,
        n_points: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """反向积分DRE（使用辛平衡技术提高数值稳定性）

        从 T_end 反向积分到 t_current，得到 P(τ) for τ ∈ [t_current, T_end]

        Parameters
        ----------
        A : (6, 6)
            当前时刻的SDC矩阵（假设在 [t_current, T_end] 内不变）
        t_current : float
            当前时刻
        T_end : float
            终端时刻
        n_points : int
            时间网格点数

        Returns
        -------
        t_grid : (n_points,)
            时间网格（从 t_current 到 T_end）
        P_traj : (n_points, 6, 6)
            P(t) 轨迹
        """
        # DRE ODE: dP/dt = -A'P - PA + PBR⁻¹B'P - Q
        # 反向积分: 令 τ = T_end - t，则 dP/dτ = -dP/dt
        # 即: dP/dτ = A'P + PA - PBR⁻¹B'P + Q

        # 使用辛平衡缩放（使用R_eff）
        S = self.B @ self.R_eff_inv @ self.B.T
        norm_Q = np.linalg.norm(self.Q, "fro")
        norm_S = np.linalg.norm(S, "fro")
        alpha = np.sqrt(norm_Q / norm_S) if norm_S > 1e-30 else 1.0

        Q_bal = self.Q / alpha
        R_bal = self.R_eff / alpha  # 使用R_eff

        def dre_rhs(tau, P_flat):
            """DRE右端（反向时间，辛平衡版本）

            P_flat: (36,) 扁平化的P矩阵
            """
            P = P_flat.reshape((6, 6))

            # dP/dτ = A'P + PA - PBR_eff⁻¹B'P + Q
            dP = (A.T @ P + P @ A
                  - P @ self.B @ np.linalg.solve(R_bal, self.B.T @ P)
                  + Q_bal)

            return dP.flatten()

        # 初值：P(T_end) = S（也需要缩放）
        S_bal = self.S / alpha
        P0_flat = S_bal.flatten()

        # 反向积分：从 tau=0 到 tau=(T_end - t_current)
        tau_span = (0, T_end - t_current)
        tau_eval = np.linspace(0, T_end - t_current, n_points)

        sol = solve_ivp(
            dre_rhs,
            tau_span,
            P0_flat,
            method='RK45',
            t_eval=tau_eval,
            rtol=1e-6,
            atol=1e-8,
        )

        if not sol.success:
            raise RuntimeError(f"DRE integration failed: {sol.message}")

        # 转换回正向时间并恢复缩放
        t_grid = T_end - sol.t[::-1]  # 反转：从 t_current 到 T_end
        P_traj = alpha * sol.y.T[::-1].reshape((n_points, 6, 6))  # 反转、恢复缩放、reshape

        return t_grid, P_traj

    def compute_control(
        self,
        A: np.ndarray,
        x_rel: np.ndarray,
        t_current: float,
        recompute: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算DRE-SDRE控制

        Parameters
        ----------
        A : (6, 6)
            当前SDC矩阵
        x_rel : (6,)
            当前相对状态
        t_current : float
            当前时刻
        recompute : bool
            是否重新求解DRE（否则使用缓存）

        Returns
        -------
        u_p : (3,)
            追逐者控制
        u_e : (3,)
            逃逸者控制（zeros）
        """
        # 计算终端时间
        if self.receding:
            # Receding horizon: 终端时间总是 t_current + T_horizon
            T_end = t_current + self.T_horizon
        else:
            # Fixed horizon: 终端时间固定
            T_end = self.T_horizon

        # 检查是否需要重新求解DRE
        need_recompute = (
            recompute or
            self._P_cache is None or
            self._T_cache != T_end or
            t_current < self._t_cache[0] or
            t_current > self._t_cache[-1]
        )

        if need_recompute:
            # 求解DRE
            t_grid, P_traj = self.solve_dre_backward(A, t_current, T_end)

            # 缓存
            self._t_cache = t_grid
            self._P_cache = P_traj
            self._T_cache = T_end

        # 插值得到 P(t_current)
        idx = np.searchsorted(self._t_cache, t_current)
        if idx == 0:
            P_current = self._P_cache[0]
        elif idx >= len(self._t_cache):
            P_current = self._P_cache[-1]
        else:
            # 线性插值
            t0, t1 = self._t_cache[idx-1], self._t_cache[idx]
            P0, P1 = self._P_cache[idx-1], self._P_cache[idx]
            alpha = (t_current - t0) / (t1 - t0)
            P_current = (1 - alpha) * P0 + alpha * P1

        # 计算控制律（使用R_eff）
        K = np.linalg.solve(self.R_eff, self.B.T @ P_current)
        u_p = -K @ x_rel
        u_e = np.zeros(3)

        return u_p, u_e

    def compute_control_infinite_horizon(
        self,
        A: np.ndarray,
        x_rel: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算ARE-SDRE控制（用于对比，使用辛平衡技术）

        Parameters
        ----------
        A : (6, 6)
            当前SDC矩阵
        x_rel : (6,)
            当前相对状态

        Returns
        -------
        u_p : (3,)
            追逐者控制
        u_e : (3,)
            逃逸者控制（zeros）
        """
        # 使用辛平衡求解ARE
        try:
            P = self._solve_are_balanced(A)
        except Exception as e:
            # Fallback: 使用scipy默认方法
            from scipy.linalg import solve_continuous_are
            P = solve_continuous_are(A, self.B, self.Q, self.R, balanced=True)

        # 计算控制律（使用R_eff）
        K = np.linalg.solve(self.R_eff, self.B.T @ P)
        u_p = -K @ x_rel
        u_e = np.zeros(3)

        return u_p, u_e


class DRESDREControllerAdaptive(DRESDREController):
    """自适应时域DRE控制器

    根据当前状态自适应调整时域长度 T_horizon：
    - 距离远 → T 长（关注长期）
    - 距离近 → T 短（关注短期，终端制动）
    """

    def __init__(
        self,
        Q: np.ndarray,
        R: np.ndarray,
        S: Optional[np.ndarray] = None,
        T_min: float = 1000.0,
        T_max: float = 20000.0,
        r_ref: float = 100.0,
    ):
        """
        Parameters
        ----------
        T_min : float
            最小时域（秒）
        T_max : float
            最大时域（秒）
        r_ref : float
            参考距离（km），用于自适应调整
        """
        super().__init__(Q, R, S, T_horizon=T_max, receding=True)
        self.T_min = T_min
        self.T_max = T_max
        self.r_ref = r_ref

    def compute_control(
        self,
        A: np.ndarray,
        x_rel: np.ndarray,
        t_current: float,
        recompute: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """自适应时域控制"""
        # 根据距离自适应调整时域
        r = np.linalg.norm(x_rel[:3])

        # T(r) = T_min + (T_max - T_min) * (r / r_ref)
        # 距离远时接近T_max，距离近时接近T_min
        self.T_horizon = self.T_min + (self.T_max - self.T_min) * min(r / self.r_ref, 1.0)

        return super().compute_control(A, x_rel, t_current, recompute)


if __name__ == "__main__":
    """简单测试：对比DRE vs ARE在简单系统上的表现"""

    # 简单1D系统：dx/dt = -x + u
    A = np.array([[-1.0]])
    B = np.array([[1.0]])
    Q = np.array([[1.0]])
    R = np.array([[0.1]])
    S = np.array([[10.0]])  # 终端惩罚

    # DRE
    from scipy.integrate import solve_ivp as sp_solve_ivp

    def dre_rhs(tau, P_flat, A, B, Q, R):
        P = P_flat.reshape((1, 1))
        dP = A.T @ P + P @ A - P @ B @ np.linalg.solve(R, B.T @ P) + Q
        return dP.flatten()

    # 从 T=10 反向积分到 t=0
    T = 10.0
    tau_span = (0, T)
    tau_eval = np.linspace(0, T, 100)

    sol_dre = sp_solve_ivp(
        lambda tau, P: dre_rhs(tau, P, A, B, Q, R),
        tau_span,
        S.flatten(),
        t_eval=tau_eval,
        method='RK45',
    )

    t_dre = T - sol_dre.t[::-1]
    P_dre = sol_dre.y.flatten()[::-1]

    # ARE
    P_are = solve_continuous_are(A, B, Q, R)

    # 绘图对比
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(t_dre, P_dre, 'b-', lw=2, label='DRE: P(t)')
    plt.axhline(P_are[0, 0], color='r', ls='--', lw=2, label=f'ARE: P∞ = {P_are[0,0]:.2f}')
    plt.axhline(S[0, 0], color='g', ls=':', lw=1, alpha=0.5, label=f'Terminal: S = {S[0,0]:.1f}')
    plt.xlabel('Time t', fontsize=12)
    plt.ylabel('P(t)', fontsize=12)
    plt.title('DRE vs ARE: Time-Varying vs Constant Riccati Solution', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    import os
    os.makedirs('outputs/dre_test', exist_ok=True)
    plt.savefig('outputs/dre_test/dre_vs_are_simple.png', dpi=150)
    print("Saved: outputs/dre_test/dre_vs_are_simple.png")

    print(f"\nDRE: P(0) = {P_dre[0]:.4f}, P(T={T}) = {P_dre[-1]:.4f}")
    print(f"ARE: P∞ = {P_are[0,0]:.4f}")
    print(f"Ratio: P_DRE(0) / P_ARE = {P_dre[0] / P_are[0,0]:.4f}x")
    print(f"\nKey insight: DRE's P(t) increases as t→T, providing stronger terminal control")
