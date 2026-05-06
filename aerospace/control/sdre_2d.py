"""
2D SDRE 零和微分博弈控制器

面内 4×4 ARE 求解 + 辛平衡预处理，计算追踪星/逃逸星最优推力。
"""

import numpy as np
from scipy.linalg import solve_continuous_are, LinAlgError


def _format_matrix(M: np.ndarray, name: str, indent: str = "    ") -> str:
    lines = [f"{indent}{name} ="]
    for row in M:
        row_str = "  ".join(f"{v:+12.4e}" for v in row)
        lines.append(f"{indent}  [ {row_str} ]")
    return "\n".join(lines)


class SDREGameController2D:
    """2D 面内 SDRE 博弈控制器。

    Parameters
    ----------
    Q : (4, 4) ndarray  状态权重
    R : (2, 2) ndarray  控制惩罚
    gamma : float       博弈调节因子
    """

    def __init__(
        self,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
        gamma: float = np.sqrt(2),
        verbose: bool = False,
        verbose_interval: int = 1,
    ):
        self.Q = np.eye(4) if Q is None else Q
        self.R = np.eye(2) * 1e13 if R is None else R
        self.gamma = gamma
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        self._step_count = 0

        self.R_inv = np.linalg.inv(self.R)
        self.R_eff_inv = self.R_inv * (1.0 - self.gamma ** (-2))

        # B_p: (4, 2), 加速度通道
        self.B_p = np.zeros((4, 2))
        self.B_p[2:, :] = np.eye(2)
        self.B_e = -self.B_p

        self.last_P: np.ndarray | None = None
        self._are_fallback_count = 0

        self._R_eff = self.R / (1.0 - self.gamma ** (-2))
        self._S = self.B_p @ self.R_eff_inv @ self.B_p.T

    def _solve_are_balanced(self, A: np.ndarray) -> np.ndarray:
        """辛平衡 + 状态缩放两级预处理求解 4×4 ARE。"""
        norm_Q = np.linalg.norm(self.Q, "fro")
        norm_S = np.linalg.norm(self._S, "fro")

        if norm_S < 1e-30:
            raise ValueError("S ≈ 0, ARE 不可解。")

        alpha = np.sqrt(norm_Q / norm_S)
        Q_bal = self.Q / alpha
        R_bal = self._R_eff / alpha

        try:
            P_bar = solve_continuous_are(A, self.B_p, Q_bal, R_bal)
            return alpha * P_bar
        except (LinAlgError, ValueError):
            pass

        n = A.shape[0]
        d = np.empty(n)
        for i in range(n):
            row_max = np.max(np.abs(A[i, :]))
            d[i] = 1.0 / row_max if row_max > 1e-15 else 1.0
        T = np.diag(d)
        T_inv = np.diag(1.0 / d)

        P2_bar = solve_continuous_are(T_inv @ A @ T, T_inv @ self.B_p, T.T @ Q_bal @ T, R_bal)
        return alpha * (T_inv.T @ P2_bar @ T_inv)

    def _log_P_info(self, P, x_rel, u_p, u_e, t=None):
        eigvals = np.linalg.eigvalsh(P)
        cond = np.linalg.cond(P)
        header = f"[SDRE2D Step {self._step_count}]"
        if t is not None:
            header += f"  t = {t:.1f} s"
        print(header)
        print(f"    相对距离: {np.linalg.norm(x_rel[:2]):.4f} km    "
              f"相对速度: {np.linalg.norm(x_rel[2:]):.4e} km/s")
        print(f"    P 特征值: {eigvals}")
        print(f"    P 条件数: {cond:.4e}")
        print(f"    u_p: {u_p}  |u_p|={np.linalg.norm(u_p):.4e}")
        print(f"    u_e: {u_e}  |u_e|={np.linalg.norm(u_e):.4e}")

    def compute_control(
        self,
        A_SDC: np.ndarray,
        x_rel: np.ndarray,
        t: float | None = None,
        solve_are: bool = True,
        x_rel_e: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """计算 2D 面内推力加速度。

        Parameters
        ----------
        A_SDC : (4, 4) ndarray  SDC 系统矩阵
        x_rel : (4,) ndarray    追踪星看到的相对状态（可能是 EKF 估计）
        t : float, optional     当前时间
        solve_are : bool        是否求解 ARE（False 复用缓存 P）
        x_rel_e : (4,) ndarray, optional  逃逸星看到的相对状态（真实值，用于全信息博弈）

        Returns
        -------
        u_p, u_e : (2,) ndarray each
        """
        self._step_count += 1
        n = self.Q.shape[0]

        if solve_are:
            try:
                P = self._solve_are_balanced(A_SDC)
                self.last_P = P
            except (LinAlgError, ValueError) as e:
                if self.last_P is not None:
                    P = self.last_P
                    self._are_fallback_count += 1
                    if self._are_fallback_count <= 5:
                        print(f"Warning: 2D ARE 求解失败 ({e})，沿用上一时刻 P。"
                              f"（第 {self._are_fallback_count} 次回退）")
                    elif self._are_fallback_count == 6:
                        print("Warning: 后续 ARE 回退不再逐条打印。")
                else:
                    print("Error: 初始 ARE 即求解失败。")
                    P = np.zeros((n, n))
        else:
            if self.last_P is None:
                raise RuntimeError("ARE 尚未初始化。")
            P = self.last_P

        x_e_ctrl = x_rel_e if x_rel_e is not None else x_rel
        u_p = -self.R_inv @ self.B_p.T @ P @ x_rel
        u_e = self.gamma ** (-2) * self.R_inv @ self.B_e.T @ P @ x_e_ctrl

        if self.verbose and (self._step_count % self.verbose_interval == 0):
            self._log_P_info(P, x_rel, u_p, u_e, t)

        return u_p, u_e

    def print_P_matrix(self, label: str = "") -> None:
        if self.last_P is None:
            print("P 矩阵尚未计算。")
            return
        title = f"P 矩阵{' — ' + label if label else ''}"
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print("=" * 60)
        print(_format_matrix(self.last_P, "P"))
        eigvals = np.linalg.eigvalsh(self.last_P)
        print(f"    特征值: {eigvals}")
        print(f"    条件数: {np.linalg.cond(self.last_P):.6e}")
        print("=" * 60 + "\n")
