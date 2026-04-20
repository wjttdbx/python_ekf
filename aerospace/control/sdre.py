"""
SDRE 零和微分博弈控制器

在每个时间步解算代数黎卡提方程（ARE），计算追踪星和逃逸星的最优控制律。
"""

import time
import numpy as np
from scipy.linalg import solve_continuous_are, LinAlgError


def _format_matrix(M: np.ndarray, name: str, indent: str = "    ") -> str:
    """将矩阵格式化为易读的多行字符串。"""
    lines = [f"{indent}{name} ="]
    for row in M:
        row_str = "  ".join(f"{v:+12.4e}" for v in row)
        lines.append(f"{indent}  [ {row_str} ]")
    return "\n".join(lines)


class SDREGameController:
    """基于状态依赖黎卡提方程的最优控制器。

    Parameters
    ----------
    Q : (6, 6) ndarray
        状态权重矩阵
    R : (3, 3) ndarray
        控制惩罚矩阵
    gamma : float
        博弈调节因子，默认 sqrt(2)
    verbose : bool
        是否在每次 ARE 求解后打印 P 矩阵关键信息，默认 False
    verbose_interval : int
        每隔多少步输出一次详细信息（仅当 verbose=True 时有效），默认 1
    """

    def __init__(self, Q: np.ndarray = None, R: np.ndarray = None,
                 gamma: float = np.sqrt(2), verbose: bool = False,
                 verbose_interval: int = 1):
        if Q is None:
            self.Q = np.diag([1.0, 1.0, 1.0, 10.0, 10.0, 10.0])
        else:
            self.Q = Q
            
        if R is None:
            self.R = np.eye(3) * 1e13
        else:
            self.R = R
            
        self.gamma = gamma
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        self._step_count = 0

        self.R_inv = np.linalg.inv(self.R)
        
        # 因为 Be = -Bp，有效控制权重 R_eff = R / (1 - gamma^-2)
        # R_eff^-1 = R^-1 * (1 - gamma^-2)
        self.R_eff_inv = self.R_inv * (1.0 - self.gamma**(-2))
        
        # 控制输入矩阵
        self.B_p = np.zeros((6, 3))
        self.B_p[3:, :] = np.eye(3)
        self.B_e = -self.B_p
        
        self.last_P = None
        self._are_fallback_count = 0
        self.step_times: list[float] = []  # 每步"环境数据→控制量"耗时 (s)

        # 预计算 R_eff 和有效增益矩阵 S = B R_eff^{-1} B^T
        self._R_eff = self.R / (1.0 - self.gamma**(-2))
        self._S = self.B_p @ self.R_eff_inv @ self.B_p.T

    def _solve_are_balanced(self, A: np.ndarray) -> np.ndarray:
        """使用辛平衡（Symplectic Balancing）求解 ARE。

        当 ||Q|| 与 ||S|| = ||B R_eff⁻¹ B^T|| 之间量级差距过大时，
        哈密顿矩阵条件数极高，导致 LAPACK Schur 分解失败。
        辛平衡通过标量缩放 α = √(||Q||/||S||) 将两者拉至同一量级：
            Q' = Q/α,  R' = R_eff/α  →  P = α·P'
        """
        norm_Q = np.linalg.norm(self.Q, "fro")
        norm_S = np.linalg.norm(self._S, "fro")

        if norm_S < 1e-30:
            raise ValueError("有效增益 S = B R_eff⁻¹ B^T 接近零，ARE 不可解。")

        alpha = np.sqrt(norm_Q / norm_S)

        Q_bal = self.Q / alpha
        R_bal = self._R_eff / alpha

        try:
            P_bar = solve_continuous_are(A, self.B_p, Q_bal, R_bal)
            return alpha * P_bar
        except (LinAlgError, ValueError):
            pass

        # 辛平衡仍失败时，尝试对 A 做行列缩放后再求解
        d = np.empty(6)
        for i in range(6):
            row_max = np.max(np.abs(A[i, :]))
            d[i] = 1.0 / row_max if row_max > 1e-15 else 1.0
        T = np.diag(d)
        T_inv = np.diag(1.0 / d)

        A2 = T_inv @ A @ T
        B2 = T_inv @ self.B_p
        Q2 = T.T @ Q_bal @ T

        P2_bar = solve_continuous_are(A2, B2, Q2, R_bal)
        return alpha * (T_inv.T @ P2_bar @ T_inv)

    def _log_P_info(self, P: np.ndarray, x_rel: np.ndarray,
                    u_p: np.ndarray, u_e: np.ndarray, t: float = None) -> None:
        """打印 P 矩阵及相关控制量的关键信息。"""
        eigvals = np.linalg.eigvalsh(P)
        cond = np.linalg.cond(P)
        rel_dist = np.linalg.norm(x_rel[:3])
        rel_vel  = np.linalg.norm(x_rel[3:])

        header = f"[SDRE Step {self._step_count}]"
        if t is not None:
            header += f"  t = {t:.1f} s"
        print(header)
        print(f"    相对距离: {rel_dist:.4f} km    相对速度: {rel_vel:.4e} km/s")
        print(f"    P 矩阵特征值 (升序): {eigvals}")
        print(f"    P 条件数: {cond:.4e}    P 对角线: {np.diag(P)}")
        print(f"    追踪星推力 u_p: {u_p}  (范数 {np.linalg.norm(u_p):.4e} km/s²)")
        print(f"    逃逸星推力 u_e: {u_e}  (范数 {np.linalg.norm(u_e):.4e} km/s²)")

    def compute_control(self, A_SDC: np.ndarray, x_rel: np.ndarray,
                        t: float = None,
                        solve_are: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """计算当前时刻的推力加速度。

        Parameters
        ----------
        A_SDC : (6, 6) ndarray
            当前状态下的系统 SDC 矩阵
        x_rel : (6,) ndarray
            相对状态向量 [x, y, z, vx, vy, vz]
        t : float, optional
            当前仿真时间 (s)，用于日志显示
        solve_are : bool, optional
            是否重新求解 ARE。设为 False 时直接复用 last_P，适用于 ARE 更新频率
            低于控制频率的场景（稀疏 ARE 更新）。默认 True。

        Returns
        -------
        u_p : (3,) ndarray
            追踪星的最优推力加速度
        u_e : (3,) ndarray
            逃逸星的最优推力加速度
        """
        self._step_count += 1
        _t0 = time.perf_counter()

        if solve_are:
            try:
                P = self._solve_are_balanced(A_SDC)
                self.last_P = P
            except (LinAlgError, ValueError) as e:
                if self.last_P is not None:
                    P = self.last_P
                    self._are_fallback_count += 1
                    if self._are_fallback_count <= 5:
                        print(f"Warning: ARE 求解失败 ({e})，沿用上一时刻的 P 矩阵。"
                              f"（第 {self._are_fallback_count} 次回退）")
                    elif self._are_fallback_count == 6:
                        print("Warning: 后续 ARE 回退不再逐条打印。")
                else:
                    print("Error: 初始 ARE 即求解失败，无法继续。")
                    P = np.zeros((6, 6))
        else:
            # 稀疏更新模式：复用缓存的 P 矩阵，仅重新计算反馈控制量
            if self.last_P is None:
                raise RuntimeError(
                    "ARE 尚未初始化，solve_are=False 时无法计算控制量。"
                    "请先以 solve_are=True 完成首次 ARE 求解。"
                )
            P = self.last_P

        # 计算推力：u_p = -R^-1 B_p^T P x
        u_p = - self.R_inv @ self.B_p.T @ P @ x_rel
        
        # u_e = gamma^-2 R^-1 B_e^T P x
        u_e = self.gamma**(-2) * self.R_inv @ self.B_e.T @ P @ x_rel

        self.step_times.append(time.perf_counter() - _t0)

        if self.verbose and (self._step_count % self.verbose_interval == 0):
            self._log_P_info(P, x_rel, u_p, u_e, t)

        return u_p, u_e

    def print_P_matrix(self, label: str = "") -> None:
        """以格式化方式打印当前保存的 P 矩阵完整内容。

        Parameters
        ----------
        label : str
            可选标题说明
        """
        if self.last_P is None:
            print("P 矩阵尚未计算。")
            return
        title = f"P 矩阵{' — ' + label if label else ''}"
        print(f"\n{'='*60}")
        print(f"  {title}")
        print('='*60)
        print(_format_matrix(self.last_P, "P"))
        eigvals = np.linalg.eigvalsh(self.last_P)
        print(f"    特征值: {eigvals}")
        print(f"    条件数: {np.linalg.cond(self.last_P):.6e}")
        print('='*60 + "\n")
