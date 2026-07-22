"""
可注入导航器接口 — 支持 SDC-EKF / Jacobian-EKF / Square-Root UKF

BaseNavigator 定义统一接口（predict + update），仿真引擎以 ctx 字典
向 predict 传递非线性传播所需上下文。
"""

from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import cholesky, qr as scipy_qr

from aerospace.estimation.ekf import RelativeStateEKF


# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def _ensure_symmetric(P: np.ndarray) -> np.ndarray:
    """强制对称化，消除数值不对称累积。"""
    return 0.5 * (P + P.T)


def _normalize_upper_factor(S: np.ndarray) -> np.ndarray:
    """Normalize a QR factor to the nonnegative-diagonal Cholesky convention."""
    S_normalized = S.copy().astype(float)
    for i in range(min(S_normalized.shape)):
        if S_normalized[i, i] < 0.0:
            S_normalized[i, :] *= -1.0
    return S_normalized


def _chol_update(S: np.ndarray, v: np.ndarray, sign: str = '+') -> np.ndarray:
    """Cholesky 因子的 rank-1 更新/下更新：S_new'@S_new = S'@S ± v@v'.

    S 为上三角（cholesky(..., lower=False) 输出约定）。
    逐行 Givens 旋转，v 更新使用刚旋转后的 S_new 行（标准公式）。
    """
    if sign not in {'+', '-'}:
        raise ValueError("sign must be '+' or '-'.")

    n = len(v)
    S_new = _normalize_upper_factor(S)
    w = v.copy().astype(float)
    for i in range(n):
        r_old = float(S_new[i, i])
        if sign == '+':
            r_new = np.sqrt(r_old**2 + w[i]**2)
        else:
            radicand = r_old**2 - w[i]**2
            scale = max(r_old**2, w[i]**2, np.finfo(float).tiny)
            if radicand <= np.finfo(float).eps * scale:
                raise np.linalg.LinAlgError(
                    "Cholesky downdate would make the covariance non-positive definite."
                )
            r_new = np.sqrt(radicand)
        if abs(r_old) <= np.finfo(float).tiny:
            raise np.linalg.LinAlgError(
                "Cholesky update encountered a near-singular diagonal factor."
            )
        c = r_new / r_old
        s = w[i] / r_old
        S_new[i, i] = r_new
        if i + 1 < n:
            old_row = S_new[i, i+1:].copy()
            if sign == '+':
                S_new[i, i+1:] = (old_row + s * w[i+1:]) / c
            else:
                S_new[i, i+1:] = (old_row - s * w[i+1:]) / c
            w[i+1:] = c * w[i+1:] - s * S_new[i, i+1:]  # 用更新后的行
    return S_new


def _sqrt_of_semidef(P: np.ndarray) -> np.ndarray:
    """对称半正定矩阵的矩阵平方根（上三角），通过特征分解。

    当 Cholesky 失败时回退使用，对零特征值返回零行。
    """
    eigvals, eigvecs = np.linalg.eigh(P)
    eigvals = np.maximum(eigvals, 0.0)
    sqrt_diag = np.sqrt(eigvals)
    # 构造上三角 sqrt: U s.t. U'@U = P
    # 使用 eigvecs @ diag(sqrt) — 不是上三角，但对 sigma 点生成足够
    L = eigvecs @ np.diag(sqrt_diag)
    # QR 分解得到上三角因子
    Q_sqrt, R_sqrt = scipy_qr(L.T, mode='economic')
    return R_sqrt[:P.shape[0], :P.shape[0]]


def _circular_mean(angles: np.ndarray) -> float:
    """圆周角加权均值（弧度）。

    Parameters
    ----------
    angles : 1D ndarray  角度样本
    """
    return float(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))


def _wrap_angle_scalar(a: float) -> float:
    """单值角度归一化到 [-π, π]。"""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


# ═══════════════════════════════════════════════════════════════════════════════
# 基础抽象类
# ═══════════════════════════════════════════════════════════════════════════════

class BaseNavigator(ABC):
    """导航器抽象基类。

    子类需实现 predict()；SDC-EKF/Jacobian-EKF 共享基类 update()，
    UKF 覆盖 update() 提供完整的 sigma-point 测量更新。

    Attributes
    ----------
    x : (6,) ndarray  当前状态估计
    P : (6,6) ndarray  当前协方差矩阵
    angles_only : bool
    """

    def __init__(self, x0: np.ndarray, P0: np.ndarray,
                 Q: np.ndarray, R: np.ndarray, angles_only: bool = False):
        self._x = x0.copy().astype(float)
        self._P = _ensure_symmetric(P0.copy().astype(float))
        self.Q = Q.copy().astype(float)
        self.R = R.copy().astype(float)
        self.angles_only = angles_only

    # ── 属性 ──────────────────────────────────────────────────────────────

    @property
    def x(self) -> np.ndarray:
        return self._x

    @x.setter
    def x(self, value: np.ndarray) -> None:
        self._x = value.copy().astype(float)

    @property
    def P(self) -> np.ndarray:
        return self._P

    @P.setter
    def P(self, value: np.ndarray) -> None:
        self._P = _ensure_symmetric(value.copy().astype(float))

    # ── 标准 EKF 测量更新（SDC-EKF / Jacobian-EKF 共用）────────────────

    def update(self, x_priori: np.ndarray, P_priori: np.ndarray,
               z_meas: np.ndarray) -> np.ndarray:
        """标准 EKF 测量更新。角度分量自动 wrap 到 [-π, π]."""
        rho_p = np.linalg.norm(x_priori[:3]) + 1e-12
        az_p = np.arctan2(x_priori[1], x_priori[0])
        el_p = np.arcsin(np.clip(x_priori[2] / rho_p, -1, 1))

        z_pred = self._build_measurement_prediction(rho_p, az_p, el_p, x_priori)
        y_innov = z_meas - z_pred

        ia, ie = self._angle_indices
        y_innov[ia] = RelativeStateEKF.wrap_angle(np.atleast_1d(y_innov[ia]))[0]
        y_innov[ie] = RelativeStateEKF.wrap_angle(np.atleast_1d(y_innov[ie]))[0]

        H = RelativeStateEKF.meas_jacobian(x_priori, angle_only=self.angles_only)
        S = H @ P_priori @ H.T + self.R
        self._last_S = S
        K = P_priori @ H.T @ np.linalg.inv(S)

        self.x = x_priori + K @ y_innov
        self.P = _ensure_symmetric((np.eye(6) - K @ H) @ P_priori)
        return y_innov

    def step(self, A_SDC: np.ndarray, B: np.ndarray,
             u_p: np.ndarray, u_e: np.ndarray, dt: float,
             z_meas: np.ndarray, **ctx) -> np.ndarray:
        """预测 + 更新一步。ctx 传递给 predict。"""
        x_priori, P_priori = self.predict(A_SDC, B, u_p, u_e, dt, **ctx)
        return self.update(x_priori, P_priori, z_meas)

    @abstractmethod
    def predict(self, A_SDC: np.ndarray, B: np.ndarray,
                u_p: np.ndarray, u_e: np.ndarray, dt: float,
                **ctx) -> tuple[np.ndarray, np.ndarray]:
        """预测步：返回 (x_priori, P_priori)。"""
        ...

    # ── 内部工具 ──────────────────────────────────────────────────────────

    @property
    def _angle_indices(self) -> tuple[int, int]:
        return (0, 1) if self.angles_only else (1, 2)

    def _build_measurement_prediction(self, rho_p: float, az_p: float,
                                       el_p: float, x_rel: np.ndarray) -> np.ndarray:
        if self.angles_only:
            return np.array([az_p, el_p])
        else:
            return np.array([rho_p, az_p, el_p])

    def _measure_from_rel_state(self, x_rel: np.ndarray) -> np.ndarray:
        """从相对状态构建测量向量（角度或角度+距离）。"""
        rho = np.linalg.norm(x_rel[:3]) + 1e-12
        az = np.arctan2(x_rel[1], x_rel[0])
        el = np.arcsin(np.clip(x_rel[2] / rho, -1, 1))
        if self.angles_only:
            return np.array([az, el])
        else:
            return np.array([rho, az, el])


# ═══════════════════════════════════════════════════════════════════════════════
# SDC-EKF Navigator：F = I + A_SDC * dt
# ═══════════════════════════════════════════════════════════════════════════════

class SDCEKFNavigator(BaseNavigator):
    """F = I + A_SDC * dt 的 EKF。不需要 ctx 中的额外上下文。"""

    def predict(self, A_SDC: np.ndarray, B: np.ndarray,
                u_p: np.ndarray, u_e: np.ndarray, dt: float,
                **ctx) -> tuple[np.ndarray, np.ndarray]:
        F = np.eye(6) + A_SDC * dt
        du = u_p - u_e
        x_priori = F @ self._x + dt * B @ du
        P_priori = _ensure_symmetric(F @ self._P @ F.T + self.Q)
        return x_priori, P_priori


# ═══════════════════════════════════════════════════════════════════════════════
# Jacobian-EKF Navigator：F = ∂g/∂x 中心差分
# ═══════════════════════════════════════════════════════════════════════════════

class JacobianEKFNavigator(BaseNavigator):
    """有限差分 Jacobian EKF。

    对离散映射 g(x) 做中心差分求 F = ∂g/∂x。
    均值传播和协方差传播使用同一个 g(x)。

    ctx 必须包含 f_discrete 或 A_SDC_fn 之一，否则抛出 ValueError。

    Parameters
    ----------
    eps : float             相对差分步长因子
    state_scales : (6,)     每维特征尺度 [pos×3, vel×3]，差分时用 scale[i] 而非 |x_i|
    """

    def __init__(self, x0: np.ndarray, P0: np.ndarray,
                 Q: np.ndarray, R: np.ndarray, angles_only: bool = False,
                 eps: float = 1e-4, state_scales: np.ndarray | None = None):
        super().__init__(x0, P0, Q, R, angles_only)
        self.eps = eps
        self._scales = state_scales  # None → 用 |x_i|
        self._last_F: np.ndarray | None = None

    @property
    def last_F(self) -> np.ndarray | None:
        return self._last_F

    def _build_g(self, A_SDC: np.ndarray, B: np.ndarray,
                 du: np.ndarray, dt: float, ctx: dict):
        """从 ctx 构建离散映射 g(x)。若缺失必要上下文则报错。"""
        f_discrete = ctx.get('f_discrete', None)
        if f_discrete is not None:
            return f_discrete

        A_SDC_fn = ctx.get('A_SDC_fn', None)
        if A_SDC_fn is not None:
            def _g(xx):
                A = A_SDC_fn(xx)
                return xx + dt * (A @ xx + B @ du)
            return _g

        raise ValueError(
            "JacobianEKFNavigator.predict() 需要 ctx['f_discrete'] 或 "
            "ctx['A_SDC_fn']，否则会静默退化为 SDC-EKF。"
        )

    def predict(self, A_SDC: np.ndarray, B: np.ndarray,
                u_p: np.ndarray, u_e: np.ndarray, dt: float,
                **ctx) -> tuple[np.ndarray, np.ndarray]:
        x = self._x
        n = 6
        du = u_p - u_e

        g = self._build_g(A_SDC, B, du, dt, ctx)

        # ── 有限差分 Jacobian ───────────────────────────────────────────
        F = np.zeros((n, n))
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = 1.0
            if self._scales is not None:
                scale = self._scales[i]
            else:
                scale = max(abs(x[i]), 1e-8)
            h = self.eps * scale
            if h < 1e-15:
                h = self.eps * 1e-6  # 绝对最小步长

            g_plus = g(x + h * ei)
            g_minus = g(x - h * ei)
            F[:, i] = (g_plus - g_minus) / (2.0 * h)

        self._last_F = F

        # ── 均值与协方差用同一离散映射 ──────────────────────────────────
        x_priori = g(x)
        P_priori = _ensure_symmetric(F @ self._P @ F.T + self.Q)
        return x_priori, P_priori


# ═══════════════════════════════════════════════════════════════════════════════
# Square-Root UKF Navigator
# ═══════════════════════════════════════════════════════════════════════════════

class SquareRootUKFNavigator(BaseNavigator):
    """平方根 UKF 导航器。

    预测：sigma 点经 f_discrete 传播 → QR + cholupdate → S_priori。
    更新：sigma 点经测量函数 h(x) 传播 → 圆周角均值 → 测量协方差 QR
          → 互协方差 → Kalman 增益 → 平方根状态/协方差更新。

    ctx 必须包含 f_discrete 或 A_SDC_fn 之一，否则抛出 ValueError。

    Parameters
    ----------
    alpha : float   sigma 点散布，默认 1.0
    beta : float    分布先验（高斯最优=2），默认 2.0
    kappa : float   次级缩放，默认 0.0
    """

    def __init__(self, x0: np.ndarray, P0: np.ndarray,
                 Q: np.ndarray, R: np.ndarray, angles_only: bool = False,
                 alpha: float = 1.0, beta: float = 2.0, kappa: float = 0.0):
        super().__init__(x0, P0, Q, R, angles_only)
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        n = 6
        self._n = n
        self._lam = alpha**2 * (n + kappa) - n
        self._gamma = np.sqrt(n + self._lam)

        # UKF 权重
        self._Wm0 = self._lam / (n + self._lam)
        self._Wc0 = self._lam / (n + self._lam) + (1.0 - alpha**2 + beta)
        self._Wi = 1.0 / (2.0 * (n + self._lam))

        # 初始 Cholesky 因子
        try:
            self._S = cholesky(self._P, lower=False)
        except Exception:
            self._S = _sqrt_of_semidef(self._P)

        # 过程噪声平方根 — 允许半正定 Q（E0 无噪声场景 Q=0）
        self._sqrt_Q = self._compute_sqrt_Q()

    def _compute_sqrt_Q(self) -> np.ndarray | None:
        """过程噪声平方根（上三角）。Q=0 时返回 None。"""
        q_norm = np.linalg.norm(self.Q, 'fro')
        if q_norm < 1e-30:
            return None
        try:
            return cholesky(self.Q, lower=False)
        except Exception:
            return _sqrt_of_semidef(self.Q)

    @property
    def P(self) -> np.ndarray:
        return _ensure_symmetric(self._S.T @ self._S)

    @P.setter
    def P(self, value: np.ndarray) -> None:
        P_sym = _ensure_symmetric(value)
        try:
            self._S = cholesky(P_sym, lower=False)
        except Exception:
            self._S = _sqrt_of_semidef(P_sym)
        self._P = P_sym

    # ── Sigma 点工具 ────────────────────────────────────────────────────

    def _sigma_points(self, x: np.ndarray, S: np.ndarray) -> np.ndarray:
        """从均值 x 和上三角 Cholesky 因子 S 生成 (2n+1)×n sigma 点矩阵。"""
        n = self._n
        chi = np.zeros((2 * n + 1, n))
        chi[0] = x
        for i in range(n):
            chi[i + 1] = x + self._gamma * S[i, :]
            chi[i + 1 + n] = x - self._gamma * S[i, :]
        return chi

    # ── 预测 ─────────────────────────────────────────────────────────────

    def predict(self, A_SDC: np.ndarray, B: np.ndarray,
                u_p: np.ndarray, u_e: np.ndarray, dt: float,
                **ctx) -> tuple[np.ndarray, np.ndarray]:
        """平方根 UKF 预测步。

        ctx 必须包含 f_discrete 或 A_SDC_fn，否则抛 ValueError。
        """
        n = self._n
        x = self._x
        S = self._S
        du = u_p - u_e

        f_discrete = self._build_f_discrete(A_SDC, B, du, dt, ctx)

        # 生成 & 传播 sigma 点
        chi = self._sigma_points(x, S)
        chi_prop = np.zeros_like(chi)
        for i in range(2 * n + 1):
            chi_prop[i] = f_discrete(chi[i])

        # 加权均值
        x_priori = self._Wm0 * chi_prop[0]
        for i in range(1, 2 * n + 1):
            x_priori += self._Wi * chi_prop[i]

        # 平方根协方差：QR 分解
        n_qr_rows = 2 * n
        if self._sqrt_Q is not None:
            n_qr_rows += n
        dev = np.zeros((n_qr_rows, n))
        sqrt_Wi = np.sqrt(self._Wi)
        for i in range(1, 2 * n + 1):
            dev[i - 1] = sqrt_Wi * (chi_prop[i] - x_priori)
        if self._sqrt_Q is not None:
            for i in range(n):
                dev[2 * n + i] = self._sqrt_Q[i, :]

        _, R_qr = scipy_qr(dev, mode='economic')
        S_priori = _normalize_upper_factor(R_qr[:n, :n])

        # 零号 sigma 点修正
        w0_dev = chi_prop[0] - x_priori
        Wc0 = self._Wc0
        if abs(Wc0) < 1e-15:
            pass  # Wc0 ≈ 0，无需修正
        elif Wc0 > 0:
            S_priori = _chol_update(S_priori, np.sqrt(Wc0) * w0_dev, '+')
        else:  # Wc0 < 0
            P_tmp = S_priori.T @ S_priori + Wc0 * np.outer(w0_dev, w0_dev)
            P_tmp = _ensure_symmetric(P_tmp)
            try:
                S_priori = cholesky(P_tmp, lower=False)
            except Exception:
                eigvals = np.linalg.eigvalsh(P_tmp)
                P_tmp += np.eye(n) * max(0.0, -eigvals.min() + 1e-14)
                S_priori = cholesky(P_tmp, lower=False)

        self._S = S_priori
        self._x = x_priori
        P_priori = _ensure_symmetric(S_priori.T @ S_priori)
        self._P = P_priori
        return x_priori, P_priori

    def _build_f_discrete(self, A_SDC: np.ndarray, B: np.ndarray,
                          du: np.ndarray, dt: float, ctx: dict):
        """从 ctx 提取离散映射，缺失则报错。"""
        f_disc = ctx.get('f_discrete', None)
        if f_disc is not None:
            return f_disc

        A_SDC_fn = ctx.get('A_SDC_fn', None)
        if A_SDC_fn is not None:
            def _g(xx):
                A = A_SDC_fn(xx)
                return xx + dt * (A @ xx + B @ du)
            return _g

        raise ValueError(
            "SquareRootUKFNavigator.predict() 需要 ctx['f_discrete'] 或 "
            "ctx['A_SDC_fn']，否则无法传播 sigma 点。"
        )

    # ── UKF 测量更新 ────────────────────────────────────────────────────

    def update(self, x_priori: np.ndarray, P_priori: np.ndarray,
               z_meas: np.ndarray) -> np.ndarray:
        """完整的 UKF 测量更新（平方根形式）。

        1. 从先验 S_priori 生成 sigma 点
        2. 经测量函数 h(x) 传播
        3. 圆周角均值 + 测量协方差 QR
        4. 计算互协方差 + Kalman 增益
        5. 平方根状态/协方差更新
        """
        n = self._n
        S_priori = self._S  # predict 后已更新
        m = len(z_meas)
        sqrt_R = cholesky(self.R, lower=False)  # 测量噪声平方根

        # ── 测量 sigma 点 ──────────────────────────────────────────────
        chi_prior = self._sigma_points(x_priori, S_priori)
        zeta = np.zeros((2 * n + 1, m))
        for i in range(2 * n + 1):
            zeta[i] = self._measure_from_rel_state(chi_prior[i])

        # ── 测量均值（角度用圆周均值）─────────────────────────────────
        z_pred = np.zeros(m)
        if self.angles_only:
            # 仅测角：[az, el]，均需圆周均值
            az_samples = zeta[:, 0]
            el_samples = zeta[:, 1]
            z_pred[0] = _circular_mean(az_samples)
            z_pred[1] = np.average(el_samples, weights=None)  # el ∈ [-π/2, π/2]，普通均值
            # 用 Wm 权重重算加权版
            z_pred[0] = float(np.arctan2(
                self._Wm0 * np.sin(zeta[0, 0]) + self._Wi * np.sum(np.sin(zeta[1:, 0])),
                self._Wm0 * np.cos(zeta[0, 0]) + self._Wi * np.sum(np.cos(zeta[1:, 0]))))
            z_pred[1] = (self._Wm0 * zeta[0, 1]
                         + self._Wi * np.sum(zeta[1:, 1]))
        else:
            # [ρ, az, el]: az 圆周，ρ 和 el 线性
            z_pred[0] = (self._Wm0 * zeta[0, 0]
                         + self._Wi * np.sum(zeta[1:, 0]))
            z_pred[1] = float(np.arctan2(
                self._Wm0 * np.sin(zeta[0, 1]) + self._Wi * np.sum(np.sin(zeta[1:, 1])),
                self._Wm0 * np.cos(zeta[0, 1]) + self._Wi * np.sum(np.cos(zeta[1:, 1]))))
            z_pred[2] = (self._Wm0 * zeta[0, 2]
                         + self._Wi * np.sum(zeta[1:, 2]))

        # ── 角度 wrap 样本（使残差在 [-π, π]）────────────────────────
        ia, ie = self._angle_indices
        zeta_wrapped = zeta.copy()
        for s in range(2 * n + 1):
            zeta_wrapped[s, ia] = _wrap_angle_scalar(zeta[s, ia] - z_pred[ia]) + z_pred[ia]
            if not self.angles_only:
                zeta_wrapped[s, ie] = _wrap_angle_scalar(zeta[s, ie] - z_pred[ie]) + z_pred[ie]

        # ── 测量协方差平方根（QR 分解）─────────────────────────────────
        # 非中心 sigma 点偏差 + 噪声 QR（与 predict 一致，中心点单独 cholupdate）
        dev_z = np.zeros((2 * n + m, m))
        sqrt_Wi = np.sqrt(self._Wi)
        for i in range(1, 2 * n + 1):
            dev_z[i - 1] = sqrt_Wi * (zeta_wrapped[i] - z_pred)
        for i in range(m):
            dev_z[2 * n + i] = sqrt_R[i, :]

        _, Rz_qr = scipy_qr(dev_z, mode='economic')
        S_y = _normalize_upper_factor(Rz_qr[:m, :m])

        # 零号 sigma 点修正（Wc0 可正可负）
        w0z_dev = zeta_wrapped[0] - z_pred
        if self._Wc0 > 1e-15:
            S_y = _chol_update(S_y, np.sqrt(self._Wc0) * w0z_dev, '+')
        elif self._Wc0 < -1e-15:
            S_y = _chol_update(S_y, np.sqrt(-self._Wc0) * w0z_dev, '-')

        # 缓存新息协方差 (P_yy = S_y' @ S_y) 供 NIS 计算
        P_yy = _ensure_symmetric(S_y.T @ S_y)
        self._last_S = P_yy

        # ── 互协方差 P_xy ──────────────────────────────────────────────
        P_xy = np.zeros((n, m))
        P_xy += self._Wc0 * np.outer(chi_prior[0] - x_priori, zeta_wrapped[0] - z_pred)
        for i in range(1, 2 * n + 1):
            P_xy += self._Wi * np.outer(chi_prior[i] - x_priori, zeta_wrapped[i] - z_pred)

        # ── Kalman 增益 + 状态更新 ─────────────────────────────────────
        # S_y 上三角：S_y'@S_y = P_yy, 解 S_y'@(S_y@K') = P_xy'
        # K = (P_xy / S_y) / S_y'  = P_xy @ inv(P_yy)
        # 用两次三角求解：
        #   step 1: K_tmp @ S_y = P_xy  →  K_tmp = P_xy @ inv(S_y)
        K_tmp = np.linalg.solve(S_y.T, P_xy.T).T  # (n,m)
        #   step 2: K @ S_y' = K_tmp  →  K = K_tmp @ inv(S_y')
        K = np.linalg.solve(S_y, K_tmp.T).T  # (n,m)

        y_innov = z_meas - z_pred
        y_innov[ia] = RelativeStateEKF.wrap_angle(np.atleast_1d(y_innov[ia]))[0]
        if not self.angles_only:
            y_innov[ie] = RelativeStateEKF.wrap_angle(np.atleast_1d(y_innov[ie]))[0]

        self._x = x_priori + K @ y_innov

        # ── 平方根协方差更新 ───────────────────────────────────────────
        # P_post = P_priori - K @ P_yy @ K.T
        #        = S'@S - K @ S_y'@S_y @ K'
        #        = S'@S - (K @ S_y') @ (K @ S_y')'
        # → U = K @ S_y.T 使 U @ U.T = K @ P_yy @ K.T
        U = K @ S_y.T  # (n, m)
        S_post = S_priori.copy()
        for j in range(m):
            if np.linalg.norm(U[:, j]) < 1e-15:
                continue
            S_post = _chol_update(S_post, U[:, j], '-')

        self._S = S_post
        self._P = _ensure_symmetric(S_post.T @ S_post)
        return y_innov


# ═══════════════════════════════════════════════════════════════════════════════
# 工厂函数
# ═══════════════════════════════════════════════════════════════════════════════

def wrap_legacy_ekf(ekf: RelativeStateEKF) -> SDCEKFNavigator:
    """将旧版 RelativeStateEKF 包装为 SDCEKFNavigator（保持现有状态）。"""
    nav = SDCEKFNavigator(
        x0=ekf.x, P0=ekf.P, Q=ekf.Q, R=ekf.R,
        angles_only=ekf.angles_only,
    )
    return nav


def create_navigator(nav_type: str, x0: np.ndarray, P0: np.ndarray,
                     Q: np.ndarray, R: np.ndarray, angles_only: bool = False,
                     **kwargs) -> BaseNavigator:
    """创建导航器的工厂函数。

    Parameters
    ----------
    nav_type : str  'sdc', 'jacobian', 或 'ukf'
    kwargs : eps, state_scales (jacobian); alpha, beta, kappa (ukf)
    """
    nav_type = nav_type.lower()
    if nav_type == 'sdc':
        return SDCEKFNavigator(x0, P0, Q, R, angles_only)
    elif nav_type == 'jacobian':
        eps = kwargs.pop('eps', 1e-4)
        scales = kwargs.pop('state_scales', None)
        return JacobianEKFNavigator(x0, P0, Q, R, angles_only,
                                     eps=eps, state_scales=scales)
    elif nav_type == 'ukf':
        alpha = kwargs.pop('alpha', 1.0)
        beta = kwargs.pop('beta', 2.0)
        kappa = kwargs.pop('kappa', 0.0)
        return SquareRootUKFNavigator(x0, P0, Q, R, angles_only,
                                       alpha=alpha, beta=beta, kappa=kappa)
    else:
        raise ValueError(f"未知导航器类型: {nav_type}，可选: sdc, jacobian, ukf")
