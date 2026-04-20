"""
P 矩阵精度对比实验：main10(JT)、PINN、CARE Schur、CARE Eigendecomp

指标：CARE 残差 ‖Aᵀ P + P A - P·S·P + Q‖_F（越小越精确）
基准：scipy.linalg.solve_continuous_are（视为真值）

运行方法：
    uv run python -m aerospace.experiments.compare_p_accuracy
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import eig, schur, solve_continuous_are, LinAlgError

from aerospace.pinn.data_generator import (
    STATE_DIM, CTRL_DIM, FEAT_DIM,
    extract_asdc_features, _solve_are_balanced,
)
from aerospace.pinn.checkpoint_utils import normalize_pinn_state_dict
from aerospace.pinn.pinn_trainer import SDREPINN, reconstruct_spd_p
from aerospace.paths import CHECKPOINTS_DIR

# ─── 共享物理参数 ──────────────────────────────────────────────────────────
MU   = 3.986e5      # km³/s²
A_C  = 15000.0      # km
E_C  = 0.5
GAMMA = np.sqrt(2.0)

# 注意：使用 PINN 训练时的参数（Q=1.0, R=1e13），而非 main10 的参数（Q=0.001, R=1e7）
# 原因：PINN 是在特定 Q/R 下训练的，不能用于其他参数组合
Q = np.eye(STATE_DIM) * 1.0      # PINN 训练参数
R = np.eye(CTRL_DIM) * 1e13      # PINN 训练参数
B_P = np.zeros((STATE_DIM, CTRL_DIM))
B_P[3, 0] = B_P[4, 1] = B_P[5, 2] = 1.0
B_E = -B_P

# 博弈 CARE 的有效 S：S = (1 - 1/γ²) · Bp R⁻¹ Bpᵀ
GAM2 = GAMMA ** 2
R_INV = np.linalg.inv(R)
S_MAT = (1.0 - 1.0 / GAM2) * (B_P @ R_INV @ B_P.T)

# 辛平衡系数（固定）
_alpha_scale = float(np.sqrt(np.linalg.norm(Q, "fro") / np.linalg.norm(S_MAT, "fro")))

# ─── CARE 残差 ────────────────────────────────────────────────────────────────
def care_residual(A: np.ndarray, P: np.ndarray) -> float:
    """‖Aᵀ P + P A - P S P + Q‖_F"""
    R_ = A.T @ P + P @ A - P @ S_MAT @ P + Q
    return float(np.linalg.norm(R_, "fro"))


# ─── 样本生成 ─────────────────────────────────────────────────────────────────
def build_asdc(x_p: np.ndarray, x_e: np.ndarray,
               rc: float, nd: float, ndd: float) -> np.ndarray:
    A = np.zeros((6, 6))
    A[0, 3] = A[1, 4] = A[2, 5] = 1.0
    rp = np.sqrt((rc + x_p[0])**2 + x_p[1]**2 + x_p[2]**2)
    re = np.sqrt((rc + x_e[0])**2 + x_e[1]**2 + x_e[2]**2)
    dx, dy, dz = x_p[0]-x_e[0], x_p[1]-x_e[1], x_p[2]-x_e[2]
    r2 = dx*dx + dy*dy + dz*dz + 1e-6
    bx = -MU*(rc+x_p[0])/rp**3 + MU*(rc+x_e[0])/re**3
    by = -MU*x_p[1]/rp**3 + MU*x_e[1]/re**3
    bz = -MU*x_p[2]/rp**3 + MU*x_e[2]/re**3
    A[3, 0] = nd**2 + bx*dx/r2;  A[3, 1] = ndd + bx*dy/r2;  A[3, 2] = bx*dz/r2
    A[4, 0] = -ndd + by*dx/r2;   A[4, 1] = nd**2 + by*dy/r2; A[4, 2] = by*dz/r2
    A[5, 0] = bz*dx/r2;          A[5, 1] = bz*dy/r2;         A[5, 2] = bz*dz/r2
    A[3, 4] = 2*nd;  A[4, 3] = -2*nd
    return A


def gen_samples(n: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    pc = A_C * (1 - E_C**2)
    samples = []
    while len(samples) < n:
        x_p = np.concatenate([rng.uniform(-1000, 1000, 3), np.zeros(3)])
        x_e = np.concatenate([rng.uniform(-1000, 1000, 3), np.zeros(3)])
        nu = rng.uniform(0, 2*np.pi)
        rc = pc / (1 + E_C * np.cos(nu))
        h = np.sqrt(MU * pc)
        nd = h / rc**2
        ndd = -2*MU*E_C*np.sin(nu) / rc**3 * (h / (MU*(1+E_C*np.cos(nu))))
        A = build_asdc(x_p, x_e, rc, nd, ndd)
        # 过滤数值不稳定的样本
        if not np.all(np.isfinite(A)):
            continue
        samples.append(A)
    return samples


# ─── Method 1: scipy 基准（真值）────────────────────────────────────────────
def solve_scipy(A: np.ndarray) -> np.ndarray:
    """辛平衡 + scipy solve_continuous_are（视为真值）"""
    return _solve_are_balanced(A, B_P, Q, R / (1.0 - 1.0/GAM2))


# ─── Method 2: CARE Eigendecomp（Hamiltonian 特征值分解 + 辛平衡）──────────
def solve_eigen(A: np.ndarray) -> np.ndarray:
    """2n×2n Hamiltonian 特征分解，使用与 scipy 一致的辛平衡策略。

    关键修复：
    1. 使用 R_eff 构造 S（与 scipy 一致）
    2. 特征向量排序（按特征值实部升序）
    3. 使用 lstsq 代替 inv 提高数值稳定性
    """
    n = A.shape[0]
    R_eff = R / (1.0 - 1.0/GAM2)
    S = B_P @ np.linalg.inv(R_eff) @ B_P.T
    norm_Q = np.linalg.norm(Q, "fro")
    norm_S = np.linalg.norm(S, "fro")
    alpha = np.sqrt(norm_Q / norm_S)
    Q_b = Q / alpha
    S_b = S / alpha

    H = np.block([[A, -S_b], [-Q_b, -A.T]])
    vals, vecs = eig(H)

    # 按特征值实部排序，选取最负的 n 个
    idx_sorted = np.argsort(vals.real)
    idx = idx_sorted[:n]

    if vals[idx[-1]].real >= 0:
        raise ValueError(f"Not enough stable eigenvalues: max Re(λ)={vals[idx[-1]].real}")

    U1 = vecs[:n, idx]
    U2 = vecs[n:, idx]

    # 使用 lstsq 代替 inv，提高数值稳定性
    Pc = np.linalg.lstsq(U1.T, U2.T, rcond=None)[0].T
    Pb = 0.5 * (Pc.real + Pc.real.T)
    return alpha * Pb


# ─── Method 3: CARE Schur（scipy 内部实现，作为对照）────────────────────────
def solve_schur(A: np.ndarray) -> np.ndarray:
    """scipy 的 solve_continuous_are 内部就是用 Schur 方法。

    这里作为独立方法列出，实际上与 scipy 基准相同，用于验证一致性。
    """
    return solve_scipy(A)


# ─── Method 4: JT (Taylor + Sylvester) ─────────────────────────────────────
# 注意：A0 和 P0 是针对 Q=0.001, R=1e7 预计算的
# 当前实验使用 Q=1.0, R=1e13，需要重新计算 P0
# 为简化，这里跳过 JT 方法（或者需要用 scipy 重新计算 A0 对应的 P0）

def solve_jt(A: np.ndarray, max_iter: int = 15, tol: float = 1e-8) -> np.ndarray:
    """JT 方法需要针对当前 Q/R 重新计算 P0，这里暂时返回 NaN"""
    raise NotImplementedError("JT 方法的 P0 是针对 Q=0.001, R=1e7 预计算的，与当前参数不匹配")


# ─── Method 5: PINN ──────────────────────────────────────────────────────────
class PINNSolver:
    def __init__(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.device = torch.device("cpu")
        self.model = SDREPINN(
            in_dim=int(ckpt["in_dim"]),
            backbone_dim=int(ckpt.get("backbone_dim", 256)),
            n_resblocks=int(ckpt.get("n_resblocks", 3)),
            head_hidden=int(ckpt.get("head_hidden", 64)),
            activation=str(ckpt.get("activation", "mish")),
        ).to(self.device)
        sd = normalize_pinn_state_dict(ckpt["model_state_dict"])
        self.model.load_state_dict(sd, strict=True)
        self.model.eval()
        self.delta_spd = float(ckpt["delta_spd"])
        self.feat_mean = np.asarray(ckpt["feat_mean"], dtype=np.float64)
        self.feat_std = np.asarray(ckpt["feat_std"], dtype=np.float64)
        self.feat_std = np.where(self.feat_std < 1e-12, 1.0, self.feat_std)
        self.l_mean = np.asarray(ckpt["l_mean"], dtype=np.float64)
        self.l_std = np.asarray(ckpt["l_std"], dtype=np.float64)
        self.l_std = np.where(self.l_std < 1e-12, 1.0, self.l_std)
        self._tril_r, self._tril_c = np.tril_indices(STATE_DIM)

    def solve(self, A: np.ndarray) -> np.ndarray:
        feat = extract_asdc_features(A)
        feat_norm = (feat - self.feat_mean) / self.feat_std
        x = torch.from_numpy(feat_norm.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            l_raw = self.model(x).squeeze(0).numpy().astype(np.float64)
        l_vec = l_raw * self.l_std + self.l_mean
        L = np.zeros((STATE_DIM, STATE_DIM))
        diag_mask = self._tril_r == self._tril_c
        for k, (r, c) in enumerate(zip(self._tril_r, self._tril_c)):
            L[r, c] = np.exp(l_vec[k]) if diag_mask[k] else l_vec[k]
        P = L @ L.T + np.eye(STATE_DIM) * self.delta_spd
        return P


# ─── 运行实验 ─────────────────────────────────────────────────────────────────
def run(n_samples: int = 1000, seed: int = 42):
    print("=" * 70)
    print(f"  P 矩阵精度对比实验  (N={n_samples} 样本)")
    print("  基准: scipy.linalg.solve_continuous_are（辛平衡预处理）")
    print("=" * 70)

    ckpt_path = str(CHECKPOINTS_DIR / "sdre_pinn" / "best_model.pt")
    pinn = PINNSolver(ckpt_path)
    print(f"\n[√] PINN 权重加载: {ckpt_path}\n")

    print("  生成样本中...", end="", flush=True)
    samples = gen_samples(n_samples, seed)
    print(f" {len(samples)} 个\n")

    methods = {
        "scipy (基准)":        solve_scipy,
        "CARE Eigendecomp":    solve_eigen,
        "CARE Schur":          solve_schur,
        "PINN":                pinn.solve,
    }

    results: dict[str, dict] = {name: {"res": [], "sym": [], "pd": [], "t_ms": []}
                                  for name in methods}

    for si, A in enumerate(samples):
        if (si + 1) % 100 == 0:
            print(f"  进度: {si+1}/{n_samples}", flush=True)
        for name, fn in methods.items():
            t0 = time.perf_counter()
            try:
                P = fn(A)
                dt = (time.perf_counter() - t0) * 1e3
                res = care_residual(A, P)
                sym = float(np.linalg.norm(P - P.T, "fro"))
                try:
                    eigs = np.linalg.eigvalsh(P)
                    pd = float(eigs.min())
                except Exception:
                    pd = float("nan")
                results[name]["res"].append(res)
                results[name]["sym"].append(sym)
                results[name]["pd"].append(pd)
                results[name]["t_ms"].append(dt)
            except Exception as e:
                results[name]["res"].append(float("nan"))
                results[name]["sym"].append(float("nan"))
                results[name]["pd"].append(float("nan"))
                results[name]["t_ms"].append(float("nan"))

    # ── 打印结果 ──
    print("\n" + "=" * 70)
    print("  CARE 残差 ‖Aᵀ P + P A - P·S·P + Q‖_F")
    print("-" * 70)
    print(f"  {'方法':<22}  {'均值':>12}  {'中位数':>12}  {'p99':>12}  {'失败数':>6}")
    print("-" * 70)

    for name in methods:
        v = np.array(results[name]["res"])
        valid = v[np.isfinite(v)]
        fails = int(np.sum(~np.isfinite(v)))
        if len(valid) == 0:
            print(f"  {name:<22}  {'ALL FAILED':>12}")
            continue
        p99 = float(np.percentile(valid, 99))
        print(f"  {name:<22}  {np.mean(valid):>12.3e}  {np.median(valid):>12.3e}"
              f"  {p99:>12.3e}  {fails:>6}")

    print("\n" + "-" * 70)
    print("  对称性误差 ‖P - Pᵀ‖_F  (均值)")
    print("-" * 70)
    for name in methods:
        v = np.array(results[name]["sym"])
        valid = v[np.isfinite(v)]
        if len(valid):
            print(f"  {name:<22}  {np.mean(valid):>12.3e}")

    print("\n" + "-" * 70)
    print("  最小特征值（正定性指标）  均值 / 最小值")
    print("-" * 70)
    for name in methods:
        v = np.array(results[name]["pd"])
        valid = v[np.isfinite(v)]
        if len(valid):
            print(f"  {name:<22}  均值={np.mean(valid):+.3e}  最小={np.min(valid):+.3e}")

    print("\n" + "-" * 70)
    print("  单次求解耗时 (ms)  均值 / 中位数")
    print("-" * 70)
    for name in methods:
        v = np.array(results[name]["t_ms"])
        valid = v[np.isfinite(v)]
        if len(valid):
            print(f"  {name:<22}  均值={np.mean(valid):7.2f}ms  中位数={np.median(valid):7.2f}ms")

    # ── 与 scipy 基准的相对误差 ──
    print("\n" + "-" * 70)
    print("  各方法 CARE 残差 / scipy基准残差  （几何均值，越接近1越好）")
    print("-" * 70)
    base = np.array(results["scipy (基准)"]["res"])
    for name in methods:
        if name == "scipy (基准)":
            continue
        v = np.array(results[name]["res"])
        mask = np.isfinite(v) & np.isfinite(base) & (base > 0)
        if mask.sum() == 0:
            continue
        ratio = v[mask] / base[mask]
        gm = float(np.exp(np.mean(np.log(ratio[ratio > 0]))))
        print(f"  {name:<22}  几何均值比={gm:.4f}x")

    print("\n" + "=" * 70)
    print("  实验完成")
    print("=" * 70)


if __name__ == "__main__":
    run(n_samples=1000)
