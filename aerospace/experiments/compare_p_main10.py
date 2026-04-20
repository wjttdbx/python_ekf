"""
P 矩阵精度对比实验：main10 参数版本（Q=0.001, R=1e7）

对比方法：
  1. scipy (基准)
  2. CARE Schur (scipy 内部实现)
  3. JT/Taylor (main10_nonlinear.cpp 算法)

运行方法：
    uv run python -m aerospace.experiments.compare_p_main10
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy.linalg import solve_continuous_are, LinAlgError

from aerospace.pinn.data_generator import _solve_are_balanced, STATE_DIM, CTRL_DIM

# ─── main10_nonlinear.cpp 的物理参数 ──────────────────────────────────────
MU   = 3.986e5
A_C  = 15000.0
E_C  = 0.5
GAMMA = np.sqrt(2.0)

Q = np.eye(STATE_DIM) * 0.001    # main10 参数
R = np.eye(CTRL_DIM) * 1e7       # main10 参数
B_P = np.zeros((STATE_DIM, CTRL_DIM))
B_P[3, 0] = B_P[4, 1] = B_P[5, 2] = 1.0
B_E = -B_P

GAM2 = GAMMA ** 2
R_INV = np.linalg.inv(R)
S_MAT = (1.0 - 1.0 / GAM2) * (B_P @ R_INV @ B_P.T)

def care_residual(A: np.ndarray, P: np.ndarray) -> float:
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
        if not np.all(np.isfinite(A)):
            continue
        samples.append(A)
    return samples

# ─── Method 1: scipy 基准 ────────────────────────────────────────────────────
def solve_scipy(A: np.ndarray) -> np.ndarray:
    return _solve_are_balanced(A, B_P, Q, R / (1.0 - 1.0/GAM2))

# ─── Method 2: CARE Schur（与 scipy 相同）────────────────────────────────────
def solve_schur(A: np.ndarray) -> np.ndarray:
    return solve_scipy(A)

# ─── Method 3: JT (Taylor + Sylvester) ───────────────────────────────────────
# main10_nonlinear.cpp 的 A0 和 P0
A0 = np.array([
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
    [0.000002037495611, 0.000000620251166, 0.000000620251166, 0, 0.002380961523792, 0],
    [-0.000000256493561, 0.000001160750883, -0.000000256493561, -0.002380961523792, 0, 0],
    [-0.000000256493561, -0.000000256493561, -0.000000256493561, 0, 0, 0],
], dtype=np.float64)

P0_raw = np.array([
    [0.000066138653469, -0.000000603895285, 0.000000850246221, 0.015493714796988, 0.008800019456648, -0.000401070213473],
    [-0.000000603895285, 0.000059845562026, -0.000002031310408, -0.007635180710369, 0.013650378954801, -0.000777872832157],
    [0.000000850246221, -0.000002031310408, 0.000052560224005, 0.001261589374423, -0.000110291586287, 0.013696254562016],
    [0.015493714796988, -0.007635180710369, 0.001261589374423, 7.734674054598234, 0.218687103819545, 0.138960742387561],
    [0.008800019456648, 0.013650378954801, -0.000110291586287, 0.218687103819545, 7.525340450808993, -0.076714985631586],
    [-0.000401070213473, -0.000777872832157, 0.013696254562016, 0.138960742387561, -0.076714985631586, 7.400001552102313],
], dtype=np.float64)
P0 = P0_raw * 1e4

C0 = B_P @ R_INV @ B_P.T - (1.0/GAM2) * B_E @ R_INV @ B_E.T
C_sylv = A0.T - P0 @ C0
D_sylv = A0 - C0 @ P0

def _sylv_solve(A_mat: np.ndarray, B_mat: np.ndarray, C_mat: np.ndarray) -> np.ndarray:
    n = A_mat.shape[0]
    K = np.kron(np.eye(n), A_mat) + np.kron(B_mat.T, np.eye(n))
    rhs = C_mat.flatten(order="F")
    x = np.linalg.solve(K, rhs)
    return x.reshape(n, n, order="F")

def solve_jt(A: np.ndarray, max_iter: int = 15, tol: float = 1e-8) -> np.ndarray:
    delta_A = A - A0
    P_store = [P0]
    Pk = P0
    P_sum = P0.copy()
    for num in range(1, max_iter + 1):
        Cn = np.zeros((STATE_DIM, STATE_DIM))
        for i in range(num):
            j = num - i
            if j < len(P_store):
                Cn += P_store[i] @ C0 @ P_store[j]
        Pn = Cn - Pk @ delta_A - delta_A.T @ Pk
        Pk = _sylv_solve(C_sylv, D_sylv, Pn)
        P_store.append(Pk)
        P_sum += Pk
        if np.linalg.norm(Pk, "fro") < tol:
            break
    return P_sum

# ─── 运行实验 ─────────────────────────────────────────────────────────────────
def run(n_samples: int = 1000, seed: int = 42):
    print("=" * 70)
    print(f"  P 矩阵精度对比实验 (main10 参数: Q=0.001, R=1e7)")
    print(f"  样本数: {n_samples}")
    print("=" * 70)

    print("\n  生成样本中...", end="", flush=True)
    samples = gen_samples(n_samples, seed)
    print(f" {len(samples)} 个\n")

    methods = {
        "scipy (基准)":       solve_scipy,
        "CARE Schur":         solve_schur,
        "JT/Taylor(main10)": solve_jt,
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
    print("  最小特征值（正定性）  均值 / 最小值")
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
    print("  各方法 CARE 残差 / scipy基准残差  （几何均值）")
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
