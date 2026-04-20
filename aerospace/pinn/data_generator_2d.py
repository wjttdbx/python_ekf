"""
2D SDRE-PINN 数据生成与预处理

输入特征 (5D)：从 A_SDC 提取 5 个独立非平凡元素 [a20, a21, a23, a30, a31]
输出标签 (10D)：Log-Cholesky 参数（对角取 log，非对角直接输出）
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.linalg import LinAlgError, solve_continuous_are
from scipy.stats import qmc
from torch.utils.data import DataLoader, Dataset

from aerospace.dynamics.nerm_2d import OrbitalDynamics2D

STATE_DIM = 4
CTRL_DIM = 2
FEAT_DIM = 5   # [A[2,0], A[2,1], A[2,3], A[3,0], A[3,1]]
L_DIM = 10     # tril(4×4)

# tril_indices 中对角元素的位置: (0,0)→0, (1,1)→2, (2,2)→5, (3,3)→9
_TRIL_R, _TRIL_C = np.tril_indices(STATE_DIM)
_DIAG_MASK = (_TRIL_R == _TRIL_C)  # [T, F, T, F, F, T, F, F, F, T]


def extract_asdc_features(a_sdc: np.ndarray) -> np.ndarray:
    """从 4×4 A_SDC 提取 5 个独立非平凡元素。"""
    return np.array([
        a_sdc[2, 0], a_sdc[2, 1], a_sdc[2, 3],
        a_sdc[3, 0], a_sdc[3, 1],
    ], dtype=np.float64)


def _solve_are_balanced(A, B, Q, R_eff):
    S = B @ np.linalg.inv(R_eff) @ B.T
    norm_Q = np.linalg.norm(Q, "fro")
    norm_S = np.linalg.norm(S, "fro")
    if norm_S < 1e-30:
        raise ValueError("S ≈ 0")
    alpha = np.sqrt(norm_Q / norm_S)
    Q_b, R_b = Q / alpha, R_eff / alpha
    try:
        return alpha * solve_continuous_are(A, B, Q_b, R_b)
    except (LinAlgError, ValueError):
        pass
    n = A.shape[0]
    d = np.array([1.0 / max(np.max(np.abs(A[i, :])), 1e-15) for i in range(n)])
    T = np.diag(d)
    Ti = np.diag(1.0 / d)
    return alpha * (Ti.T @ solve_continuous_are(Ti @ A @ T, Ti @ B, T.T @ Q_b @ T, R_b) @ Ti)


def _p_to_log_cholesky_vec(p: np.ndarray) -> np.ndarray:
    """P → Log-Cholesky 10-vec（对角取 log，非对角直接输出）。"""
    l_mat = np.linalg.cholesky(p)
    vec = l_mat[_TRIL_R, _TRIL_C].copy()
    vec[_DIAG_MASK] = np.log(vec[_DIAG_MASK])
    return vec


@dataclass
class SamplingBounds2D:
    pos_min_km: float = -1000.0
    pos_max_km: float = 1000.0
    ac_min_km: float = 7000.0
    ac_max_km: float = 42000.0
    ec_min: float = 0.0
    ec_max: float = 0.7


class PINNDataset2D(Dataset):
    def __init__(self, feat_norm, l_norm, a_sdc, p_true=None, x_rel=None):
        self.feat_norm = torch.from_numpy(feat_norm.astype(np.float32))
        self.l_norm = torch.from_numpy(l_norm.astype(np.float32))
        self.a_sdc = torch.from_numpy(a_sdc.astype(np.float32))
        n = self.feat_norm.shape[0]
        self.p_true = (
            torch.from_numpy(p_true.astype(np.float32))
            if p_true is not None
            else torch.zeros(n, STATE_DIM, STATE_DIM)
        )
        self.x_rel = (
            torch.from_numpy(x_rel.astype(np.float32))
            if x_rel is not None
            else torch.zeros(n, STATE_DIM)
        )

    def __len__(self):
        return self.feat_norm.shape[0]

    def __getitem__(self, idx):
        return (self.feat_norm[idx], self.l_norm[idx], self.a_sdc[idx],
                self.p_true[idx], self.x_rel[idx])


def _is_spd(mat, eps=1e-10):
    if not np.allclose(mat, mat.T, atol=1e-7):
        return False
    return bool(np.all(np.linalg.eigvalsh(mat) > eps))


def _lhs_positions_2d(n_samples, bounds, seed):
    """2D LHS 采样绝对位置 [x, y]。"""
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    u = sampler.random(n=n_samples)
    low = np.array([bounds.pos_min_km, bounds.pos_min_km])
    high = np.array([bounds.pos_max_km, bounds.pos_max_km])
    return qmc.scale(u, low, high)


def _lhs_orbital_params(n_samples: int, bounds: SamplingBounds2D, seed: int) -> np.ndarray:
    """LHS 采样半长轴和离心率 [a_c, e_c]。"""
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    u = sampler.random(n=n_samples)
    low = np.array([bounds.ac_min_km, bounds.ec_min], dtype=float)
    high = np.array([bounds.ac_max_km, bounds.ec_max], dtype=float)
    return qmc.scale(u, low, high)


def _lhs_true_anomaly(n_samples, seed):
    sampler = qmc.LatinHypercube(d=1, seed=seed + 97)
    return 2.0 * np.pi * sampler.random(n=n_samples).reshape(-1)


def _run_rollouts_2d(
    bounds: SamplingBounds2D,
    b_p: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    r_eff: np.ndarray,
    gamma: float,
    n_rollouts: int = 500,
    steps_per_rollout: int = 300,
    dt: float = 100.0,
    seed: int = 7777,
) -> tuple[list, list, list, list, int]:
    """运行闭环轨迹回放，收集真实流形上的训练样本。"""
    rng = np.random.default_rng(seed)
    r_inv = np.linalg.inv(r)
    b_e = -b_p

    feat_list: list[np.ndarray] = []
    p_list: list[np.ndarray] = []
    a_sdc_list: list[np.ndarray] = []
    xrel_list: list[np.ndarray] = []
    fail_count = 0

    for traj_i in range(n_rollouts):
        pos_p = rng.uniform(-500, 500, size=2)
        pos_e = rng.uniform(-500, 500, size=2)
        vel_p = rng.uniform(-0.05, 0.05, size=2)
        vel_e = rng.uniform(-0.05, 0.05, size=2)
        nu0 = rng.uniform(0, 2 * np.pi)

        # 针对当前轨迹随机选取参考轨道
        a_c = rng.uniform(bounds.ac_min_km, bounds.ac_max_km)
        e_c = rng.uniform(bounds.ec_min, bounds.ec_max)
        dynamics = OrbitalDynamics2D(mu=3.986e5, a_c=a_c, e_c=e_c)

        state = np.zeros(9)
        state[0:2], state[2:4] = pos_p, vel_p
        state[4:6], state[6:8] = pos_e, vel_e
        state[8] = nu0

        for _ in range(steps_per_rollout):
            X_p, X_e = state[0:4], state[4:8]
            nu_val = state[8]
            x_rel = X_p - X_e

            r_c, nu_dot, nu_ddot = dynamics.get_orbital_params(nu_val)
            a_sdc = dynamics.get_SDC_matrix(X_p, X_e, r_c, nu_dot, nu_ddot)

            try:
                p = _solve_are_balanced(a_sdc, b_p, q, r_eff)
            except (LinAlgError, ValueError):
                fail_count += 1
                break
            if not _is_spd(p):
                fail_count += 1
                break

            feat_list.append(extract_asdc_features(a_sdc))
            p_list.append(p)
            a_sdc_list.append(a_sdc)
            xrel_list.append(x_rel.copy())

            u_p = -r_inv @ b_p.T @ p @ x_rel
            u_e = gamma ** (-2) * r_inv @ b_e.T @ p @ x_rel

            try:
                sol = solve_ivp(
                    lambda _t, y: dynamics.dynamics_9d(_t, y, u_p, u_e),
                    (0, dt), state, method="RK45", rtol=1e-8, atol=1e-10,
                )
                if not sol.success:
                    break
                state = sol.y[:, -1].copy()
            except Exception:
                break

        if (traj_i + 1) % 10 == 0:
            print(f"  [Rollout] {traj_i + 1}/{n_rollouts} done, {len(feat_list)} samples")

    return feat_list, p_list, a_sdc_list, xrel_list, fail_count


def generate_dataset(
    n_samples: int = 500_000,
    output_path: str = "data/sdre_pinn_2d_dataset.npz",
    val_ratio: float = 0.2,
    seed: int = 42,
    bounds: SamplingBounds2D | None = None,
    q: np.ndarray | None = None,
    r: np.ndarray | None = None,
    gamma: float = np.sqrt(2.0),
    n_rollouts: int = 500,
    steps_per_rollout: int = 300,
    rollout_dt: float = 100.0,
) -> dict:
    """生成 2D SDRE-PINN 数据集（LHS 随机采样 + 闭环轨迹回放混合）。

    通过 LHS 采样位置和真近点角生成多样的 A_SDC 矩阵，
    然后从 A_SDC 中提取 5 个独立参数作为网络输入特征。
    标签使用 Log-Cholesky 参数化（对角取 log）以压缩动态范围。
    闭环轨迹回放补充真实流形上的样本，提升闭环控制精度。
    """
    if bounds is None:
        bounds = SamplingBounds2D()
    if q is None:
        q = np.eye(STATE_DIM)
    if r is None:
        r = np.eye(CTRL_DIM) * 1e13

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    b_p = np.zeros((STATE_DIM, CTRL_DIM))
    b_p[2:, :] = np.eye(CTRL_DIM)
    r_eff = r / (1.0 - gamma ** (-2))

    pos_p_samples = _lhs_positions_2d(n_samples, bounds, seed=seed)
    pos_e_samples = _lhs_positions_2d(n_samples, bounds, seed=seed + 13)
    nu_samples = _lhs_true_anomaly(n_samples, seed=seed)
    orb_samples = _lhs_orbital_params(n_samples, bounds, seed=seed + 27)

    feat_raw = []
    p_true = []
    a_sdc_all = []
    x_rel_all = []

    fail_count = 0
    for i in range(n_samples):
        pos_p = pos_p_samples[i]
        pos_e = pos_e_samples[i]
        nu = float(nu_samples[i])
        a_c, e_c = orb_samples[i]

        X_p = np.array([pos_p[0], pos_p[1], 0.0, 0.0])
        X_e = np.array([pos_e[0], pos_e[1], 0.0, 0.0])

        dynamics = OrbitalDynamics2D(mu=3.986e5, a_c=a_c, e_c=e_c)
        r_c, nu_dot, nu_ddot = dynamics.get_orbital_params(nu)
        a_sdc = dynamics.get_SDC_matrix(X_p, X_e, r_c, nu_dot, nu_ddot)

        try:
            p = _solve_are_balanced(a_sdc, b_p, q, r_eff)
        except (LinAlgError, ValueError):
            fail_count += 1
            continue

        if not _is_spd(p):
            fail_count += 1
            continue

        feat = extract_asdc_features(a_sdc)  # (5,)
        feat_raw.append(feat)
        p_true.append(p)
        a_sdc_all.append(a_sdc)
        x_rel_all.append(X_p - X_e)

    n_lhs_valid = len(feat_raw)
    print(f"  [LHS] {n_lhs_valid}/{n_samples} valid ({fail_count} failed)")

    if n_rollouts > 0:
        print(f"  [Rollout] {n_rollouts} trajectories × {steps_per_rollout} steps ...")
        rf, rp, ra, rx, rfail = _run_rollouts_2d(
            bounds, b_p, q, r, r_eff, gamma,
            n_rollouts=n_rollouts,
            steps_per_rollout=steps_per_rollout,
            dt=rollout_dt,
            seed=seed + 9999,
        )
        feat_raw.extend(rf)
        p_true.extend(rp)
        a_sdc_all.extend(ra)
        x_rel_all.extend(rx)
        fail_count += rfail
        print(f"  [Rollout] {len(rf)} samples collected ({rfail} failed)")

    if len(feat_raw) == 0:
        raise RuntimeError("数据生成失败：没有可用样本。")

    feat_raw_np = np.asarray(feat_raw, dtype=np.float64)
    p_true_np = np.asarray(p_true, dtype=np.float64)
    a_sdc_np = np.asarray(a_sdc_all, dtype=np.float64)
    x_rel_np = np.asarray(x_rel_all, dtype=np.float64)

    # Z-score 标准化输入特征
    feat_mean = feat_raw_np.mean(axis=0)
    feat_std = feat_raw_np.std(axis=0)
    feat_std = np.where(feat_std < 1e-12, 1.0, feat_std)
    feat_norm = (feat_raw_np - feat_mean) / feat_std

    # Log-Cholesky 标签：对角取 log，非对角直接输出
    l_vecs = np.array(
        [_p_to_log_cholesky_vec(p) for p in p_true_np],
        dtype=np.float64,
    )  # (N, 10)
    l_mean = l_vecs.mean(axis=0)
    l_std = l_vecs.std(axis=0)
    l_std = np.where(l_std < 1e-12, 1.0, l_std)
    l_norm = (l_vecs - l_mean) / l_std

    rng = np.random.default_rng(seed)
    n_valid = feat_norm.shape[0]
    indices = rng.permutation(n_valid)
    n_val = int(n_valid * val_ratio)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    np.savez_compressed(
        output,
        feat_raw=feat_raw_np,
        feat_norm=feat_norm,
        p_true=p_true_np,
        a_sdc=a_sdc_np,
        x_rel=x_rel_np,
        feat_mean=feat_mean,
        feat_std=feat_std,
        l_norm=l_norm,
        l_mean=l_mean,
        l_std=l_std,
        train_idx=train_idx,
        val_idx=val_idx,
        q=q,
        r=r,
        gamma=np.array([gamma]),
        state_dim=np.array([STATE_DIM]),
        log_cholesky=np.array([True]),
    )

    summary = {
        "output_path": str(output),
        "requested_samples": int(n_samples),
        "valid_samples": int(n_valid),
        "failed_samples": int(fail_count),
        "success_rate": float(n_valid / n_samples),
    }
    print("2D 数据集生成完成:", summary)
    return summary


def build_dataloaders(
    dataset_path: str,
    batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
) -> Tuple[DataLoader, DataLoader, dict]:
    data = np.load(dataset_path)
    has_p = "p_true" in data
    has_x = "x_rel" in data

    def _sub(key, idx):
        return data[key][idx] if key in data else None

    train_ds = PINNDataset2D(
        data["feat_norm"][data["train_idx"]],
        data["l_norm"][data["train_idx"]],
        data["a_sdc"][data["train_idx"]],
        p_true=_sub("p_true", data["train_idx"]),
        x_rel=_sub("x_rel", data["train_idx"]),
    )
    val_ds = PINNDataset2D(
        data["feat_norm"][data["val_idx"]],
        data["l_norm"][data["val_idx"]],
        data["a_sdc"][data["val_idx"]],
        p_true=_sub("p_true", data["val_idx"]),
        x_rel=_sub("x_rel", data["val_idx"]),
    )

    dl_kwargs: dict = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = persistent_workers
        dl_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **dl_kwargs)

    stats = {
        "feat_mean": data["feat_mean"],
        "feat_std": data["feat_std"],
        "l_mean": data["l_mean"],
        "l_std": data["l_std"],
        "q": data["q"],
        "r": data["r"],
        "gamma": float(data["gamma"][0]),
    }
    return train_loader, val_loader, stats


def main():
    generate_dataset(
        n_samples=500_000,
        output_path="data/sdre_pinn_2d_dataset.npz",
        val_ratio=0.2,
        seed=42,
    )


if __name__ == "__main__":
    main()
