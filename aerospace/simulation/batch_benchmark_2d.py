"""
2D 海量初值批量推断性能测试

CARE vs PINN 速度 + Frobenius 误差 + 相对 Frobenius 误差对比。
输入特征：A_SDC 5D 非平凡元素。
"""

from __future__ import annotations

import time

import numpy as np
from scipy.linalg import LinAlgError, solve_continuous_are
from scipy.stats import qmc

from aerospace.control.neural_2d import NeuralSDREController2D
from aerospace.pinn.data_generator_2d import (
    STATE_DIM, CTRL_DIM, _solve_are_balanced, extract_asdc_features,
)
from aerospace.dynamics.nerm_2d import OrbitalDynamics2D


def _generate_eval_states_2d(
    n_samples: int = 10_000,
    seed: int = 2026,
    pos_min_km: float = -1000.0,
    pos_max_km: float = 1000.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成评测用绝对位置和真近点角。"""
    low = np.array([pos_min_km, pos_min_km])
    high = np.array([pos_max_km, pos_max_km])

    sampler_p = qmc.LatinHypercube(d=2, seed=seed)
    pos_p = qmc.scale(sampler_p.random(n=n_samples), low, high)

    sampler_e = qmc.LatinHypercube(d=2, seed=seed + 7)
    pos_e = qmc.scale(sampler_e.random(n=n_samples), low, high)

    sampler_nu = qmc.LatinHypercube(d=1, seed=seed + 11)
    nu = 2.0 * np.pi * sampler_nu.random(n=n_samples).reshape(-1)
    return pos_p, pos_e, nu


def run_batch_inference_test(
    checkpoint_path: str = "checkpoints/sdre_pinn_2d/best_model.pt",
    n_samples: int = 10_000,
    seed: int = 2026,
) -> dict:
    dynamics = OrbitalDynamics2D(mu=3.986e5, a_c=15000.0, e_c=0.5)
    ctrl = NeuralSDREController2D(checkpoint_path=checkpoint_path)

    q = ctrl.q
    r = ctrl.r
    gamma = ctrl.gamma
    b_p = np.zeros((STATE_DIM, CTRL_DIM), dtype=np.float64)
    b_p[2:, :] = np.eye(CTRL_DIM)
    r_eff = r / (1.0 - gamma ** (-2))

    pos_p_batch, pos_e_batch, nu_batch = _generate_eval_states_2d(
        n_samples=n_samples, seed=seed,
    )

    feat_raw_valid = []
    p_true_valid = []

    # Task A: CARE 循环
    t0 = time.perf_counter()
    care_fail = 0
    for i in range(n_samples):
        pos_p = pos_p_batch[i]
        pos_e = pos_e_batch[i]
        nu = float(nu_batch[i])
        X_p = np.array([pos_p[0], pos_p[1], 0.0, 0.0])
        X_e = np.array([pos_e[0], pos_e[1], 0.0, 0.0])
        r_c, nu_dot, nu_ddot = dynamics.get_orbital_params(nu)
        a_sdc = dynamics.get_SDC_matrix(X_p, X_e, r_c, nu_dot, nu_ddot)

        try:
            p = _solve_are_balanced(a_sdc, b_p, q, r_eff)
        except (LinAlgError, ValueError):
            care_fail += 1
            continue

        feat_raw_valid.append(extract_asdc_features(a_sdc))
        p_true_valid.append(p)
    t_care = time.perf_counter() - t0

    if len(p_true_valid) == 0:
        raise RuntimeError("CARE 全部失败。")

    feat_raw_valid = np.asarray(feat_raw_valid, dtype=np.float64)
    p_true_valid = np.asarray(p_true_valid, dtype=np.float64)

    # Task B: PINN batch
    t1 = time.perf_counter()
    p_pred_valid = ctrl.predict_p_batch(feat_raw_valid, batch_size=2048)
    t_pinn = time.perf_counter() - t1

    diff = p_pred_valid - p_true_valid
    fro_err = np.sqrt(np.sum(diff * diff, axis=(1, 2)))

    p_true_norms = np.sqrt(np.sum(p_true_valid * p_true_valid, axis=(1, 2)))
    p_true_norms_safe = np.where(p_true_norms < 1e-12, 1.0, p_true_norms)
    rel_fro_err = fro_err / p_true_norms_safe

    result = {
        "requested_samples": int(n_samples),
        "valid_samples": int(p_true_valid.shape[0]),
        "care_fail_samples": int(care_fail),
        "care_total_time_s": float(t_care),
        "pinn_total_time_s": float(t_pinn),
        "speedup": float(t_care / t_pinn if t_pinn > 1e-12 else np.inf),
        "fro_error_mean": float(np.mean(fro_err)),
        "fro_error_p50": float(np.percentile(fro_err, 50)),
        "fro_error_p90": float(np.percentile(fro_err, 90)),
        "fro_error_p99": float(np.percentile(fro_err, 99)),
        "rel_fro_error_mean": float(np.mean(rel_fro_err)),
        "rel_fro_error_p50": float(np.percentile(rel_fro_err, 50)),
        "rel_fro_error_p90": float(np.percentile(rel_fro_err, 90)),
        "rel_fro_error_p99": float(np.percentile(rel_fro_err, 99)),
    }

    print("2D Batch Inference Test Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    return result


def main():
    run_batch_inference_test()


if __name__ == "__main__":
    main()
