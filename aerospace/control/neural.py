"""
3D 神经代理 SDRE 控制器与闭环性能对比

用 PINN 近似在线 ARE 求解，含 OOD 安全回退。
输入特征 (10D)：从 A_SDC 提取独立非平凡元素。
输出使用 Log-Cholesky 参数化。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from scipy.linalg import LinAlgError, solve_continuous_are

from aerospace.pinn.checkpoint_utils import normalize_pinn_state_dict
from aerospace.pinn.pinn_trainer import SDREPINN, reconstruct_spd_p
from aerospace.pinn.data_generator import STATE_DIM, CTRL_DIM, extract_asdc_features
from aerospace.control.sdre import SDREGameController
from aerospace.dynamics.nerm import OrbitalDynamics
from aerospace.simulation.nerm_sdre import SDRESimulation


@dataclass
class SimBenchmarkResult:
    baseline_total_time_s: float
    neural_total_time_s: float
    baseline_fps: float
    neural_fps: float
    speedup: float
    neural_fallback_count: int
    neural_total_calls: int
    neural_fallback_ratio: float
    baseline_final_distance_km: float
    neural_final_distance_km: float


class NeuralSDREController:
    """基于 PINN 的 3D SDRE 代理控制器，接口与 SDREGameController 兼容。"""

    def __init__(
        self,
        checkpoint_path: str,
        ood_zscore_threshold: float = 5.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.device = torch.device(device)

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

        self.l_mean_t = torch.from_numpy(
            np.asarray(ckpt["l_mean"], dtype=np.float32)
        ).to(self.device)
        self.l_std_t = torch.from_numpy(
            np.asarray(ckpt["l_std"], dtype=np.float32)
        ).to(self.device)

        self.q = np.asarray(ckpt["q"], dtype=np.float64)
        self.r = np.asarray(ckpt["r"], dtype=np.float64)
        self.gamma = float(ckpt["gamma"])
        self.r_inv = np.linalg.inv(self.r)

        self.b_p = np.zeros((STATE_DIM, CTRL_DIM), dtype=np.float64)
        self.b_p[3:, :] = np.eye(CTRL_DIM)
        self.b_e = -self.b_p

        self.ood_zscore_threshold = ood_zscore_threshold
        self.last_p: np.ndarray | None = None

        self.total_calls = 0
        self.fallback_calls = 0
        self.net_infer_time_s = 0.0
        self.are_fallback_time_s = 0.0
        self.step_times: list[float] = []

    def set_environment(self, r_c: float, nu_dot: float, nu_ddot: float) -> None:
        """保留接口兼容性。"""
        pass

    def set_positions(self, X_p: np.ndarray, X_e: np.ndarray) -> None:
        """保留接口兼容性。"""
        pass

    def _build_feature(self, x_rel: np.ndarray, a_sdc: np.ndarray) -> np.ndarray:
        """直接从 A_SDC 提取 10 个独立非平凡元素。"""
        return extract_asdc_features(a_sdc)

    def _is_ood(self, feat_raw: np.ndarray) -> bool:
        z = np.abs((feat_raw - self.feat_mean) / self.feat_std)
        return bool(np.any(z > self.ood_zscore_threshold))

    def _solve_are_fallback(self, a_sdc: np.ndarray) -> np.ndarray:
        r_eff = self.r / (1.0 - self.gamma ** (-2))
        try:
            p = solve_continuous_are(a_sdc, self.b_p, self.q, r_eff)
            self.last_p = p
            return p
        except (LinAlgError, ValueError):
            if self.last_p is not None:
                return self.last_p
            return np.zeros((STATE_DIM, STATE_DIM), dtype=np.float64)

    def compute_control(self, A_SDC: np.ndarray, x_rel: np.ndarray, t=None, solve_are=True) -> tuple[np.ndarray, np.ndarray]:
        self.total_calls += 1
        _t0 = time.perf_counter()
        feat_raw = self._build_feature(x_rel, A_SDC)

        if self._is_ood(feat_raw):
            self.fallback_calls += 1
            t0 = time.perf_counter()
            p = self._solve_are_fallback(A_SDC)
            self.are_fallback_time_s += time.perf_counter() - t0
        else:
            feat_norm = (feat_raw - self.feat_mean) / self.feat_std
            feat_t = torch.from_numpy(feat_norm.astype(np.float32)).unsqueeze(0).to(self.device)

            t0 = time.perf_counter()
            with torch.no_grad():
                l_pred_norm = self.model(feat_t)
                l_pred = l_pred_norm * self.l_std_t + self.l_mean_t
                p_t = reconstruct_spd_p(l_pred, delta=self.delta_spd)
            self.net_infer_time_s += time.perf_counter() - t0
            p = p_t.squeeze(0).detach().cpu().numpy().astype(np.float64)
            self.last_p = p

        self.last_P = p

        u_p = -self.r_inv @ self.b_p.T @ p @ x_rel
        u_e = self.gamma ** (-2) * self.r_inv @ self.b_e.T @ p @ x_rel
        self.step_times.append(time.perf_counter() - _t0)
        return u_p, u_e

    def predict_p_batch(self, feat_raw_batch: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """批量预测 P 矩阵。feat_raw_batch 应为 (N, 10) A_SDC 特征。"""
        feat_raw_batch = np.asarray(feat_raw_batch, dtype=np.float64)
        feat_norm = (feat_raw_batch - self.feat_mean) / self.feat_std

        n = feat_norm.shape[0]
        p_all = []
        with torch.no_grad():
            for i in range(0, n, batch_size):
                chunk = feat_norm[i: i + batch_size]
                feat_t = torch.from_numpy(chunk.astype(np.float32)).to(self.device)
                l_pred_norm = self.model(feat_t)
                l_pred = l_pred_norm * self.l_std_t + self.l_mean_t
                p_t = reconstruct_spd_p(l_pred, delta=self.delta_spd)
                p_all.append(p_t.cpu().numpy())
        return np.concatenate(p_all, axis=0)


def _relative_distance_km(states: np.ndarray) -> float:
    x_rel = states[0:3, -1] - states[6:9, -1]
    return float(np.linalg.norm(x_rel))


def run_closed_loop_benchmark(
    checkpoint_path: str,
    dt: float = 20.0,
) -> SimBenchmarkResult:
    dynamics = OrbitalDynamics(mu=3.986e5, a_c=6771.0, e_c=0)

    q = np.eye(STATE_DIM)
    r = np.eye(CTRL_DIM) * 1e13
    gamma = np.sqrt(2)

    x_p0 = np.array([500.0, 500.0, 500.0, 0.01, 0.01, 0.01], dtype=float)
    x_e0 = np.zeros(STATE_DIM, dtype=float)

    baseline_ctrl = SDREGameController(q, r, gamma)
    baseline_sim = SDRESimulation(
        dynamics=dynamics, controller=baseline_ctrl,
        X_p0=x_p0, X_e0=x_e0, nu0=0.0, dt=dt,
    )
    t0 = time.perf_counter()
    baseline_res = baseline_sim.run()
    baseline_time = time.perf_counter() - t0

    neural_ctrl = NeuralSDREController(checkpoint_path=checkpoint_path)
    neural_sim = SDRESimulation(
        dynamics=dynamics, controller=neural_ctrl,
        X_p0=x_p0, X_e0=x_e0, nu0=0.0, dt=dt,
    )
    t1 = time.perf_counter()
    neural_res = neural_sim.run()
    neural_time = time.perf_counter() - t1

    steps = max(1, baseline_res.t.size)
    return SimBenchmarkResult(
        baseline_total_time_s=baseline_time,
        neural_total_time_s=neural_time,
        baseline_fps=steps / baseline_time,
        neural_fps=steps / neural_time,
        speedup=baseline_time / neural_time if neural_time > 1e-12 else np.inf,
        neural_fallback_count=neural_ctrl.fallback_calls,
        neural_total_calls=neural_ctrl.total_calls,
        neural_fallback_ratio=(
            neural_ctrl.fallback_calls / neural_ctrl.total_calls
            if neural_ctrl.total_calls > 0 else 0.0
        ),
        baseline_final_distance_km=_relative_distance_km(baseline_res.states),
        neural_final_distance_km=_relative_distance_km(neural_res.states),
    )


def main() -> None:
    # 对应 C++ start_JT：模型加载（离线预计算阶段）
    start_JT = time.perf_counter()
    neural_ctrl = NeuralSDREController(checkpoint_path="checkpoints/sdre_pinn/best_model.pt")
    end_JT = time.perf_counter()
    print(f"模型加载时间: {end_JT - start_JT:.4f}s")

    dynamics = OrbitalDynamics(mu=3.986e5, a_c=15000.0, e_c=0.5)
    x_p0 = np.array([500.0, 500.0, 500.0, 0.01, 0.01, 0.01], dtype=float)
    x_e0 = np.zeros(6, dtype=float)

    from aerospace.simulation.nerm_sdre import SDRESimulation
    sim = SDRESimulation(dynamics=dynamics, controller=neural_ctrl,
                         X_p0=x_p0, X_e0=x_e0, nu0=0.0, dt=20.0)

    # 对应 C++ start：仿真阶段
    start = time.perf_counter()
    result = sim.run()
    end = time.perf_counter()
    print(f"仿真时间: {end - start:.4f}s")
    times = np.array(neural_ctrl.step_times)
    print(f"单步控制计算时间  均值: {times.mean()*1e3:.4f}ms  最大: {times.max()*1e3:.4f}ms  总计: {times.sum():.4f}s")


if __name__ == "__main__":
    main()
