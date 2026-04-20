"""
将 PINN checkpoint 导出为 ONNX 格式

导出的模型包含完整推断流水线：
  归一化输入 → 网络前向 → 反归一化输出 → Log-Cholesky → SPD P 矩阵

输入:  (B, feat_dim) float32  原始 A_SDC 特征（无需预处理）
输出:  (B, n, n)     float32  SPD P 矩阵

用法:
    uv run python -m aerospace.pinn.export_onnx          # 导出 2D
    uv run python -m aerospace.pinn.export_onnx --dim 3  # 导出 3D
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from aerospace.pinn.checkpoint_utils import normalize_pinn_state_dict


class _NormBackbone(nn.Module):
    """归一化 + 网络前向 + 反归一化 → Log-Cholesky L 向量 (B, l_dim)

    输出的 L 向量中对角位置仍是 log 空间值（exp 留给 C++ 端做），
    这样 ONNX 图不含动态 batch 不兼容的 scatter 操作。
    """

    def __init__(self, backbone: nn.Module, feat_mean, feat_std, l_mean, l_std):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("feat_mean", torch.tensor(feat_mean, dtype=torch.float32))
        self.register_buffer("feat_std", torch.tensor(feat_std, dtype=torch.float32))
        self.register_buffer("l_mean", torch.tensor(l_mean, dtype=torch.float32))
        self.register_buffer("l_std", torch.tensor(l_std, dtype=torch.float32))

    def forward(self, feat_raw: torch.Tensor) -> torch.Tensor:
        feat_norm = (feat_raw - self.feat_mean) / self.feat_std
        l_norm = self.backbone(feat_norm)
        return l_norm * self.l_std + self.l_mean


def export_2d(
    checkpoint_path: str = "checkpoints/sdre_pinn_2d/best_model.pt",
    output_path: str = "checkpoints/sdre_pinn_2d/model.onnx",
):
    from aerospace.pinn.pinn_trainer_2d import SDREPINN2D

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = SDREPINN2D(
        in_dim=int(ckpt["in_dim"]),
        backbone_dim=int(ckpt.get("backbone_dim", 256)),
        n_resblocks=int(ckpt.get("n_resblocks", 3)),
        head_hidden=int(ckpt.get("head_hidden", 64)),
        activation=str(ckpt.get("activation", "mish")),
    )
    sd = normalize_pinn_state_dict(ckpt["model_state_dict"])
    model.load_state_dict(sd, strict=True)
    model.eval()

    pipeline = _NormBackbone(
        backbone=model,
        feat_mean=np.asarray(ckpt["feat_mean"], dtype=np.float32),
        feat_std=np.where(
            np.asarray(ckpt["feat_std"], dtype=np.float32) < 1e-12,
            1.0,
            np.asarray(ckpt["feat_std"], dtype=np.float32),
        ),
        l_mean=np.asarray(ckpt["l_mean"], dtype=np.float32),
        l_std=np.asarray(ckpt["l_std"], dtype=np.float32),
    )
    pipeline.eval()

    feat_dim = int(ckpt["in_dim"])
    feat_mean_t = torch.tensor(np.asarray(ckpt["feat_mean"], dtype=np.float32))
    feat_std_t = torch.tensor(np.where(
        np.asarray(ckpt["feat_std"], dtype=np.float32) < 1e-12, 1.0,
        np.asarray(ckpt["feat_std"], dtype=np.float32),
    ))
    dummy = feat_mean_t.unsqueeze(0) + 0.5 * feat_std_t.unsqueeze(0)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        pipeline, dummy, output_path,
        input_names=["feat_raw"],
        output_names=["l_vec"],
        dynamic_axes={"feat_raw": {0: "batch"}, "l_vec": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )

    _verify_onnx(output_path, dummy, pipeline)

    meta = {
        "feat_dim": feat_dim,
        "state_dim": 4,
        "l_dim": 10,
        "diag_indices": [0, 2, 5, 9],
        "delta_spd": float(ckpt["delta_spd"]),
        "q": ckpt["q"].tolist() if hasattr(ckpt["q"], "tolist") else ckpt["q"],
        "r": ckpt["r"].tolist() if hasattr(ckpt["r"], "tolist") else ckpt["r"],
        "gamma": float(ckpt["gamma"]),
    }
    import json
    meta_path = str(Path(output_path).with_suffix(".json"))
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[2D] ONNX exported: {output_path}")
    print(f"[2D] Metadata:      {meta_path}")
    return output_path


def export_3d(
    checkpoint_path: str = "checkpoints/sdre_pinn/best_model.pt",
    output_path: str = "checkpoints/sdre_pinn/model.onnx",
):
    from aerospace.pinn.pinn_trainer import SDREPINN

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = SDREPINN(
        in_dim=int(ckpt["in_dim"]),
        backbone_dim=int(ckpt.get("backbone_dim", 256)),
        n_resblocks=int(ckpt.get("n_resblocks", 3)),
        head_hidden=int(ckpt.get("head_hidden", 64)),
        activation=str(ckpt.get("activation", "mish")),
    )
    sd = normalize_pinn_state_dict(ckpt["model_state_dict"])
    model.load_state_dict(sd, strict=True)
    model.eval()

    pipeline = _NormBackbone(
        backbone=model,
        feat_mean=np.asarray(ckpt["feat_mean"], dtype=np.float32),
        feat_std=np.where(
            np.asarray(ckpt["feat_std"], dtype=np.float32) < 1e-12,
            1.0,
            np.asarray(ckpt["feat_std"], dtype=np.float32),
        ),
        l_mean=np.asarray(ckpt["l_mean"], dtype=np.float32),
        l_std=np.asarray(ckpt["l_std"], dtype=np.float32),
    )
    pipeline.eval()

    feat_dim = int(ckpt["in_dim"])
    feat_mean_t = torch.tensor(np.asarray(ckpt["feat_mean"], dtype=np.float32))
    feat_std_t = torch.tensor(np.where(
        np.asarray(ckpt["feat_std"], dtype=np.float32) < 1e-12, 1.0,
        np.asarray(ckpt["feat_std"], dtype=np.float32),
    ))
    dummy = feat_mean_t.unsqueeze(0) + 0.5 * feat_std_t.unsqueeze(0)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        pipeline, dummy, output_path,
        input_names=["feat_raw"],
        output_names=["l_vec"],
        dynamic_axes={"feat_raw": {0: "batch"}, "l_vec": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )

    _verify_onnx(output_path, dummy, pipeline)

    meta = {
        "feat_dim": feat_dim,
        "state_dim": 6,
        "l_dim": 21,
        "diag_indices": [0, 2, 5, 9, 14, 20],
        "delta_spd": float(ckpt["delta_spd"]),
        "q": ckpt["q"].tolist() if hasattr(ckpt["q"], "tolist") else ckpt["q"],
        "r": ckpt["r"].tolist() if hasattr(ckpt["r"], "tolist") else ckpt["r"],
        "gamma": float(ckpt["gamma"]),
    }
    import json
    meta_path = str(Path(output_path).with_suffix(".json"))
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[3D] ONNX exported: {output_path}")
    print(f"[3D] Metadata:      {meta_path}")
    return output_path


def _verify_onnx(onnx_path, dummy_input, pipeline):
    """用 onnxruntime 验证导出精度。"""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  (onnxruntime not installed, skipping verification)")
        return

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = pipeline(dummy_input).numpy()
    ort_out = sess.run(None, {"feat_raw": dummy_input.numpy()})[0]
    max_diff = np.max(np.abs(ref - ort_out))
    rel_diff = np.max(np.abs(ref - ort_out) / np.maximum(np.abs(ref), 1e-12))
    print(f"  ONNX verification: abs_diff={max_diff:.4e}  rel_diff={rel_diff:.4e}  "
          f"{'PASS' if rel_diff < 1e-4 else 'WARN'}")


def main():
    parser = argparse.ArgumentParser(description="Export PINN to ONNX")
    parser.add_argument("--dim", type=int, default=2, choices=[2, 3])
    args = parser.parse_args()

    if args.dim == 2:
        export_2d()
    else:
        export_3d()


if __name__ == "__main__":
    main()
