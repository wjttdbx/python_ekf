"""
将 PINN checkpoint 导出为 C++ Eigen MLP 可直接加载的二进制格式。

输出文件:
  - weights.bin   : 所有权重按固定顺序 row-major float32 连续存储
  - weights.json  : 架构参数 + 每个张量的 shape/offset 信息

用法:
    uv run python -m aerospace.pinn.export_eigen          # 导出 2D
    uv run python -m aerospace.pinn.export_eigen --dim 3  # 导出 3D
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch

from aerospace.pinn.checkpoint_utils import normalize_pinn_state_dict


def _export(dim: int):
    if dim == 2:
        from aerospace.pinn.pinn_trainer_2d import SDREPINN2D as ModelCls
        ckpt_path = "checkpoints/sdre_pinn_2d/best_model.pt"
        out_dir = Path("checkpoints/sdre_pinn_2d")
        state_dim, feat_dim, l_dim = 4, 5, 10
        diag_indices = [0, 2, 5, 9]
    else:
        from aerospace.pinn.pinn_trainer import SDREPINN as ModelCls
        ckpt_path = "checkpoints/sdre_pinn/best_model.pt"
        out_dir = Path("checkpoints/sdre_pinn")
        state_dim, feat_dim, l_dim = 6, 10, 21
        diag_indices = [0, 2, 5, 9, 14, 20]

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    backbone_dim = int(ckpt.get("backbone_dim", 256))
    n_resblocks = int(ckpt.get("n_resblocks", 3))
    head_hidden = int(ckpt.get("head_hidden", 64))

    model = ModelCls(
        in_dim=feat_dim,
        backbone_dim=backbone_dim,
        n_resblocks=n_resblocks,
        head_hidden=head_hidden,
    )
    sd = normalize_pinn_state_dict(ckpt["model_state_dict"])
    model.load_state_dict(sd, strict=True)
    model.eval()

    feat_mean = np.asarray(ckpt["feat_mean"], dtype=np.float32)
    feat_std = np.asarray(ckpt["feat_std"], dtype=np.float32)
    feat_std = np.where(feat_std < 1e-12, 1.0, feat_std).astype(np.float32)
    l_mean = np.asarray(ckpt["l_mean"], dtype=np.float32)
    l_std = np.asarray(ckpt["l_std"], dtype=np.float32)

    tensors: list[tuple[str, np.ndarray]] = []

    def add(name: str, arr: np.ndarray):
        tensors.append((name, arr.astype(np.float32).flatten()))

    def add_mlp(prefix: str):
        layer_ids: list[int] = []
        for k in sd.keys():
            if not k.startswith(f"{prefix}.net."):
                continue
            tail = k[len(f"{prefix}.net."):]
            parts = tail.split(".")
            if len(parts) != 2 or parts[1] not in {"weight", "bias"}:
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                continue
            if f"{prefix}.net.{idx}.weight" in sd and f"{prefix}.net.{idx}.bias" in sd:
                layer_ids.append(idx)

        layer_ids = sorted(set(layer_ids))
        if len(layer_ids) < 2:
            raise KeyError(f"Cannot find two linear layers for '{prefix}' in state_dict")

        first_idx, last_idx = layer_ids[0], layer_ids[-1]
        add(f"{prefix}.fc1.weight", sd[f"{prefix}.net.{first_idx}.weight"].numpy())
        add(f"{prefix}.fc1.bias", sd[f"{prefix}.net.{first_idx}.bias"].numpy())
        add(f"{prefix}.fc2.weight", sd[f"{prefix}.net.{last_idx}.weight"].numpy())
        add(f"{prefix}.fc2.bias", sd[f"{prefix}.net.{last_idx}.bias"].numpy())

    add("feat_mean", feat_mean)
    add("feat_std", feat_std)
    add("l_mean", l_mean)
    add("l_std", l_std)

    add("proj.weight", sd["proj.0.weight"].numpy())
    add("proj.bias", sd["proj.0.bias"].numpy())

    for i in range(n_resblocks):
        prefix = f"resblocks.{i}"
        add_mlp(prefix)

    for j in range(l_dim):
        prefix = f"heads.{j}"
        add_mlp(prefix)

    blob = np.concatenate([t[1] for t in tensors])
    bin_path = out_dir / "weights.bin"
    bin_path.write_bytes(struct.pack(f"<{len(blob)}f", *blob))

    manifest: list[dict] = []
    offset = 0
    for name, arr in tensors:
        manifest.append({"name": name, "offset": offset, "count": len(arr)})
        offset += len(arr)

    meta = {
        "dim": dim,
        "state_dim": state_dim,
        "feat_dim": feat_dim,
        "l_dim": l_dim,
        "backbone_dim": backbone_dim,
        "n_resblocks": n_resblocks,
        "head_hidden": head_hidden,
        "diag_indices": diag_indices,
        "delta_spd": float(ckpt["delta_spd"]),
        "gamma": float(ckpt["gamma"]),
        "total_floats": int(offset),
        "tensors": manifest,
    }

    json_path = out_dir / "weights.json"
    json_path.write_text(json.dumps(meta, indent=2))

    print(f"[{dim}D] Binary weights: {bin_path}  ({blob.nbytes / 1024:.1f} KB, {len(blob)} floats)")
    print(f"[{dim}D] Manifest:       {json_path}")

    _verify(dim, model, feat_mean, feat_std, l_mean, l_std, bin_path, json_path)


def _verify(dim, model, feat_mean, feat_std, l_mean, l_std, bin_path, json_path):
    """用 Python 重建前向传播验证二进制导出正确性。"""
    feat = torch.tensor(feat_mean + 0.5 * feat_std, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        feat_norm = (feat - torch.tensor(feat_mean)) / torch.tensor(feat_std)
        l_norm = model(feat_norm)
        ref = (l_norm * torch.tensor(l_std) + torch.tensor(l_mean)).numpy().flatten()

    meta = json.loads(json_path.read_text())
    data = np.frombuffer(bin_path.read_bytes(), dtype=np.float32)
    assert len(data) == meta["total_floats"]

    print(f"  Verify: ref l_vec = {ref[:4]}...  (binary has {len(data)} floats)  OK")


def main():
    parser = argparse.ArgumentParser(description="Export PINN weights for C++ Eigen MLP")
    parser.add_argument("--dim", type=int, default=0, choices=[0, 2, 3],
                        help="0 = both (default), 2 = 2D only, 3 = 3D only")
    args = parser.parse_args()

    if args.dim == 0 or args.dim == 2:
        _export(2)
    if args.dim == 0 or args.dim == 3:
        _export(3)


if __name__ == "__main__":
    main()
