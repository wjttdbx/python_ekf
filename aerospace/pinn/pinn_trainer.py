"""
3D SDRE-PINN 训练模块

输入 10D A_SDC 特征 → 残差主干 (3×ResBlock 256) → 21 个独立任务头
Log-Cholesky 参数化：对角头输出 d_i → L_ii = exp(d_i)，非对角头直接输出 L_ij
P = L L^T + δI₆
Data Loss + Physics Loss (ARE 残差)

用法：uv run python -m aerospace.pinn.pinn_trainer
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from aerospace.pinn.checkpoint_utils import normalize_pinn_state_dict
from aerospace.pinn.data_generator import build_dataloaders, STATE_DIM, CTRL_DIM, L_DIM

# tril_indices(6) 中对角元素位置: (0,0)→0, (1,1)→2, (2,2)→5, (3,3)→9, (4,4)→14, (5,5)→20
DIAG_INDICES = [0, 2, 5, 9, 14, 20]
OFFDIAG_INDICES = [i for i in range(L_DIM) if i not in DIAG_INDICES]


def log_cholesky_to_cholesky_l(vec: torch.Tensor) -> torch.Tensor:
    """(B, 21) Log-Cholesky 向量 → (B, 6, 6) 下三角矩阵。"""
    bsz = vec.shape[0]
    l = torch.zeros((bsz, STATE_DIM, STATE_DIM), dtype=vec.dtype, device=vec.device)
    tril_idx = torch.tril_indices(row=STATE_DIM, col=STATE_DIM, offset=0, device=vec.device)

    vec_exp = vec.clone()
    for di in DIAG_INDICES:
        # Clamp log-diagonal before exp to avoid overflow in early training.
        vec_exp[:, di] = torch.exp(torch.clamp(vec[:, di], min=-15.0, max=15.0))

    l[:, tril_idx[0], tril_idx[1]] = vec_exp
    return l


def reconstruct_spd_p(vec: torch.Tensor, delta: float = 1e-4) -> torch.Tensor:
    """由 21 维 Log-Cholesky 输出重构 SPD P (B, 6, 6)。"""
    l = log_cholesky_to_cholesky_l(vec)
    p = torch.bmm(l, l.transpose(1, 2))
    eye = torch.eye(STATE_DIM, dtype=p.dtype, device=p.device).unsqueeze(0)
    return p + delta * eye


class ResBlock(nn.Module):
    """残差块: x + [Linear → Mish → Dropout → Linear]"""

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.act = nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class TaskHead(nn.Module):
    """独立任务头: Linear(backbone_dim → hidden) → Mish → Linear(hidden → 1)"""

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Mish(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SDREPINN(nn.Module):
    """残差主干 + 21 路独立任务头

    Backbone: Linear(in_dim → 256) + Mish + 3×ResBlock(256)
    Heads:    21 × TaskHead(256 → 64 → 1)
    """

    def __init__(
        self,
        in_dim: int = 10,
        backbone_dim: int = 256,
        n_resblocks: int = 3,
        head_hidden: int = 64,
        activation: str = "mish",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.backbone_dim = backbone_dim

        self.proj = nn.Sequential(
            nn.Linear(in_dim, backbone_dim),
            nn.Mish(),
        )

        self.resblocks = nn.Sequential(
            *[ResBlock(backbone_dim, dropout=dropout) for _ in range(n_resblocks)]
        )

        self.heads = nn.ModuleList([
            TaskHead(backbone_dim, head_hidden) for _ in range(L_DIM)
        ])

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        h = self.proj(feat)
        h = self.resblocks(h)
        outputs = [head(h) for head in self.heads]
        return torch.cat(outputs, dim=1)  # (B, 21)


@dataclass
class TrainConfig:
    dataset_path: str = "data/sdre_pinn_dataset.npz"
    output_dir: str = "checkpoints/sdre_pinn/ctrl_loss05_phys_loss20"
    epochs: int = 600
    batch_size: int = 8192
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    auto_batch_size: bool = True
    auto_batch_max: int = 32768
    auto_batch_growth_factor: int = 2
    lr: float = 5e-4
    weight_decay: float = 1e-3
    lambda_phys: float = 0.2
    lambda_ctrl: float = 0.05
    are_den_floor: float = 1e-6
    ctrl_den_floor: float = 1e-2
    rel_loss_clip: float = 1e4
    delta_spd: float = 1e-4
    max_grad_norm: float = 1.0
    enable_amp: bool = True
    amp_dtype: str = "bf16"  # bf16/fp16
    enable_compile: bool = True
    compile_mode: str = "max-autotune"  # default/reduce-overhead/max-autotune
    compile_fullgraph: bool = False
    allow_tf32: bool = True
    activation: str = "mish"
    backbone_dim: int = 256
    n_resblocks: int = 4
    head_hidden: int = 64
    dropout: float = 0.1
    lr_factor: float = 0.5
    lr_patience: int = 30
    early_stop_patience: int = 80
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


def _is_oom_error(err: RuntimeError) -> bool:
    msg = str(err).lower()
    return (
        "out of memory" in msg
        or "cuda error: out of memory" in msg
        or "cublas_status_alloc_failed" in msg
    )


def _resolve_amp_dtype(device: torch.device, dtype_name: str) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    if dtype_name.lower() == "fp16":
        return torch.float16
    return torch.bfloat16


def _maybe_compile_model(model: nn.Module, config: TrainConfig) -> nn.Module:
    if not config.enable_compile or not hasattr(torch, "compile"):
        return model
    return torch.compile(  # type: ignore[attr-defined]
        model,
        mode=config.compile_mode,
        fullgraph=config.compile_fullgraph,
    )


def _build_game_matrix_g(r: np.ndarray, gamma: float, device: torch.device) -> torch.Tensor:
    """G = Bp R⁻¹ Bp^T − γ⁻² Be R⁻¹ Be^T (6×6)。"""
    b_p = np.zeros((STATE_DIM, CTRL_DIM), dtype=np.float64)
    b_p[3:, :] = np.eye(CTRL_DIM)
    b_e = -b_p
    r_inv = np.linalg.inv(r)
    g = b_p @ r_inv @ b_p.T - (gamma ** -2) * (b_e @ r_inv @ b_e.T)
    return torch.from_numpy(g.astype(np.float32)).to(device)


def _build_R_inv_Bp_T(r: np.ndarray, device: torch.device) -> torch.Tensor:
    """预计算 R⁻¹ Bₚᵀ (CTRL_DIM, STATE_DIM)，用于控制量损失。"""
    b_p = np.zeros((STATE_DIM, CTRL_DIM), dtype=np.float64)
    b_p[3:, :] = np.eye(CTRL_DIM)
    r_inv = np.linalg.inv(r)
    return torch.from_numpy((r_inv @ b_p.T).astype(np.float32)).to(device)


def compute_losses(l_pred_norm, l_true_norm, p_pred, p_true, x_rel,
                   a_sdc, q_mat, g_mat, R_inv_Bp_T,
                   lambda_phys, lambda_ctrl,
                   are_den_floor=1e-6, ctrl_den_floor=1e-4, rel_loss_clip=1e4, eps=1e-8):
    loss_data = torch.mean((l_pred_norm - l_true_norm) ** 2)

    bsz = p_pred.shape[0]
    g_batch = g_mat.unsqueeze(0).expand(bsz, -1, -1)
    q_batch = q_mat.unsqueeze(0).expand(bsz, -1, -1)

    p_a = torch.bmm(p_pred, a_sdc)
    a_t_p = torch.bmm(a_sdc.transpose(1, 2), p_pred)
    p_g_p = torch.bmm(torch.bmm(p_pred, g_batch), p_pred)

    r_are = p_a + a_t_p - p_g_p + q_batch

    num = torch.sum(r_are * r_are, dim=(1, 2))
    den = (
        torch.sum(p_a * p_a, dim=(1, 2))
        + torch.sum(p_g_p * p_g_p, dim=(1, 2))
        + torch.sum(q_batch * q_batch, dim=(1, 2))
        + eps
    )
    phys_rel = num / torch.clamp(den, min=are_den_floor)
    phys_rel = torch.clamp(phys_rel, max=rel_loss_clip)
    loss_phys = torch.mean(phys_rel)

    K_batch = R_inv_Bp_T.unsqueeze(0).expand(bsz, -1, -1)
    u_pred = -torch.bmm(torch.bmm(K_batch, p_pred), x_rel.unsqueeze(-1)).squeeze(-1)
    u_true = -torch.bmm(torch.bmm(K_batch, p_true), x_rel.unsqueeze(-1)).squeeze(-1)
    loss_ctrl = torch.mean((u_pred - u_true) ** 2)

    total = loss_data + lambda_phys * loss_phys + lambda_ctrl * loss_ctrl
    return loss_data, loss_phys, loss_ctrl, total


def _probe_batch_size_once(
    model: nn.Module,
    dataset_path: str,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype | None,
    q_mat: torch.Tensor,
    g_mat: torch.Tensor,
    R_inv_Bp_T: torch.Tensor,
    l_mean_t: torch.Tensor,
    l_std_t: torch.Tensor,
    lambda_phys: float,
    lambda_ctrl: float,
    delta_spd: float,
) -> bool:
    train_loader, _, _ = build_dataloaders(
        dataset_path=dataset_path,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2,
    )
    feat, l_norm, a_sdc, p_true, x_rel = next(iter(train_loader))
    feat = feat.to(device, non_blocking=True)
    l_norm = l_norm.to(device, non_blocking=True)
    a_sdc = a_sdc.to(device, non_blocking=True)
    p_true = p_true.to(device, non_blocking=True)
    x_rel = x_rel.to(device, non_blocking=True)

    model.zero_grad(set_to_none=True)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if use_amp
        else contextlib.nullcontext()
    )
    with autocast_ctx:
        l_pred_norm = model(feat)
        l_pred = l_pred_norm * l_std_t + l_mean_t
        p_pred = reconstruct_spd_p(l_pred, delta=delta_spd)
        _, _, _, loss_total = compute_losses(
            l_pred_norm, l_norm, p_pred, p_true, x_rel,
            a_sdc, q_mat, g_mat, R_inv_Bp_T,
            lambda_phys, lambda_ctrl,
            are_den_floor=1e-6,
            ctrl_den_floor=1e-4,
            rel_loss_clip=1e4,
        )
    loss_total.backward()
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return True


def _autotune_batch_size(
    config: TrainConfig,
    model: nn.Module,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype | None,
    q_mat: torch.Tensor,
    g_mat: torch.Tensor,
    R_inv_Bp_T: torch.Tensor,
    l_mean_t: torch.Tensor,
    l_std_t: torch.Tensor,
) -> int:
    if (not config.auto_batch_size) or device.type != "cuda":
        return config.batch_size

    with np.load(config.dataset_path) as data:
        train_size = int(data["train_idx"].shape[0])
    max_probe_bs = min(int(config.auto_batch_max), max(1, train_size))

    current = max(1, int(config.batch_size))
    best = current

    while current <= max_probe_bs:
        try:
            _probe_batch_size_once(
                model=model,
                dataset_path=config.dataset_path,
                batch_size=current,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                q_mat=q_mat,
                g_mat=g_mat,
                R_inv_Bp_T=R_inv_Bp_T,
                l_mean_t=l_mean_t,
                l_std_t=l_std_t,
                lambda_phys=config.lambda_phys,
                lambda_ctrl=config.lambda_ctrl,
                delta_spd=config.delta_spd,
            )
            best = current
            print(f"[AutoBatch] batch_size={current} probe passed")
            next_bs = current * int(config.auto_batch_growth_factor)
            if next_bs == current:
                break
            current = next_bs
        except RuntimeError as e:
            if _is_oom_error(e):
                print(f"[AutoBatch] batch_size={current} OOM, fallback to {best}")
                torch.cuda.empty_cache()
                break
            raise

    return min(best, max_probe_bs)


def train_pinn(config: TrainConfig) -> dict:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, stats = build_dataloaders(
        dataset_path=config.dataset_path,
        batch_size=config.batch_size,
        num_workers=0,
        pin_memory=config.pin_memory,
        persistent_workers=False,
        prefetch_factor=2,
    )

    device = torch.device(config.device)
    if device.type == "cuda" and config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    in_dim = int(stats["feat_mean"].shape[0])
    model = SDREPINN(
        in_dim=in_dim,
        backbone_dim=config.backbone_dim,
        n_resblocks=config.n_resblocks,
        head_hidden=config.head_hidden,
        activation=config.activation,
        dropout=config.dropout,
    ).to(device)
    model = _maybe_compile_model(model, config)

    amp_dtype = _resolve_amp_dtype(device, config.amp_dtype)
    use_amp = config.enable_amp and (amp_dtype is not None)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config.lr_factor,
        patience=config.lr_patience, min_lr=1e-6,
    )

    q_mat = torch.from_numpy(stats["q"].astype(np.float32)).to(device)
    g_mat = _build_game_matrix_g(stats["r"], stats["gamma"], device=device)
    R_inv_Bp_T = _build_R_inv_Bp_T(stats["r"], device=device)

    l_mean_t = torch.from_numpy(stats["l_mean"].astype(np.float32)).to(device)
    l_std_t = torch.from_numpy(stats["l_std"].astype(np.float32)).to(device)

    tuned_batch_size = _autotune_batch_size(
        config=config,
        model=model,
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        q_mat=q_mat,
        g_mat=g_mat,
        R_inv_Bp_T=R_inv_Bp_T,
        l_mean_t=l_mean_t,
        l_std_t=l_std_t,
    )

    train_loader, val_loader, _ = build_dataloaders(
        dataset_path=config.dataset_path,
        batch_size=tuned_batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
    )
    if tuned_batch_size != config.batch_size:
        print(f"[AutoBatch] use tuned batch_size={tuned_batch_size} (requested={config.batch_size})")

    best_val = float("inf")
    best_path = output_dir / "best_model.pt"
    patience_counter = 0

    history = {"train_total": [], "train_data": [], "train_phys": [], "train_ctrl": [],
               "val_total": [], "val_data": [], "val_phys": [], "val_ctrl": []}

    for epoch in range(1, config.epochs + 1):
        model.train()
        tr_total = tr_data = tr_phys = tr_ctrl = 0.0
        tr_count = 0

        for feat, l_norm, a_sdc, p_true_b, x_rel_b in train_loader:
            feat = feat.to(device, non_blocking=True)
            l_norm = l_norm.to(device, non_blocking=True)
            a_sdc = a_sdc.to(device, non_blocking=True)
            p_true_b = p_true_b.to(device, non_blocking=True)
            x_rel_b = x_rel_b.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if use_amp
                else contextlib.nullcontext()
            )
            with autocast_ctx:
                l_pred_norm = model(feat)
                l_pred = l_pred_norm * l_std_t + l_mean_t
                p_pred = reconstruct_spd_p(l_pred, delta=config.delta_spd)

                loss_data, loss_phys, loss_ctrl, loss_total = compute_losses(
                    l_pred_norm, l_norm, p_pred, p_true_b, x_rel_b,
                    a_sdc, q_mat, g_mat, R_inv_Bp_T,
                    config.lambda_phys, config.lambda_ctrl,
                    are_den_floor=config.are_den_floor,
                    ctrl_den_floor=config.ctrl_den_floor,
                    rel_loss_clip=config.rel_loss_clip,
                )

            if scaler.is_enabled():
                scaler.scale(loss_total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
                optimizer.step()

            bsz = feat.shape[0]
            tr_total += loss_total.item() * bsz
            tr_data += loss_data.item() * bsz
            tr_phys += loss_phys.item() * bsz
            tr_ctrl += loss_ctrl.item() * bsz
            tr_count += bsz

        model.eval()
        va_total = va_data = va_phys = va_ctrl = 0.0
        va_count = 0
        with torch.no_grad():
            for feat, l_norm, a_sdc, p_true_b, x_rel_b in val_loader:
                feat = feat.to(device, non_blocking=True)
                l_norm = l_norm.to(device, non_blocking=True)
                a_sdc = a_sdc.to(device, non_blocking=True)
                p_true_b = p_true_b.to(device, non_blocking=True)
                x_rel_b = x_rel_b.to(device, non_blocking=True)

                autocast_ctx = (
                    torch.autocast(device_type="cuda", dtype=amp_dtype)
                    if use_amp
                    else contextlib.nullcontext()
                )
                with autocast_ctx:
                    l_pred_norm = model(feat)
                    l_pred = l_pred_norm * l_std_t + l_mean_t
                    p_pred = reconstruct_spd_p(l_pred, delta=config.delta_spd)

                    ld, lp, lc, lt = compute_losses(
                        l_pred_norm, l_norm, p_pred, p_true_b, x_rel_b,
                        a_sdc, q_mat, g_mat, R_inv_Bp_T,
                        config.lambda_phys, config.lambda_ctrl,
                        are_den_floor=config.are_den_floor,
                        ctrl_den_floor=config.ctrl_den_floor,
                        rel_loss_clip=config.rel_loss_clip,
                    )
                bsz = feat.shape[0]
                va_total += lt.item() * bsz
                va_data += ld.item() * bsz
                va_phys += lp.item() * bsz
                va_ctrl += lc.item() * bsz
                va_count += bsz

        t_total = tr_total / max(1, tr_count)
        t_data = tr_data / max(1, tr_count)
        t_phys = tr_phys / max(1, tr_count)
        t_ctrl = tr_ctrl / max(1, tr_count)
        v_total = va_total / max(1, va_count)
        v_data = va_data / max(1, va_count)
        v_phys = va_phys / max(1, va_count)
        v_ctrl = va_ctrl / max(1, va_count)

        for k, v in [("train_total", t_total), ("train_data", t_data),
                      ("train_phys", t_phys), ("train_ctrl", t_ctrl),
                      ("val_total", v_total), ("val_data", v_data),
                      ("val_phys", v_phys), ("val_ctrl", v_ctrl)]:
            history[k].append(v)

        # 用验证损失驱动 LR 调度
        scheduler.step(v_total)

        if v_total < best_val:
            best_val = v_total
            patience_counter = 0
            torch.save({
                "model_state_dict": normalize_pinn_state_dict(model.state_dict()),
                "in_dim": in_dim,
                "backbone_dim": config.backbone_dim,
                "n_resblocks": config.n_resblocks,
                "head_hidden": config.head_hidden,
                "activation": config.activation,
                "batch_size": config.batch_size,
                "effective_batch_size": tuned_batch_size,
                "num_workers": config.num_workers,
                "enable_amp": config.enable_amp,
                "amp_dtype": config.amp_dtype,
                "enable_compile": config.enable_compile,
                "delta_spd": config.delta_spd,
                "feat_mean": stats["feat_mean"],
                "feat_std": stats["feat_std"],
                "l_mean": stats["l_mean"],
                "l_std": stats["l_std"],
                "q": stats["q"],
                "r": stats["r"],
                "gamma": stats["gamma"],
                "state_dim": STATE_DIM,
                "best_val_loss": best_val,
                "log_cholesky": True,
            }, best_path)
        else:
            patience_counter += 1

        if epoch % 50 == 0 or epoch == 1 or epoch == config.epochs:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(
                f"[Epoch {epoch:04d}] "
                f"train={t_total:.4e} (data={t_data:.4e} phys={t_phys:.4e} ctrl={t_ctrl:.4e}) | "
                f"val={v_total:.4e} (data={v_data:.4e} phys={v_phys:.4e} ctrl={v_ctrl:.4e}) | "
                f"lr={cur_lr:.2e} patience={patience_counter}/{config.early_stop_patience}"
            )

        if patience_counter >= config.early_stop_patience:
            print(f"[EarlyStopping] 在 epoch {epoch} 触发，最佳 val={best_val:.6f}")
            break

    np.savez_compressed(output_dir / "history.npz", **history)
    _plot_loss_curves(history, output_dir)
    summary = {"best_model_path": str(best_path), "best_val_loss": float(best_val)}
    print("3D PINN 训练完成:", summary)
    return summary


def _plot_loss_curves(history: dict, output_dir: Path) -> None:
    """训练结束后绘制 loss 曲线并保存。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_total"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, key, title in [
        (axes[0, 0], "total", "Total Loss"),
        (axes[0, 1], "data", "Data Loss (MSE)"),
        (axes[1, 0], "phys", "Physics Loss (ARE residual)"),
        (axes[1, 1], "ctrl", "Control Loss (L_u)"),
    ]:
        tk, vk = f"train_{key}", f"val_{key}"
        if tk in history and len(history[tk]) > 0:
            ax.semilogy(epochs, history[tk], label="Train", alpha=0.8)
            ax.semilogy(epochs, history[vk], label="Val", alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("3D PINN Training Loss Curves", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / "loss_curves.png", dpi=150)
    plt.close(fig)
    print(f"Loss curves saved to {output_dir / 'loss_curves.png'}")


def main() -> None:
    train_pinn(TrainConfig())


if __name__ == "__main__":
    main()
