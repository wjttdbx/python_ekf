"""
统一路径：仓库根目录为 `aerospace` 包的上一级目录。
"""
from __future__ import annotations

from pathlib import Path

_PKG = Path(__file__).resolve().parent
ROOT: Path = _PKG.parent

FIGURES_DIR: Path = ROOT / "outputs" / "figures"
DATA_DIR: Path = ROOT / "data"
CHECKPOINTS_DIR: Path = ROOT / "checkpoints"


def ensure_figures_dir() -> Path:
    """确保图片输出目录存在。"""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR
