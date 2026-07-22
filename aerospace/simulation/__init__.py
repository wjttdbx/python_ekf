"""闭环仿真引擎：将"动力学 + 控制 + 导航"组合为可运行的追逃博弈仿真。"""

from aerospace.simulation.nerm_ekf_sdre import EKFSDRESimulation, EKFSDRESimResult
from aerospace.simulation.metrics import compute_metrics, CaptureMetrics

__all__ = [
    "EKFSDRESimulation",
    "EKFSDRESimResult",
    "compute_metrics",
    "CaptureMetrics",
]
