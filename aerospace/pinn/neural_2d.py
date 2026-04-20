"""向后兼容重导出 — 实际实现已迁移至 aerospace.control.neural_2d。"""

from aerospace.control.neural_2d import (  # noqa: F401
    NeuralSDREController2D,
    SimBenchmarkResult2D,
    run_closed_loop_benchmark,
)
