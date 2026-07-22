"""Regression tests for terminal-approach event metrics."""

import numpy as np

from types import SimpleNamespace

from aerospace.simulation.metrics import compute_metrics
from aerospace.simulation.terminal_metrics import compute_terminal_metrics


def test_metrics_distinguish_first_entry_from_sustained_soft_arrival():
    """A high-speed first entry must not be confused with later soft residence."""
    t = np.arange(0.0, 80.0, 10.0)
    distance = np.array([0.20, 0.09, 0.11, 0.09, 0.09, 0.09, 0.09, 0.09])
    position = np.column_stack((distance, np.zeros_like(distance), np.zeros_like(distance)))
    speed = np.array([0.20, 0.20, 0.20, 0.05, 0.05, 0.05, 0.05, 0.05])
    velocity = np.column_stack((-speed, np.zeros_like(speed), np.zeros_like(speed)))

    metrics = compute_terminal_metrics(
        t,
        position,
        velocity,
        capture_distance_km=0.1,
        soft_speed_km_s=0.1,
        dwell_time_s=30.0,
    )

    assert metrics.first_entry_time_s == 10.0
    assert metrics.first_entry_speed_km_s == 0.20
    assert metrics.first_soft_entry_time_s == 30.0
    assert metrics.first_sustained_soft_time_s == 30.0
    assert metrics.entry_count == 2
    assert metrics.exit_count == 1
    assert metrics.dwell_fraction_after_first_entry == 5 / 7


def _formal_metrics_result(t: np.ndarray):
    n = len(t)
    states = np.zeros((13, n))
    states[0, :] = 0.05
    return SimpleNamespace(
        t=t,
        states=states,
        dist_history=np.full(n, 0.05),
        u_p_history=np.zeros((3, n)),
        ekf_err_history=np.zeros((6, n)),
        x_est_history=np.zeros((6, n)),
        innov_history=np.zeros((2, n)),
        nis_history=np.full(n, np.nan),
        nees_history=np.full(n, np.nan),
    )


def test_formal_metrics_uses_elapsed_time_for_sustained_arrival():
    result_590 = _formal_metrics_result(np.arange(60, dtype=float) * 10.0)
    metrics_590 = compute_metrics(
        result_590, dt=10.0, capture_dist=0.1,
        v_thresh=1e-4, sustain_dur=600.0,
        compute_consistency=False,
    )
    assert metrics_590.sustained is False

    result_600 = _formal_metrics_result(np.arange(61, dtype=float) * 10.0)
    metrics_600 = compute_metrics(
        result_600, dt=10.0, capture_dist=0.1,
        v_thresh=1e-4, sustain_dur=600.0,
        compute_consistency=False,
    )
    assert metrics_600.sustained is True
    assert metrics_600.T_sustain == 0.0

def test_consistency_metrics_are_reported_without_terminal_entry():
    result = _formal_metrics_result(np.arange(3, dtype=float) * 10.0)
    result.dist_history[:] = 1.0
    result.states[0, :] = 1.0
    result.nis_history = np.array([np.nan, 1.0, 3.0])
    result.nees_history = np.array([np.nan, 4.0, 8.0])

    metrics = compute_metrics(
        result, dt=10.0, capture_dist=0.1,
        v_thresh=1e-4, sustain_dur=600.0,
    )

    assert metrics.T_FP is None
    assert metrics.nis_mean == 2.0
    assert metrics.nees_mean == 6.0