"""Event metrics for safe terminal approach studies."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TerminalMetrics:
    """Distance, speed, and residence metrics evaluated on a full trajectory."""

    first_entry_time_s: float | None
    first_entry_speed_km_s: float | None
    first_entry_closing_speed_km_s: float | None
    first_soft_entry_time_s: float | None
    first_sustained_soft_time_s: float | None
    entry_count: int
    exit_count: int
    minimum_distance_km: float
    maximum_distance_after_first_entry_km: float | None
    dwell_fraction_after_first_entry: float | None


def compute_terminal_metrics(
    t_s: np.ndarray,
    relative_position_km: np.ndarray,
    relative_velocity_km_s: np.ndarray,
    *,
    capture_distance_km: float,
    soft_speed_km_s: float,
    dwell_time_s: float,
) -> TerminalMetrics:
    """Compute discrete first-entry, soft-entry, and residence metrics.

    The trajectory must span the full common experiment horizon.  Residence is
    evaluated at the simulation sampling times: every sample from the proposed
    start through the dwell horizon must meet the soft-arrival condition.
    """
    t_s = np.asarray(t_s, dtype=float)
    position = np.asarray(relative_position_km, dtype=float)
    velocity = np.asarray(relative_velocity_km_s, dtype=float)

    if t_s.ndim != 1 or t_s.size == 0:
        raise ValueError("t_s must be a nonempty one-dimensional array.")
    if position.shape != (t_s.size, 3) or velocity.shape != (t_s.size, 3):
        raise ValueError("position and velocity must have shape (len(t_s), 3).")
    if np.any(np.diff(t_s) <= 0.0):
        raise ValueError("t_s must be strictly increasing.")
    if capture_distance_km <= 0.0 or soft_speed_km_s <= 0.0 or dwell_time_s < 0.0:
        raise ValueError("terminal thresholds must be positive and dwell time nonnegative.")

    distance = np.linalg.norm(position, axis=1)
    speed = np.linalg.norm(velocity, axis=1)
    radial_speed = np.sum(position * velocity, axis=1) / np.maximum(distance, 1e-12)
    closing_speed = np.maximum(-radial_speed, 0.0)

    inside = distance <= capture_distance_km
    soft = inside & (speed <= soft_speed_km_s)

    entry_count = int(inside[0]) + int(np.count_nonzero(~inside[:-1] & inside[1:]))
    exit_count = int(np.count_nonzero(inside[:-1] & ~inside[1:]))

    entry_indices = np.flatnonzero(inside)
    first_entry_index = int(entry_indices[0]) if entry_indices.size else None
    soft_indices = np.flatnonzero(soft)
    first_soft_index = int(soft_indices[0]) if soft_indices.size else None

    first_sustained_index: int | None = None
    for start_index in soft_indices:
        dwell_end_s = t_s[start_index] + dwell_time_s
        end_index = int(np.searchsorted(t_s, dwell_end_s, side="right") - 1)
        if end_index >= start_index and t_s[end_index] >= dwell_end_s:
            if np.all(soft[start_index:end_index + 1]):
                first_sustained_index = int(start_index)
                break

    if first_entry_index is None:
        first_entry_time_s = None
        first_entry_speed_km_s = None
        first_entry_closing_speed_km_s = None
        maximum_distance_after_first_entry_km = None
        dwell_fraction_after_first_entry = None
    else:
        first_entry_time_s = float(t_s[first_entry_index])
        first_entry_speed_km_s = float(speed[first_entry_index])
        first_entry_closing_speed_km_s = float(closing_speed[first_entry_index])
        maximum_distance_after_first_entry_km = float(np.max(distance[first_entry_index:]))
        dwell_fraction_after_first_entry = float(np.mean(soft[first_entry_index:]))

    return TerminalMetrics(
        first_entry_time_s=first_entry_time_s,
        first_entry_speed_km_s=first_entry_speed_km_s,
        first_entry_closing_speed_km_s=first_entry_closing_speed_km_s,
        first_soft_entry_time_s=(None if first_soft_index is None else float(t_s[first_soft_index])),
        first_sustained_soft_time_s=(
            None if first_sustained_index is None else float(t_s[first_sustained_index])
        ),
        entry_count=entry_count,
        exit_count=exit_count,
        minimum_distance_km=float(np.min(distance)),
        maximum_distance_after_first_entry_km=maximum_distance_after_first_entry_km,
        dwell_fraction_after_first_entry=dwell_fraction_after_first_entry,
    )
