"""Known-average-rate baseline and rate estimation."""

from __future__ import annotations

from typing import Optional

import numpy as np


def known_average_rate_baseline(
    phi_initial: np.ndarray,
    duration_s: float,
    average_rate: float,
    process_sign: float,
) -> np.ndarray:
    if duration_s <= 0:
        raise ValueError(f"duration_s must be positive, got {duration_s}")
    if average_rate <= 0:
        raise ValueError(f"average_rate must be positive, got {average_rate}")
    return np.ascontiguousarray(
        np.asarray(phi_initial, dtype=np.float64) - float(process_sign) * average_rate * duration_s,
        dtype=np.float64,
    )


def estimate_average_rate_from_pair(
    phi_initial: np.ndarray,
    phi_target: np.ndarray,
    duration_s: float,
    process_sign: float,
    narrow_band_distance: Optional[float] = None,
) -> float:
    if duration_s <= 0:
        raise ValueError(f"duration_s must be positive, got {duration_s}")
    initial = np.asarray(phi_initial, dtype=np.float64)
    target = np.asarray(phi_target, dtype=np.float64)
    if initial.shape != target.shape:
        raise ValueError(f"Shape mismatch: initial {initial.shape}, target {target.shape}")
    if narrow_band_distance is None:
        mask = np.ones(initial.shape, dtype=bool)
    else:
        mask = np.abs(target) <= float(narrow_band_distance)
        if not mask.any():
            mask = np.ones(initial.shape, dtype=bool)
    displacement = (initial[mask] - target[mask]) / float(process_sign)
    rate = float(np.median(displacement) / duration_s)
    if rate <= 0 or not np.isfinite(rate):
        rate = float(np.mean(np.abs(target[mask] - initial[mask])) / duration_s)
    if rate <= 0 or not np.isfinite(rate):
        raise ValueError("Could not infer a positive finite average rate from the state pair")
    return rate
