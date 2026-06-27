"""Signed-distance field preprocessing."""

from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy import ndimage


def rebuild_sdf_from_mask(mask_inside: np.ndarray) -> np.ndarray:
    """Build an SDF using negative values inside the material mask."""

    inside = np.asarray(mask_inside, dtype=bool)
    outside = ~inside
    distance_outside = ndimage.distance_transform_edt(outside)
    distance_inside = ndimage.distance_transform_edt(inside)
    return np.ascontiguousarray(distance_outside - distance_inside, dtype=np.float64)


def ensure_signed_distance(phi: np.ndarray, level_set_config: Mapping[str, object]) -> np.ndarray:
    input_kind = str(level_set_config.get("input_kind", "signed_distance"))
    rebuild = bool(level_set_config.get("rebuild_sdf", False))
    array = np.asarray(phi, dtype=np.float64)

    if rebuild or input_kind == "mask":
        return rebuild_sdf_from_mask(array < 0)
    if input_kind in ("signed_distance", "level_set"):
        return np.ascontiguousarray(array, dtype=np.float64)
    raise ValueError(f"Unsupported level_set.input_kind: {input_kind!r}")


def clip_and_normalize(phi: np.ndarray, clip_distance: float) -> np.ndarray:
    if clip_distance <= 0:
        raise ValueError(f"clip_distance must be positive, got {clip_distance}")
    clipped = np.clip(np.asarray(phi, dtype=np.float64), -clip_distance, clip_distance)
    return np.ascontiguousarray(clipped / clip_distance, dtype=np.float64)


def denormalize_phi(phi_tilde: np.ndarray, clip_distance: float) -> np.ndarray:
    if clip_distance <= 0:
        raise ValueError(f"clip_distance must be positive, got {clip_distance}")
    return np.ascontiguousarray(np.asarray(phi_tilde, dtype=np.float64) * clip_distance)


def material_mask(phi: np.ndarray) -> np.ndarray:
    return np.asarray(phi) < 0
