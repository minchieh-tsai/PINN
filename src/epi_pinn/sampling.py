"""Sampling and feature construction for the conditional PINN."""

from __future__ import annotations

from typing import Mapping, Tuple

import numpy as np

from epi_pinn.contour import ContourCondition, interpolate_contour_eta
from epi_pinn.geometry import (
    bilinear_sample,
    curvature,
    normalized_grid,
    normalized_to_pixel_xy,
    spatial_gradients,
)


FEATURE_NAMES = [
    "xi",
    "eta",
    "tau",
    "duration_normalized",
    "rate_normalized",
    "nominal_displacement_normalized",
    "sampled_phi0",
    "sampled_phi0_x",
    "sampled_phi0_y",
    "sampled_normal0_x",
    "sampled_normal0_y",
    "sampled_kappa0",
    "interpolated_contour_eta",
    "relative_eta_to_contour",
]


def full_grid_query(height: int, width: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xi, eta = normalized_grid(height, width)
    x, y = normalized_to_pixel_xy(xi.reshape(-1), eta.reshape(-1), height, width)
    return xi.reshape(-1), eta.reshape(-1), x, y


def build_features(
    phi_initial: np.ndarray,
    contour: ContourCondition,
    xi: np.ndarray,
    eta: np.ndarray,
    tau: np.ndarray,
    duration_s: float,
    average_rate: float,
    duration_reference_s: float,
    rate_reference: float,
    clip_distance: float,
    process_sign: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if duration_s <= 0:
        raise ValueError(f"duration_s must be positive, got {duration_s}")
    if average_rate <= 0:
        raise ValueError(f"average_rate must be positive, got {average_rate}")
    if duration_reference_s <= 0:
        raise ValueError(f"duration_reference_s must be positive, got {duration_reference_s}")
    if rate_reference <= 0:
        raise ValueError(f"rate_reference must be positive, got {rate_reference}")
    if clip_distance <= 0:
        raise ValueError(f"clip_distance must be positive, got {clip_distance}")

    phi0 = np.asarray(phi_initial, dtype=np.float64)
    height, width = phi0.shape
    xi_arr = np.asarray(xi, dtype=np.float64).reshape(-1)
    eta_arr = np.asarray(eta, dtype=np.float64).reshape(-1)
    tau_arr = np.asarray(tau, dtype=np.float64).reshape(-1)
    if tau_arr.size == 1:
        tau_arr = np.full_like(xi_arr, float(tau_arr[0]))
    if not (xi_arr.size == eta_arr.size == tau_arr.size):
        raise ValueError("xi, eta, and tau must have matching lengths")

    x, y = normalized_to_pixel_xy(xi_arr, eta_arr, height, width)
    phi0_x, phi0_y = spatial_gradients(phi0)
    grad_norm = np.sqrt(phi0_x * phi0_x + phi0_y * phi0_y + 1.0e-12)
    normal_x = phi0_x / grad_norm
    normal_y = phi0_y / grad_norm
    kappa = curvature(phi0)

    sampled_phi0 = bilinear_sample(phi0, x, y)
    sampled_phi0_x = bilinear_sample(phi0_x, x, y)
    sampled_phi0_y = bilinear_sample(phi0_y, x, y)
    sampled_normal_x = bilinear_sample(normal_x, x, y)
    sampled_normal_y = bilinear_sample(normal_y, x, y)
    sampled_kappa = bilinear_sample(kappa, x, y)
    contour_eta, contour_mask = interpolate_contour_eta(contour, xi_arr)
    relative_eta = (eta_arr - contour_eta) * contour_mask

    duration_norm = np.full_like(xi_arr, duration_s / duration_reference_s)
    rate_norm = np.full_like(xi_arr, average_rate / rate_reference)
    nominal_disp = np.full_like(
        xi_arr,
        process_sign * average_rate * duration_s / clip_distance,
    )

    features = np.stack(
        [
            xi_arr,
            eta_arr,
            tau_arr,
            duration_norm,
            rate_norm,
            nominal_disp,
            np.clip(sampled_phi0 / clip_distance, -1.0, 1.0),
            sampled_phi0_x,
            sampled_phi0_y,
            sampled_normal_x,
            sampled_normal_y,
            sampled_kappa,
            contour_eta,
            relative_eta,
        ],
        axis=1,
    )
    return np.ascontiguousarray(features, dtype=np.float64), np.ascontiguousarray(sampled_phi0, dtype=np.float64)


def sample_endpoint_indices(
    phi_target: np.ndarray,
    batch_size: int,
    interface_fraction: float,
    narrow_band_distance: float,
    rng: np.random.Generator,
) -> np.ndarray:
    flat_target = np.asarray(phi_target, dtype=np.float64).reshape(-1)
    n_total = flat_target.size
    n_interface = int(round(batch_size * interface_fraction))
    n_global = max(0, batch_size - n_interface)

    interface_pool = np.flatnonzero(np.abs(flat_target) <= narrow_band_distance)
    if interface_pool.size == 0:
        interface_pool = np.arange(n_total)

    chosen = []
    if n_interface > 0:
        chosen.append(rng.choice(interface_pool, size=n_interface, replace=interface_pool.size < n_interface))
    if n_global > 0:
        chosen.append(rng.integers(0, n_total, size=n_global))
    return np.concatenate(chosen).astype(np.int64)


def contour_tensor_features(contour: ContourCondition) -> np.ndarray:
    return np.ascontiguousarray(contour.as_features(), dtype=np.float64)


def transition_key(config: Mapping[str, object], process_name: str) -> str:
    return "deposition_train" if process_name == "deposition" else "etch_train"
