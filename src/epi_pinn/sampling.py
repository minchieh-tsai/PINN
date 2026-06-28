"""Sampling and feature construction for the conditional PINN."""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import numpy as np

from epi_pinn.contour import ContourCondition, interpolate_contour_eta
from epi_pinn.geometry import (
    bilinear_sample,
    curvature,
    normalized_grid,
    normalized_to_pixel_xy,
    pixel_to_normalized_xy,
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


def build_collocation_pools(
    phi_initial: np.ndarray,
    contour: ContourCondition,
    narrow_band_distance: float,
) -> Dict[str, np.ndarray]:
    phi0 = np.asarray(phi_initial, dtype=np.float64)
    height, width = phi0.shape
    n_total = height * width
    global_pool = np.arange(n_total, dtype=np.int64)
    interface_pool = np.flatnonzero(np.abs(phi0.reshape(-1)) <= narrow_band_distance).astype(np.int64)
    if interface_pool.size == 0:
        interface_pool = global_pool

    valid = contour.valid_mask > 0.0
    contour_indices = []
    if valid.any():
        x_pixels, y_pixels = normalized_to_pixel_xy(contour.points_xy[valid, 0], contour.points_xy[valid, 1], height, width)
        radius = max(1, int(round(narrow_band_distance)))
        x_radius = min(2, radius)
        for x_value, y_value in zip(x_pixels, y_pixels):
            x0 = int(round(float(x_value)))
            y0 = int(round(float(y_value)))
            for yy in range(max(0, y0 - radius), min(height, y0 + radius + 1)):
                for xx in range(max(0, x0 - x_radius), min(width, x0 + x_radius + 1)):
                    contour_indices.append(yy * width + xx)
    contour_pool = np.unique(np.asarray(contour_indices, dtype=np.int64)) if contour_indices else interface_pool
    return {"interface": interface_pool, "contour": contour_pool, "global": global_pool}


def sample_collocation_indices(
    pools: Mapping[str, np.ndarray],
    batch_size: int,
    interface_fraction: float,
    contour_fraction: float,
    global_fraction: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    n_interface = int(round(batch_size * max(interface_fraction, 0.0)))
    n_contour = int(round(batch_size * max(contour_fraction, 0.0)))
    requested_global = int(round(batch_size * max(global_fraction, 0.0)))
    n_global = max(0, batch_size - n_interface - n_contour)
    if requested_global > n_global and n_interface + n_contour + requested_global <= batch_size:
        n_global = requested_global

    def choose(pool_name: str, size: int) -> np.ndarray:
        pool = np.asarray(pools.get(pool_name, pools["global"]), dtype=np.int64)
        if pool.size == 0:
            pool = np.asarray(pools["global"], dtype=np.int64)
        return rng.choice(pool, size=size, replace=pool.size < size) if size > 0 else np.empty(0, dtype=np.int64)

    indices = np.concatenate(
        [
            choose("interface", n_interface),
            choose("contour", n_contour),
            choose("global", n_global),
        ]
    ).astype(np.int64)
    if indices.size < batch_size:
        indices = np.concatenate([indices, choose("global", batch_size - indices.size)])
    rng.shuffle(indices)
    tau = rng.uniform(1.0e-4, 1.0 - 1.0e-4, size=indices.size).astype(np.float64)
    return indices, tau


def contour_tensor_features(contour: ContourCondition) -> np.ndarray:
    return np.ascontiguousarray(contour.as_features(), dtype=np.float64)


def transition_key(config: Mapping[str, object], process_name: str) -> str:
    return "deposition_train" if process_name == "deposition" else "etch_train"