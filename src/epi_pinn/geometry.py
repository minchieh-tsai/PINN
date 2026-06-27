"""Grid geometry and interpolation utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def normalized_grid(height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-1.0, 1.0, width, dtype=np.float64)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float64)
    xi, eta = np.meshgrid(x, y)
    return xi, eta


def pixel_to_normalized_xy(x: np.ndarray, y: np.ndarray, height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    xi = 2.0 * np.asarray(x, dtype=np.float64) / float(width - 1) - 1.0
    eta = 2.0 * np.asarray(y, dtype=np.float64) / float(height - 1) - 1.0
    return xi, eta


def normalized_to_pixel_xy(xi: np.ndarray, eta: np.ndarray, height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    x = (np.asarray(xi, dtype=np.float64) + 1.0) * 0.5 * float(width - 1)
    y = (np.asarray(eta, dtype=np.float64) + 1.0) * 0.5 * float(height - 1)
    return x, y


def spatial_gradients(phi: np.ndarray, pixel_size_y: float = 1.0, pixel_size_x: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    grad_y, grad_x = np.gradient(np.asarray(phi, dtype=np.float64), pixel_size_y, pixel_size_x)
    return np.ascontiguousarray(grad_x), np.ascontiguousarray(grad_y)


def bilinear_sample(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    height, width = arr.shape
    x_clamped = np.clip(np.asarray(x, dtype=np.float64), 0.0, width - 1.0)
    y_clamped = np.clip(np.asarray(y, dtype=np.float64), 0.0, height - 1.0)

    x0 = np.floor(x_clamped).astype(np.int64)
    y0 = np.floor(y_clamped).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)

    wx = x_clamped - x0
    wy = y_clamped - y0
    top = (1.0 - wx) * arr[y0, x0] + wx * arr[y0, x1]
    bottom = (1.0 - wx) * arr[y1, x0] + wx * arr[y1, x1]
    return (1.0 - wy) * top + wy * bottom


def curvature(phi: np.ndarray, epsilon: float = 1.0e-6) -> np.ndarray:
    phi_x, phi_y = spatial_gradients(phi)
    phi_xx = np.gradient(phi_x, axis=1)
    phi_yy = np.gradient(phi_y, axis=0)
    phi_xy = np.gradient(phi_x, axis=0)
    denom = np.power(phi_x * phi_x + phi_y * phi_y + epsilon * epsilon, 1.5)
    kappa = (phi_xx * phi_y * phi_y - 2.0 * phi_xy * phi_x * phi_y + phi_yy * phi_x * phi_x) / denom
    return np.ascontiguousarray(kappa, dtype=np.float64)
