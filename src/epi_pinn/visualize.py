"""Plotting helpers for prediction artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np


def save_levelset_triplet(phi_true: np.ndarray, phi_pred: np.ndarray, output_path: str, title: Optional[str] = None) -> None:
    import matplotlib.pyplot as plt

    true = np.asarray(phi_true, dtype=np.float64)
    pred = np.asarray(phi_pred, dtype=np.float64)
    error = np.abs(pred - true)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    if title:
        fig.suptitle(title)
    for axis, data, label in zip(axes, [true, pred, error], ["ground truth", "prediction", "absolute error"]):
        image = axis.imshow(data, cmap="coolwarm", origin="upper")
        axis.set_title(label)
        axis.set_axis_off()
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _has_zero_level(phi: np.ndarray) -> bool:
    array = np.asarray(phi, dtype=np.float64)
    return bool(np.nanmin(array) <= 0.0 <= np.nanmax(array))


def _plot_zero_contour(axis, phi: np.ndarray, color: str, linewidth: float, linestyle: str, label: str) -> bool:
    array = np.asarray(phi, dtype=np.float64)
    if not _has_zero_level(array):
        return False
    contour = axis.contour(array, levels=[0.0], colors=[color], linewidths=linewidth, linestyles=linestyle)
    if contour.collections:
        contour.collections[0].set_label(label)
    return True


def save_zero_contour_grid(
    predictions: Mapping[str, np.ndarray],
    output_path: str,
    state_order: Sequence[str],
    gt_arrays: Optional[Mapping[str, np.ndarray]] = None,
    title: Optional[str] = None,
) -> None:
    """Save one large figure showing predicted phi=0 contours for each state.

    Prediction contours are black solid lines.  Ground-truth contours, when
    supplied for a state, are red dashed lines.
    """

    import math
    import matplotlib.pyplot as plt

    gt_arrays = gt_arrays or {}
    available = [state for state in state_order if state in predictions]
    if not available:
        raise ValueError("No requested prediction states were available to plot")

    cols = min(3, len(available))
    rows = int(math.ceil(len(available) / cols))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    all_values = [np.asarray(predictions[state], dtype=np.float64) for state in available]
    max_abs = max(float(np.nanmax(np.abs(array))) for array in all_values)
    if not np.isfinite(max_abs) or max_abs <= 0.0:
        max_abs = 1.0

    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 5.2 * rows), squeeze=False, constrained_layout=True)
    if title:
        fig.suptitle(title, fontsize=16)

    image_for_colorbar = None
    for axis, state in zip(axes.flat, available):
        pred = np.asarray(predictions[state], dtype=np.float64)
        image_for_colorbar = axis.imshow(pred, cmap="coolwarm", origin="upper", vmin=-max_abs, vmax=max_abs)
        pred_drawn = _plot_zero_contour(axis, pred, color="black", linewidth=1.8, linestyle="solid", label="pred phi=0")
        gt_drawn = False
        if state in gt_arrays:
            gt_drawn = _plot_zero_contour(axis, gt_arrays[state], color="red", linewidth=1.5, linestyle="dashed", label="GT phi=0")
        axis.set_title(state)
        axis.set_xlabel("x pixel")
        axis.set_ylabel("y pixel")
        labels = []
        if pred_drawn:
            labels.append("pred phi=0")
        if gt_drawn:
            labels.append("GT phi=0")
        if labels:
            axis.legend(loc="upper right", fontsize=8, framealpha=0.85)

    for axis in axes.flat[len(available):]:
        axis.set_visible(False)

    if image_for_colorbar is not None:
        fig.colorbar(image_for_colorbar, ax=axes, shrink=0.82, label="level-set phi")
    fig.savefig(path, dpi=180)
    plt.close(fig)
