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
    if array.size == 0 or not np.isfinite(array).any():
        return False
    return bool(np.nanmin(array) <= 0.0 <= np.nanmax(array))


def _zero_contours(phi: np.ndarray) -> list:
    if not _has_zero_level(phi):
        return []
    from skimage import measure

    array = np.asarray(phi, dtype=np.float64)
    contours = measure.find_contours(array, level=0.0)
    return [np.asarray(contour, dtype=np.float64) for contour in contours if contour.shape[0] >= 2]


def _contour_length(contour: np.ndarray) -> float:
    if contour.shape[0] < 2:
        return 0.0
    diffs = np.diff(contour, axis=0)
    return float(np.sum(np.sqrt(np.sum(diffs * diffs, axis=1))))


def _touches_y_border(contour: np.ndarray, height: int, border_margin: float) -> bool:
    rows = contour[:, 0]
    return bool(np.nanmin(rows) <= border_margin or np.nanmax(rows) >= float(height - 1) - border_margin)


def _is_closed_contour(contour: np.ndarray, tolerance: float = 2.0) -> bool:
    if contour.shape[0] < 2:
        return False
    return bool(np.linalg.norm(contour[0] - contour[-1]) <= tolerance)


def _filter_zero_contours(
    contours: Sequence[np.ndarray],
    height: int,
    min_points: int,
    border_margin: float,
) -> list:
    candidates = [contour for contour in contours if contour.shape[0] >= min_points]
    if not candidates:
        candidates = list(contours)
    without_y_border = [contour for contour in candidates if not _touches_y_border(contour, height, border_margin)]
    return without_y_border if without_y_border else candidates


def _select_main_zero_contour(
    contours: Sequence[np.ndarray],
    height: int,
    width: int,
    min_points: int,
    border_margin: float,
) -> list:
    candidates = _filter_zero_contours(contours, height, min_points, border_margin)
    if not candidates:
        return []

    def score(contour: np.ndarray) -> float:
        rows = contour[:, 0]
        cols = contour[:, 1]
        x_span = float(np.nanmax(cols) - np.nanmin(cols)) / max(float(width - 1), 1.0)
        y_span = float(np.nanmax(rows) - np.nanmin(rows)) / max(float(height - 1), 1.0)
        length = _contour_length(contour) / max(float(width), 1.0)
        closed_penalty = 0.75 if _is_closed_contour(contour) else 0.0
        y_border_penalty = 2.0 if _touches_y_border(contour, height, border_margin) else 0.0
        return 3.0 * x_span + 0.35 * length - 0.25 * y_span - closed_penalty - y_border_penalty

    return [max(candidates, key=score)]


def _select_contours_for_plot(
    phi: np.ndarray,
    mode: str,
    min_points: int,
    border_margin: float,
) -> list:
    contours = _zero_contours(phi)
    if not contours:
        return []
    height, width = np.asarray(phi).shape
    if mode == "all":
        return contours
    if mode == "filtered":
        return _filter_zero_contours(contours, height, min_points, border_margin)
    if mode == "main":
        return _select_main_zero_contour(contours, height, width, min_points, border_margin)
    raise ValueError(f"Unsupported contour_mode: {mode!r}; expected main, filtered, or all")


def _plot_zero_contours(
    axis,
    phi: np.ndarray,
    color: str,
    linewidth: float,
    linestyle: str,
    mode: str,
    min_points: int,
    border_margin: float,
) -> bool:
    contours = _select_contours_for_plot(phi, mode, min_points, border_margin)
    for contour in contours:
        axis.plot(contour[:, 1], contour[:, 0], color=color, linewidth=linewidth, linestyle=linestyle)
    return bool(contours)


def save_zero_contour_grid(
    predictions: Mapping[str, np.ndarray],
    output_path: str,
    state_order: Sequence[str],
    gt_arrays: Optional[Mapping[str, np.ndarray]] = None,
    title: Optional[str] = None,
    contour_mode: str = "main",
    min_contour_points: int = 25,
    border_margin: float = 2.0,
) -> None:
    """Save one large figure showing predicted phi=0 contours for each state.

    By default only the selected main contour is drawn.  This avoids rendering
    background frame contours, small closed holes, and isolated zero-level
    islands that make the plot hard to read.  Use contour_mode="filtered" or
    contour_mode="all" when debugging every zero-level component.
    """

    import math
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

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
        height, width = pred.shape
        image_for_colorbar = axis.imshow(pred, cmap="coolwarm", origin="upper", vmin=-max_abs, vmax=max_abs)
        pred_drawn = _plot_zero_contours(
            axis,
            pred,
            color="black",
            linewidth=1.8,
            linestyle="solid",
            mode=contour_mode,
            min_points=min_contour_points,
            border_margin=border_margin,
        )
        gt_drawn = False
        if state in gt_arrays:
            gt_drawn = _plot_zero_contours(
                axis,
                gt_arrays[state],
                color="red",
                linewidth=1.5,
                linestyle="dashed",
                mode=contour_mode,
                min_points=min_contour_points,
                border_margin=border_margin,
            )
        axis.set_title(state)
        axis.set_xlabel("x pixel")
        axis.set_ylabel("y pixel")
        axis.set_xlim(0, width - 1)
        axis.set_ylim(height - 1, 0)
        handles = []
        if pred_drawn:
            handles.append(Line2D([0], [0], color="black", linewidth=1.8, linestyle="solid", label=f"pred {contour_mode} phi=0"))
        if gt_drawn:
            handles.append(Line2D([0], [0], color="red", linewidth=1.5, linestyle="dashed", label=f"GT {contour_mode} phi=0"))
        if handles:
            axis.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.85)

    for axis in axes.flat[len(available):]:
        axis.set_visible(False)

    if image_for_colorbar is not None:
        fig.colorbar(image_for_colorbar, ax=axes, shrink=0.82, label="level-set phi")
    fig.savefig(path, dpi=180)
    plt.close(fig)
