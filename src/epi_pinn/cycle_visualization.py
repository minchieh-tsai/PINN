"""Cycle contour visualization helpers with display-only smoothing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from epi_pinn.prediction_runtime import PredictionStepResult

from epi_pinn.sdf import gaussian_smooth_interface
from epi_pinn.visualize import _plot_zero_contours


@dataclass(frozen=True)
class GroundTruthOverlay:
    phi: np.ndarray
    label: str
    band_color: str = "#8a8f98"


def gt_band_mask(phi: np.ndarray, band_px: float) -> np.ndarray:
    """Return the inclusive signed-distance band around a GT zero contour."""
    width = float(band_px)
    if not np.isfinite(width) or width < 0.0:
        raise ValueError("gt_band_px must be finite and nonnegative")
    array = np.asarray(phi, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("GT level set must be a finite 2D array")
    return np.abs(array) <= width


def display_levelset(phi: np.ndarray, sigma: float) -> np.ndarray:
    """Build a smoothed display copy without mutating prediction artifacts."""
    return gaussian_smooth_interface(np.asarray(phi, dtype=np.float64).copy(), sigma)


def create_cycle_figure(
    steps: Sequence[PredictionStepResult],
    state_order: Sequence[str],
    gt_overlays: Optional[Mapping[str, GroundTruthOverlay]] = None,
    *,
    plot_gaussian_sigma: float = 0.0,
    gt_band_px: float = 3.0,
    contour_mode: str = "main",
    min_contour_points: int = 25,
    border_margin: float = 2.0,
    title: Optional[str] = None,
):
    """Create one row of cycle panels without altering inputs or metrics."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if contour_mode not in {"main", "filtered", "all"}:
        raise ValueError("contour_mode must be main, filtered, or all")
    if min_contour_points < 2:
        raise ValueError("min_contour_points must be at least 2")
    if not np.isfinite(border_margin) or border_margin < 0.0:
        raise ValueError("border_margin must be finite and nonnegative")
    by_state = {step.output_state: step for step in steps}
    missing = [state for state in state_order if state not in by_state]
    if missing:
        raise KeyError("Missing prediction steps: " + ", ".join(missing))
    overlays = gt_overlays or {}
    count = len(state_order)
    if count == 0:
        raise ValueError("state_order cannot be empty")

    fig, axes = plt.subplots(
        1,
        count,
        figsize=(3.45 * count, 6.0),
        squeeze=False,
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    axes_row = axes[0]
    if title:
        fig.suptitle(title, fontsize=14)

    has_overlay = False
    has_green_band = False
    reference_shape = None
    for axis, state in zip(axes_row, state_order):
        step = by_state[state]
        input_phi = display_levelset(step.input_phi, plot_gaussian_sigma)
        prediction_phi = display_levelset(step.prediction_phi, plot_gaussian_sigma)
        if reference_shape is None:
            reference_shape = prediction_phi.shape
        if input_phi.shape != prediction_phi.shape or prediction_phi.shape != reference_shape:
            raise ValueError("all cycle arrays must share one 2D shape")

        overlay = overlays.get(state)
        target_text = ""
        if overlay is not None:
            gt_phi = display_levelset(overlay.phi, plot_gaussian_sigma)
            if gt_phi.shape != prediction_phi.shape:
                raise ValueError(f"GT shape mismatch for {state}")
            band = gt_band_mask(gt_phi, gt_band_px)
            rgba = np.zeros((*band.shape, 4), dtype=np.float64)
            rgba[band] = to_rgba(overlay.band_color, alpha=0.22)
            axis.imshow(rgba, origin="upper", interpolation="nearest")
            _plot_zero_contours(
                axis, gt_phi, color="black", linewidth=1.35,
                linestyle="dashed", mode=contour_mode,
                min_points=min_contour_points, border_margin=border_margin,
            )
            has_overlay = True
            has_green_band = has_green_band or overlay.band_color == "#2f9e44"
            target_text = f"\nGT target: {overlay.label}"

        _plot_zero_contours(
            axis, input_phi, color="#4b4f56", linewidth=1.15,
            linestyle="solid", mode=contour_mode,
            min_points=min_contour_points, border_margin=border_margin,
        )
        _plot_zero_contours(
            axis, prediction_phi, color="#1864ab", linewidth=1.8,
            linestyle="solid", mode=contour_mode,
            min_points=min_contour_points, border_margin=border_margin,
        )
        height, width = prediction_phi.shape
        axis.set_xlim(0, width - 1)
        axis.set_ylim(height - 1, 0)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(
            f"{state} | {step.duration_s:.3g} s\ninput: {step.input_state}{target_text}",
            fontsize=9,
        )
        axis.set_xlabel("x pixel")
    axes_row[0].set_ylabel("y pixel")

    handles = [
        Line2D([0], [0], color="#4b4f56", linewidth=1.15, label="step input phi=0"),
        Line2D([0], [0], color="#1864ab", linewidth=1.8, label="PINN prediction phi=0"),
    ]
    if has_overlay:
        handles.extend(
            [
                Line2D([0], [0], color="black", linewidth=1.35, linestyle="dashed", label="GT phi=0"),
                Patch(facecolor="#8a8f98", alpha=0.22, label=f"GT +/-{gt_band_px:g} px band"),
            ]
        )
    if has_green_band:
        handles.append(Patch(facecolor="#2f9e44", alpha=0.22, label="5E target band"))
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(len(handles), 5),
        frameon=False,
    )
    return fig


def save_cycle_figure(
    steps: Sequence[PredictionStepResult],
    state_order: Sequence[str],
    gt_overlays: Mapping[str, GroundTruthOverlay],
    output_path: str | Path,
    *,
    dpi: int = 180,
    **kwargs,
) -> Path:
    """Create, save, and close a cycle figure."""
    import matplotlib.pyplot as plt

    if dpi <= 0:
        raise ValueError("dpi must be positive")
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = create_cycle_figure(steps, state_order, gt_overlays, **kwargs)
    try:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)
    return path