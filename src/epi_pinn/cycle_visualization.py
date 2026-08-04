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


def display_levelset(phi: np.ndarray, sigma: float) -> np.ndarray:
    """Build a smoothed display copy without mutating prediction artifacts."""
    array = np.asarray(phi, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("display level set must be a finite 2D array")
    return gaussian_smooth_interface(array.copy(), sigma)


def create_cycle_figure(
    steps: Sequence[PredictionStepResult],
    state_order: Sequence[str],
    gt_overlays: Optional[Mapping[str, GroundTruthOverlay]] = None,
    *,
    initial_phi: np.ndarray,
    initial_label: str = "init",
    plot_gaussian_sigma: float = 0.0,
    contour_mode: str = "main",
    min_contour_points: int = 25,
    border_margin: float = 2.0,
    panel_width: float = 4.8,
    figure_height: float = 8.5,
    title: Optional[str] = None,
):
    """Create cycle panels using one fixed initial boundary in every panel."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if contour_mode not in {"main", "filtered", "all"}:
        raise ValueError("contour_mode must be main, filtered, or all")
    if min_contour_points < 2:
        raise ValueError("min_contour_points must be at least 2")
    if not np.isfinite(border_margin) or border_margin < 0.0:
        raise ValueError("border_margin must be finite and nonnegative")
    if not np.isfinite(panel_width) or panel_width <= 0.0:
        raise ValueError("panel_width must be positive and finite")
    if not np.isfinite(figure_height) or figure_height <= 0.0:
        raise ValueError("figure_height must be positive and finite")

    by_state = {step.output_state: step for step in steps}
    missing = [state for state in state_order if state not in by_state]
    if missing:
        raise KeyError("Missing prediction steps: " + ", ".join(missing))
    overlays = gt_overlays or {}
    count = len(state_order)
    if count == 0:
        raise ValueError("state_order cannot be empty")

    initial_display = display_levelset(initial_phi, plot_gaussian_sigma)
    reference_shape = initial_display.shape
    fig, axes = plt.subplots(
        1,
        count,
        figsize=(panel_width * count, figure_height),
        squeeze=False,
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    axes_row = axes[0]
    if title:
        fig.suptitle(title, fontsize=16)

    has_overlay = False
    for axis, state in zip(axes_row, state_order):
        step = by_state[state]
        prediction_phi = display_levelset(step.prediction_phi, plot_gaussian_sigma)
        if prediction_phi.shape != reference_shape:
            raise ValueError(
                f"prediction shape mismatch for {state}: "
                f"{prediction_phi.shape} vs {reference_shape}"
            )

        _plot_zero_contours(
            axis,
            initial_display,
            color="#202124",
            linewidth=1.4,
            linestyle="solid",
            mode=contour_mode,
            min_points=min_contour_points,
            border_margin=border_margin,
        )
        _plot_zero_contours(
            axis,
            prediction_phi,
            color="#1864ab",
            linewidth=2.0,
            linestyle="solid",
            mode=contour_mode,
            min_points=min_contour_points,
            border_margin=border_margin,
        )

        overlay = overlays.get(state)
        target_text = ""
        if overlay is not None:
            gt_phi = display_levelset(overlay.phi, plot_gaussian_sigma)
            if gt_phi.shape != reference_shape:
                raise ValueError(
                    f"GT shape mismatch for {state}: "
                    f"{gt_phi.shape} vs {reference_shape}"
                )
            _plot_zero_contours(
                axis,
                gt_phi,
                color="#d9485f",
                linewidth=1.6,
                linestyle="dashed",
                mode=contour_mode,
                min_points=min_contour_points,
                border_margin=border_margin,
            )
            has_overlay = True
            target_text = f"\nGT target: {overlay.label}"

        height, width = prediction_phi.shape
        axis.set_xlim(0, width - 1)
        axis.set_ylim(height - 1, 0)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(
            f"{state}\nduration: {step.duration_s:.3g} s{target_text}",
            fontsize=10,
            pad=10,
        )
        axis.set_xlabel("x pixel")
    axes_row[0].set_ylabel("y pixel")

    handles = [
        Line2D(
            [0],
            [0],
            color="#202124",
            linewidth=1.4,
            label=f"initial boundary ({initial_label})",
        ),
        Line2D(
            [0],
            [0],
            color="#1864ab",
            linewidth=2.0,
            label="PINN prediction phi=0",
        ),
    ]
    if has_overlay:
        handles.append(
            Line2D(
                [0],
                [0],
                color="#d9485f",
                linewidth=1.6,
                linestyle="dashed",
                label="GT phi=0",
            )
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(handles),
        frameon=False,
        fontsize=10,
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