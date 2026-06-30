"""Training loss visualization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LossComponent:
    name: str
    log_column: str
    config_key: str
    label: str


LOSS_COMPONENTS = (
    LossComponent("sdf", "sdf_loss", "sdf", "SDF"),
    LossComponent("dice", "dice_loss", "dice", "Dice"),
    LossComponent("pde", "pde_loss", "pde", "PDE"),
    LossComponent("eikonal", "eikonal_loss", "eikonal", "Eikonal"),
    LossComponent("sign", "sign_loss", "sign", "Sign"),
    LossComponent("velocity_jacobian", "velocity_jacobian_loss", "velocity_jacobian", "Velocity Jacobian"),
    LossComponent("curvature_velocity", "curvature_velocity_loss", "curvature_velocity", "Curvature Velocity"),
)


LOSS_COLORS = {
    "sdf": "#3b82f6",
    "dice": "#10b981",
    "pde": "#ef4444",
    "eikonal": "#8b5cf6",
    "sign": "#f59e0b",
    "velocity_jacobian": "#14b8a6",
    "curvature_velocity": "#ec4899",
}


def available_loss_components(frame: pd.DataFrame) -> Sequence[LossComponent]:
    return [component for component in LOSS_COMPONENTS if component.log_column in frame.columns]


def raw_loss_components(frame: pd.DataFrame) -> pd.DataFrame:
    raw: Dict[str, pd.Series] = {}
    for component in available_loss_components(frame):
        raw[component.name] = pd.to_numeric(frame[component.log_column], errors="coerce").fillna(0.0)
    return pd.DataFrame(raw, index=frame.index, dtype=float)


def weighted_loss_components(frame: pd.DataFrame, loss_config: Mapping[str, object]) -> pd.DataFrame:
    weighted: Dict[str, pd.Series] = {}
    for component in available_loss_components(frame):
        weight = float(loss_config.get(component.config_key, 0.0))
        values = pd.to_numeric(frame[component.log_column], errors="coerce").fillna(0.0)
        weighted[component.name] = values * weight
    return pd.DataFrame(weighted, index=frame.index, dtype=float)


def loss_component_fractions(weighted: pd.DataFrame) -> pd.DataFrame:
    if weighted.empty:
        return weighted.copy()
    totals = weighted.sum(axis=1)
    safe_totals = totals.where(totals != 0.0, np.nan)
    return weighted.div(safe_totals, axis=0).fillna(0.0)


def _component_labels(columns: Iterable[str]) -> list[str]:
    labels = {component.name: component.label for component in LOSS_COMPONENTS}
    return [labels.get(column, column) for column in columns]


def _component_colors(columns: Iterable[str]) -> list[str]:
    return [LOSS_COLORS.get(column, "#64748b") for column in columns]


def _read_training_log(log_path: Path) -> pd.DataFrame:
    if not log_path.exists():
        raise FileNotFoundError(f"Missing training log: {log_path}")
    frame = pd.read_csv(log_path)
    if "step" not in frame.columns:
        raise ValueError(f"Training log {log_path} is missing required column: step")
    return frame


def _plot_curve_group(ax, steps: pd.Series, values: pd.DataFrame, title: str, ylabel: str) -> None:
    labels = _component_labels(values.columns)
    colors = _component_colors(values.columns)
    for column, label, color in zip(values.columns, labels, colors):
        ax.plot(steps, values[column], label=label, color=color, linewidth=1.7)
    ax.set_title(title)
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.set_yscale("symlog", linthresh=1.0e-8)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)


def _plot_process_losses(
    ax_raw,
    ax_weighted,
    ax_fraction,
    process_name: str,
    frame: pd.DataFrame,
    loss_config: Mapping[str, object],
) -> None:
    raw = raw_loss_components(frame)
    weighted = weighted_loss_components(frame, loss_config)
    if raw.empty or weighted.empty:
        raise ValueError(f"No known loss columns found for process {process_name}")

    fractions = loss_component_fractions(weighted)
    steps = pd.to_numeric(frame["step"], errors="coerce")

    _plot_curve_group(ax_raw, steps, raw, f"{process_name} raw losses", "Raw loss")
    _plot_curve_group(ax_weighted, steps, weighted, f"{process_name} weighted losses", "Weighted loss")
    if "loss" in frame.columns:
        ax_weighted.plot(steps, frame["loss"], label="Total", color="#111827", linewidth=1.6, linestyle="--")
        ax_weighted.legend(loc="best", fontsize=8)

    labels = _component_labels(weighted.columns)
    colors = _component_colors(weighted.columns)
    ax_fraction.stackplot(
        steps,
        [fractions[column].to_numpy(dtype=float) for column in weighted.columns],
        labels=labels,
        colors=colors,
        alpha=0.88,
    )
    ax_fraction.set_title(f"{process_name} weighted loss share")
    ax_fraction.set_xlabel("Training step")
    ax_fraction.set_ylabel("Share of weighted sum")
    ax_fraction.set_ylim(0.0, 1.0)
    ax_fraction.grid(True, alpha=0.25)
    ax_fraction.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)


def plot_training_losses(
    config_path: str,
    output_path: Optional[str] = None,
    processes: Optional[Sequence[str]] = None,
    log_dir: Optional[str] = None,
) -> Path:
    from epi_pinn.config import load_config, output_dir, project_root_from_config_path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = load_config(config_path)
    root = project_root_from_config_path(config_path)
    out_dir = output_dir(config, root)
    logs_dir = Path(log_dir) if log_dir is not None else out_dir / "logs"
    if not logs_dir.is_absolute():
        logs_dir = root / logs_dir
    figure_path = Path(output_path) if output_path is not None else out_dir / "figures" / "training_loss_breakdown.png"
    if not figure_path.is_absolute():
        figure_path = root / figure_path

    selected_processes = list(processes) if processes else ["deposition", "etch"]
    process_frames = []
    for process_name in selected_processes:
        log_path = logs_dir / f"{process_name}_training.csv"
        if log_path.exists():
            process_frames.append((process_name, _read_training_log(log_path)))
    if not process_frames:
        expected = ", ".join(str(logs_dir / f"{name}_training.csv") for name in selected_processes)
        raise FileNotFoundError(f"No training logs found. Expected one of: {expected}")

    fig, axes = plt.subplots(
        nrows=len(process_frames),
        ncols=3,
        figsize=(20.0, max(4.8, 4.4 * len(process_frames))),
        squeeze=False,
        constrained_layout=True,
    )
    loss_config = config.get("loss", {})
    for row, (process_name, frame) in enumerate(process_frames):
        _plot_process_losses(axes[row][0], axes[row][1], axes[row][2], process_name, frame, loss_config)

    fig.suptitle("Training Loss Breakdown", fontsize=15, fontweight="bold")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return figure_path
