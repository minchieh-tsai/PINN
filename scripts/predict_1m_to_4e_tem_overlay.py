#!/usr/bin/env python
"""Predict 1M through 4E and overlay final contours on 5M/5E data."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROLLOUT_STEPS = [
    ("deposition", 1, "1M"),
    ("etch", 1, "1E"),
    ("deposition", 2, "2M"),
    ("etch", 2, "2E"),
    ("deposition", 3, "3M"),
    ("etch", 3, "3E"),
    ("deposition", 4, "4M"),
    ("etch", 4, "4E"),
]
COMPARISONS = {"4M": "5M", "4E": "5E"}


def add_src_path():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))


def resolve_path(path_text, base_dir):
    path = Path(path_text)
    return path if path.is_absolute() else base_dir / path


def save_metrics(predictions, states, config, output_dir):
    from epi_pinn.evaluate import evaluate_pair

    results = {
        f"{pred_state}_vs_{gt_state}": evaluate_pair(
            predictions[pred_state], states[gt_state], config
        )
        for pred_state, gt_state in COMPARISONS.items()
    }
    with (output_dir / "comparison_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    metric_names = sorted({key for values in results.values() for key in values})
    with (output_dir / "comparison_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["comparison"] + metric_names)
        writer.writeheader()
        for comparison, values in results.items():
            writer.writerow({"comparison": comparison, **values})
    return results


def load_tem(path):
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"TEM image does not exist: {path}")
    import matplotlib.pyplot as plt

    return plt.imread(str(path))


def save_overlay(
    predictions,
    states,
    tem_paths,
    output_path,
    contour_mode,
    min_contour_points,
    border_margin,
):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    from epi_pinn.visualize import _plot_zero_contours

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 7.0), constrained_layout=True)
    fig.suptitle("Eight-step init-to-4E prediction over current 5M/5E TEM", fontsize=16)

    for axis, pred_state in zip(axes, ("4M", "4E")):
        gt_state = COMPARISONS[pred_state]
        prediction = np.asarray(predictions[pred_state], dtype=np.float64)
        gt = np.asarray(states[gt_state], dtype=np.float64)
        height, width = prediction.shape
        tem = load_tem(tem_paths[gt_state])

        if tem is None:
            max_abs = float(np.nanmax(np.abs(gt)))
            max_abs = max_abs if np.isfinite(max_abs) and max_abs > 0.0 else 1.0
            axis.imshow(
                gt,
                cmap="coolwarm",
                origin="upper",
                vmin=-max_abs,
                vmax=max_abs,
                extent=(0, width - 1, height - 1, 0),
            )
            background_label = f"{gt_state} GT level-set"
        else:
            axis.imshow(
                tem,
                cmap="gray" if tem.ndim == 2 else None,
                origin="upper",
                extent=(0, width - 1, height - 1, 0),
            )
            background_label = f"{gt_state} TEM"

        pred_drawn = _plot_zero_contours(
            axis,
            prediction,
            color="black",
            linewidth=2.2,
            linestyle="solid",
            mode=contour_mode,
            min_points=min_contour_points,
            border_margin=border_margin,
        )
        gt_drawn = _plot_zero_contours(
            axis,
            gt,
            color="#ef4444",
            linewidth=2.0,
            linestyle="dashed",
            mode=contour_mode,
            min_points=min_contour_points,
            border_margin=border_margin,
        )
        axis.set_title(f"Pred {pred_state} and GT {gt_state}\non {background_label}")
        axis.set_xlabel("x pixel")
        axis.set_ylabel("y pixel")
        axis.set_xlim(0, width - 1)
        axis.set_ylim(height - 1, 0)

        handles = []
        if pred_drawn:
            handles.append(
                Line2D([0], [0], color="black", linewidth=2.2, label=f"Pred {pred_state} phi=0")
            )
        if gt_drawn:
            handles.append(
                Line2D(
                    [0], [0], color="#ef4444", linewidth=2.0,
                    linestyle="dashed", label=f"GT {gt_state} phi=0"
                )
            )
        if handles:
            axis.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.88)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Predict init -> 1M -> 1E -> ... -> 4M -> 4E using eight input "
            "durations, then overlay predicted 4M/4E and GT 5M/5E contours."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--workbook",
        default=None,
        help="Optional workbook with exact sheets init, 1M, 1E, 2M, 2E, 5M, 5E",
    )
    parser.add_argument("--infer-missing-rates", action="store_true")
    parser.add_argument("--allow-baseline-fallback", action="store_true")
    parser.add_argument("--tem-5m", default=None, help="Optional 5M TEM image path")
    parser.add_argument("--tem-5e", default=None, help="Optional 5E TEM image path")
    parser.add_argument(
        "--times",
        nargs=8,
        type=float,
        metavar="SECONDS",
        default=None,
        help=(
            "Eight durations ordered as 1M 1E 2M 2E 3M 3E 4M 4E; "
            "defaults to config schedule"
        ),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--figure", default=None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--contour-mode", choices=["main", "filtered", "all"], default="main")
    parser.add_argument("--min-contour-points", type=int, default=25)
    parser.add_argument("--border-margin", type=float, default=2.0)
    args = parser.parse_args()
    if args.times is not None and any(value <= 0.0 for value in args.times):
        parser.error("every value passed to --times must be positive")

    import numpy as np

    add_src_path()
    from epi_pinn.config import (
        average_rate, device_name, load_config, output_dir, process_config,
        project_root_from_config_path, schedule_seconds,
    )
    from epi_pinn.contour import extract_contour20, save_contour_csv
    from epi_pinn.excel_io import (
        load_state_arrays,
        read_excel_array,
        write_prediction_workbook,
    )
    from epi_pinn.rollout import _infer_process_rate, _load_model, predict_next_levelset
    from epi_pinn.sdf import ensure_signed_distance, rebuild_sdf_from_mask
    from epi_pinn.train import torch_dtype

    config = load_config(args.config)
    config_root = project_root_from_config_path(args.config)
    workbook_path = resolve_path(args.workbook, config_root) if args.workbook else None
    if workbook_path is None:
        raw_states = load_state_arrays(config, base_dir=config_root)
    else:
        data_cfg = config.get("data", {})
        expected_shape = (
            int(data_cfg.get("expected_height", 350)),
            int(data_cfg.get("expected_width", 200)),
        )
        allow_transpose = bool(data_cfg.get("allow_transpose", True))
        raw_states = {
            state: read_excel_array(
                str(workbook_path),
                state,
                expected_shape=expected_shape,
                allow_transpose=allow_transpose,
            )
            for state in ("init", "1M", "1E", "2M", "2E", "5M", "5E")
        }
    required_states = ("init", "5M", "5E")
    missing = [state for state in required_states if state not in raw_states]
    if missing:
        raise KeyError(
            "Configured data.state_sources is missing required states: " + ", ".join(missing)
        )

    level_cfg = config.get("level_set", {})
    states = {
        state: ensure_signed_distance(raw_states[state], level_cfg)
        for state in required_states
    }
    base_output_dir = output_dir(config, config_root)
    prediction_dir = (
        resolve_path(args.output_dir, config_root)
        if args.output_dir
        else base_output_dir / "prediction_1m_to_4e_tem_overlay"
    )
    contour_dir = prediction_dir / "contours"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    contour_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = base_output_dir / "checkpoints"
    device = device_name(config)
    dtype = torch_dtype(config.get("training", {}).get("dtype", "float64"))
    models = {
        process_name: _load_model(
            config, process_name, checkpoint_dir / f"{process_name}_best.pt", device, dtype
        )
        for process_name in ("deposition", "etch")
    }
    for process_name, model in models.items():
        if model is None and not args.allow_baseline_fallback:
            expected = checkpoint_dir / f"{process_name}_best.pt"
            raise FileNotFoundError(
                f"Missing checkpoint for {process_name}: {expected}. "
                "Train first or pass --allow-baseline-fallback."
            )

    inferred_rates = {
        process_name: (
            _infer_process_rate(config, raw_states, process_name)
            if args.infer_missing_rates else None
        )
        for process_name in ("deposition", "etch")
    }
    reinitialize_sdf = bool(
        config.get("rollout", {}).get("reinitialize_sdf_each_step", True)
    )

    durations = (
        list(args.times)
        if args.times is not None
        else [
            schedule_seconds(config, process_name, cycle)
            for process_name, cycle, _output_state in ROLLOUT_STEPS
        ]
    )
    phi = states["init"]
    predictions = {}
    manifest_steps = []
    for (process_name, cycle, output_state), duration in zip(
        ROLLOUT_STEPS, durations
    ):
        rate = average_rate(
            config, process_name, cycle, fallback=inferred_rates[process_name]
        )
        process_sign = float(process_config(config, process_name)["sign"])
        phi = predict_next_levelset(
            phi, duration, rate, process_sign, config, models[process_name],
            device=device, dtype=dtype,
        )
        if reinitialize_sdf:
            phi = rebuild_sdf_from_mask(phi < 0.0)

        predictions[output_state] = phi
        np.save(prediction_dir / f"{output_state}.npy", phi)
        contour = extract_contour20(
            phi,
            num_points=int(config.get("contour", {}).get("num_points", 20)),
            min_valid_points=int(config.get("contour", {}).get("min_valid_points", 10)),
        )
        save_contour_csv(contour, str(contour_dir / f"{output_state}_contour20.csv"))
        manifest_steps.append({
            "process": process_name,
            "cycle": cycle,
            "output_state": output_state,
            "duration_s": duration,
            "average_rate": rate,
        })

    write_prediction_workbook(
        predictions, str(prediction_dir / "predictions_1m_to_4e.xlsx")
    )
    metrics = save_metrics(predictions, states, config, prediction_dir)

    tem_paths = {
        "5M": resolve_path(args.tem_5m, config_root) if args.tem_5m else None,
        "5E": resolve_path(args.tem_5e, config_root) if args.tem_5e else None,
    }
    figure = (
        resolve_path(args.figure, config_root)
        if args.figure else prediction_dir / "4M_4E_on_5M_5E_TEM.png"
    )
    if not args.no_plot:
        save_overlay(
            predictions, states, tem_paths, figure,
            contour_mode=args.contour_mode,
            min_contour_points=args.min_contour_points,
            border_margin=args.border_margin,
        )

    manifest = {
        \
        "start_state": "init",
        "rollout_steps": manifest_steps,
        "iteration_times_s": {
            output_state: duration
            for (_process, _cycle, output_state), duration in zip(
                ROLLOUT_STEPS, durations
            )
        },
        "comparison_mapping": COMPARISONS,
        "tem_images": {
            state: str(path.resolve()) if path is not None else None
            for state, path in tem_paths.items()
        },
        "reinitialize_sdf_each_step": reinitialize_sdf,
        "figure": str(figure.resolve()) if not args.no_plot else None,
    }
    with (prediction_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("Saved prediction states: 1M, 1E, 2M, 2E, 3M, 3E, 4M, 4E")
    print(f"Saved predictions: {prediction_dir}")
    if not args.no_plot:
        print(f"Saved TEM overlay: {figure}")
    for comparison, values in metrics.items():
        print(
            f"{comparison}: zero_contour_symmetric_chamfer_px="
            f"{values['zero_contour_symmetric_chamfer_px']:.6g}, "
            f"contour20_y_mae_px={values['contour20_y_mae_px']:.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())