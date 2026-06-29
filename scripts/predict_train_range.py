#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


TRAIN_RANGE_STEPS = [
    ("deposition", 1, "1M"),
    ("etch", 1, "1E"),
    ("deposition", 2, "2M"),
    ("etch", 2, "2E"),
]
TRAIN_RANGE_STATES = ["1M", "1E", "2M", "2E"]


def add_src_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict 1M through 2E from init and plot phi=0 contours.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--infer-missing-rates", action="store_true")
    parser.add_argument("--allow-baseline-fallback", action="store_true")
    parser.add_argument("--output-dir", default=None, help="Directory for 1M.npy ... 2E.npy")
    parser.add_argument("--figure", default=None, help="Output PNG path")
    parser.add_argument("--no-plot", action="store_true", help="Only save prediction npy files")
    parser.add_argument(
        "--contour-mode",
        choices=["main", "filtered", "all"],
        default="main",
        help="main draws one selected interface; filtered removes tiny/border components; all draws every phi=0 component",
    )
    parser.add_argument("--min-contour-points", type=int, default=25)
    parser.add_argument("--border-margin", type=float, default=2.0)
    args = parser.parse_args()

    import numpy as np

    root = add_src_path()
    from epi_pinn.config import (
        average_rate,
        device_name,
        load_config,
        output_dir,
        process_config,
        project_root_from_config_path,
        schedule_seconds,
    )
    from epi_pinn.contour import extract_contour20, save_contour_csv
    from epi_pinn.excel_io import load_state_arrays, write_prediction_workbook
    from epi_pinn.rollout import _infer_process_rate, _load_model, predict_next_levelset
    from epi_pinn.sdf import ensure_signed_distance
    from epi_pinn.train import torch_dtype
    from epi_pinn.visualize import save_zero_contour_grid

    config = load_config(args.config)
    config_root = project_root_from_config_path(args.config)
    states = load_state_arrays(config, base_dir=config_root)
    level_cfg = config.get("level_set", {})
    out_dir = output_dir(config, config_root)
    checkpoint_dir = out_dir / "checkpoints"
    prediction_dir = Path(args.output_dir) if args.output_dir else out_dir / "train_range_predictions"
    if not prediction_dir.is_absolute():
        prediction_dir = root / prediction_dir
    contour_dir = prediction_dir / "contours"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    contour_dir.mkdir(parents=True, exist_ok=True)

    device = device_name(config)
    dtype = torch_dtype(config.get("training", {}).get("dtype", "float64"))
    models = {
        "deposition": _load_model(config, "deposition", checkpoint_dir / "deposition_best.pt", device, dtype),
        "etch": _load_model(config, "etch", checkpoint_dir / "etch_best.pt", device, dtype),
    }
    inferred = {
        "deposition": _infer_process_rate(config, states, "deposition") if args.infer_missing_rates else None,
        "etch": _infer_process_rate(config, states, "etch") if args.infer_missing_rates else None,
    }

    phi = ensure_signed_distance(states["init"], level_cfg)
    predictions = {}
    for process_name, cycle, output_state in TRAIN_RANGE_STEPS:
        proc = process_config(config, process_name)
        process_sign = float(proc["sign"])
        duration = schedule_seconds(config, process_name, cycle)
        rate = average_rate(config, process_name, cycle, fallback=inferred[process_name])
        model = models[process_name]
        if model is None and not args.allow_baseline_fallback:
            raise FileNotFoundError(
                f"Missing checkpoint for {process_name}; expected "
                f"{checkpoint_dir / (process_name + '_best.pt')}. "
                "Train first or pass --allow-baseline-fallback."
            )
        phi = predict_next_levelset(phi, duration, rate, process_sign, config, model, device=device, dtype=dtype)
        predictions[output_state] = phi
        np.save(prediction_dir / f"{output_state}.npy", phi)
        contour = extract_contour20(
            phi,
            num_points=int(config.get("contour", {}).get("num_points", 20)),
            min_valid_points=int(config.get("contour", {}).get("min_valid_points", 10)),
        )
        save_contour_csv(contour, str(contour_dir / f"{output_state}_contour20.csv"))

    write_prediction_workbook(predictions, str(prediction_dir / "train_range_predictions.xlsx"))

    if not args.no_plot:
        figure = Path(args.figure) if args.figure else out_dir / "figures" / "train_range_zero_contours_1M_to_2E.png"
        if not figure.is_absolute():
            figure = root / figure
        gt_arrays = {
            state: ensure_signed_distance(states[state], level_cfg)
            for state in TRAIN_RANGE_STATES
            if state in states
        }
        save_zero_contour_grid(
            predictions,
            str(figure),
            TRAIN_RANGE_STATES,
            gt_arrays=gt_arrays,
            title="Train-range prediction phi=0 contours (GT overlay)",
            contour_mode=args.contour_mode,
            min_contour_points=args.min_contour_points,
            border_margin=args.border_margin,
        )
        print(f"Saved figure: {figure}")

    print(f"Saved train-range predictions: {prediction_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())