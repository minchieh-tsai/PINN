#!/usr/bin/env python
"""Render standard or CMA-selected cycle prediction contours."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


M_STATES_STANDARD = ("1M", "2M", "3M", "4M", "5M")
E_STATES_STANDARD = ("1E", "2E", "3E", "4E", "5E")
M_STATES_CMAES = ("1M", "2M", "3M", "4M")
E_STATES_CMAES = ("1E", "2E", "3E", "4E")


def add_src_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    return root


def _positive_sigma(value: Optional[float], label: str) -> Optional[float]:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return number


def _read_best_result(path: str) -> tuple[Mapping[str, Any], tuple[float, ...]]:
    payload = json.loads(Path(path).resolve().read_text(encoding="utf-8"))
    order = tuple(payload.get("time_order", ()))
    expected = ("1M", "1E", "2M", "2E", "3M", "3E", "4M", "4E")
    if order != expected:
        raise ValueError("best_result.json has an incompatible time_order")
    times_by_state = payload.get("best_times_s", {})
    times = tuple(float(times_by_state[state]) for state in expected)
    if any(not math.isfinite(value) or value <= 0.0 for value in times):
        raise ValueError("best_result.json contains an invalid process time")
    return payload, times


def _overlays(states, mapping):
    from epi_pinn.cycle_visualization import GroundTruthOverlay

    result = {}
    for prediction_state, target_state in mapping.items():
        if target_state not in states:
            continue
        result[prediction_state] = GroundTruthOverlay(
            phi=states[target_state],
            label=target_state,
        )
    return result


def _save_pair(
    output_dir: Path,
    prefix: str,
    steps,
    m_states,
    e_states,
    overlays,
    initial_phi,
    args,
    normalization_mode: str,
):
    from epi_pinn.cycle_visualization import save_cycle_figure

    common = {
        "initial_phi": initial_phi,
        "initial_label": "init",
        "plot_gaussian_sigma": args.plot_gaussian_sigma,
        "contour_mode": args.contour_mode,
        "min_contour_points": args.min_contour_points,
        "border_margin": args.border_margin,
        "panel_width": args.panel_width,
        "figure_height": args.figure_height,
        "dpi": args.dpi,
    }
    m_path = save_cycle_figure(
        steps,
        m_states,
        overlays,
        output_dir / f"{prefix}_M_{m_states[0]}_to_{m_states[-1]}.png",
        title=f"{prefix.upper()} deposition cycles | normalization: {normalization_mode}",
        **common,
    )
    e_path = save_cycle_figure(
        steps,
        e_states,
        overlays,
        output_dir / f"{prefix}_E_{e_states[0]}_to_{e_states[-1]}.png",
        title=f"{prefix.upper()} etch cycles | normalization: {normalization_mode}",
        **common,
    )
    return m_path, e_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize cycle-by-cycle PINN predictions.")
    parser.add_argument("--mode", choices=("standard", "cmaes"), required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--workbook", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--infer-missing-rates", action="store_true")
    parser.add_argument("--allow-baseline-fallback", action="store_true")
    parser.add_argument("--normalization-mode", choices=("legacy", "training-reference"), default=None)
    parser.add_argument("--prediction-gaussian-sigma", type=float, default=None)
    parser.add_argument("--plot-gaussian-sigma", type=float, default=0.0)
    parser.add_argument("--contour-mode", choices=("main", "filtered", "all"), default="main")
    parser.add_argument("--min-contour-points", type=int, default=25)
    parser.add_argument("--border-margin", type=float, default=2.0)
    parser.add_argument("--panel-width", type=float, default=4.8)
    parser.add_argument("--figure-height", type=float, default=8.5)
    parser.add_argument("--rollout-start", choices=("gt-2e", "predicted-2e"), default="gt-2e")
    parser.add_argument("--best-result", default=None)
    parser.add_argument("--times", nargs=8, type=float, metavar="SECONDS")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args(argv)

    try:
        _positive_sigma(args.prediction_gaussian_sigma, "prediction_gaussian_sigma")
        _positive_sigma(args.plot_gaussian_sigma, "plot_gaussian_sigma")
        _positive_sigma(args.border_margin, "border_margin")
    except ValueError as exc:
        parser.error(str(exc))
    if args.min_contour_points < 2 or args.dpi <= 0:
        parser.error("--min-contour-points must be >= 2 and --dpi must be positive")
    if not math.isfinite(args.panel_width) or args.panel_width <= 0.0:
        parser.error("--panel-width must be positive and finite")
    if not math.isfinite(args.figure_height) or args.figure_height <= 0.0:
        parser.error("--figure-height must be positive and finite")
    if args.times is not None and any(not math.isfinite(value) or value <= 0.0 for value in args.times):
        parser.error("all --times values must be positive and finite")
    if args.mode == "cmaes" and (bool(args.best_result) == bool(args.times)):
        parser.error("CMA-ES mode requires exactly one of --best-result or --times")
    if args.mode == "standard" and (args.best_result or args.times):
        parser.error("--best-result and --times are only valid in CMA-ES mode")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    add_src_path()

    from epi_pinn.config import load_config, output_dir, project_root_from_config_path, schedule_seconds
    from epi_pinn.prediction_runtime import (
        PREDICT_1M_TO_4E_STEPS,
        ROLLOUT_2E_TO_5E_STEPS,
        load_prediction_runtime,
        normalization_metadata,
        run_prediction_sequence,
    )

    best_payload = None
    best_source = None
    if args.mode == "cmaes" and args.best_result:
        best_payload, times = _read_best_result(args.best_result)
        best_source = str(Path(args.best_result).resolve())
        saved_mode = str(best_payload.get("normalization", {}).get("mode", "legacy"))
        if args.normalization_mode is not None and args.normalization_mode != saved_mode:
            raise ValueError(
                f"Requested normalization {args.normalization_mode!r} does not match "
                f"best result mode {saved_mode!r}"
            )
        normalization_mode = saved_mode
        saved_sigma = float(best_payload.get("prediction_gaussian_sigma", 0.0))
        prediction_sigma = (
            saved_sigma if args.prediction_gaussian_sigma is None
            else float(args.prediction_gaussian_sigma)
        )
    else:
        normalization_mode = args.normalization_mode or "legacy"
        prediction_sigma = float(args.prediction_gaussian_sigma or 0.0)
        times = tuple(args.times) if args.times is not None else None

    runtime = load_prediction_runtime(
        args.config,
        workbook_path=args.workbook,
        checkpoint_dir=args.checkpoint_dir,
        infer_missing_rates=args.infer_missing_rates,
        allow_baseline_fallback=args.allow_baseline_fallback,
        normalization_mode=normalization_mode,
    )
    config = load_config(args.config)
    config_root = project_root_from_config_path(args.config)
    artifact_root = output_dir(config, config_root)
    destination = (
        Path(args.output_dir).resolve()
        if args.output_dir else artifact_root / "cycle_visualization" / normalization_mode
    )
    destination.mkdir(parents=True, exist_ok=True)

    if args.mode == "standard":
        train_steps = PREDICT_1M_TO_4E_STEPS[:4]
        train_times = tuple(
            schedule_seconds(config, process, cycle)
            for process, cycle, _state in train_steps
        )
        train_result = run_prediction_sequence(
            runtime,
            runtime.states["init"],
            train_steps,
            train_times,
            prediction_gaussian_sigma=prediction_sigma,
            start_state="init",
        )
        if args.rollout_start == "gt-2e":
            rollout_start_phi = runtime.states["2E"]
            rollout_start_state = "GT 2E"
        else:
            rollout_start_phi = train_result.predictions["2E"]
            rollout_start_state = "predicted 2E"
        rollout_times = tuple(
            schedule_seconds(config, process, cycle)
            for process, cycle, _state in ROLLOUT_2E_TO_5E_STEPS
        )
        rollout_result = run_prediction_sequence(
            runtime,
            rollout_start_phi,
            ROLLOUT_2E_TO_5E_STEPS,
            rollout_times,
            prediction_gaussian_sigma=prediction_sigma,
            start_state=rollout_start_state,
        )
        steps = [*train_result.steps, *rollout_result.steps]
        overlays = _overlays(
            runtime.states,
            {"1M": "1M", "1E": "1E", "2M": "2M", "2E": "2E", "5M": "5M", "5E": "5E"},
        )
        figure_paths = _save_pair(
            destination,
            "standard",
            steps,
            M_STATES_STANDARD,
            E_STATES_STANDARD,
            overlays,
            runtime.states["init"],
            args,
            normalization_mode,
        )
        input_source = {"rollout_start": args.rollout_start}
        manifest_name = "standard_manifest.json"
    else:
        if times is None:
            raise AssertionError("CMA-ES times were not resolved")
        result = run_prediction_sequence(
            runtime,
            runtime.states["init"],
            PREDICT_1M_TO_4E_STEPS,
            times,
            prediction_gaussian_sigma=prediction_sigma,
            start_state="init",
        )
        steps = result.steps
        overlays = _overlays(
            runtime.states,
            {"1M": "1M", "1E": "1E", "2M": "2M", "2E": "2E", "4M": "5M", "4E": "5E"},
        )
        figure_paths = _save_pair(
            destination,
            "cmaes",
            steps,
            M_STATES_CMAES,
            E_STATES_CMAES,
            overlays,
            runtime.states["init"],
            args,
            normalization_mode,
        )
        input_source = {
            "best_result": best_source,
            "explicit_times": list(times) if best_source is None else None,
        }
        manifest_name = "cmaes_manifest.json"

    manifest = {
        "mode": args.mode,
        "normalization": normalization_metadata(runtime),
        "prediction_gaussian_sigma": prediction_sigma,
        "plot_gaussian_sigma": args.plot_gaussian_sigma,
        "prediction_gaussian_filter": {
            "called_before_each_inference": True,
            "smoothing_active": prediction_sigma > 0.0,
        },
        "initial_boundary": "init",
        "panel_width": args.panel_width,
        "figure_height": args.figure_height,
        "contour_mode": args.contour_mode,
        "input_source": input_source,
        "figures": [str(path) for path in figure_paths],
        "steps": [
            {
                "state": step.output_state,
                "input_state": step.input_state,
                "process": step.process_name,
                "cycle": step.cycle,
                "duration_s": step.duration_s,
                "average_rate": step.average_rate,
            }
            for step in steps
        ],
    }
    manifest_path = destination / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if prediction_sigma > 0.0:
        print(
            f"Prediction Gaussian filter: sigma={prediction_sigma:g}, "
            f"applied before all {len(steps)} inference steps."
        )
    else:
        print(
            "Prediction Gaussian filter: sigma=0; called before every inference "
            "step as an identity copy (no smoothing)."
        )
    print(f"Saved cycle figures: {figure_paths[0]}, {figure_paths[1]}")
    print(f"Saved manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())