#!/usr/bin/env python
"""Optimize init-to-4E process durations with CMA-ES.

This entry point intentionally does not call the repository's existing
``predict_next_levelset`` helper.  It supplies the fixed duration and rate
references used during training, so candidate process times are represented
consistently in the network features without changing existing inference code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple


ROLLOUT_STEPS: Tuple[Tuple[str, int, str], ...] = (
    ("deposition", 1, "1M"),
    ("etch", 1, "1E"),
    ("deposition", 2, "2M"),
    ("etch", 2, "2E"),
    ("deposition", 3, "3M"),
    ("etch", 3, "3E"),
    ("deposition", 4, "4M"),
    ("etch", 4, "4E"),
)
STEP_LABELS: Tuple[str, ...] = tuple(step[2] for step in ROLLOUT_STEPS)
WORKBOOK_STATES: Tuple[str, ...] = ("init", "1M", "1E", "2M", "2E", "5M", "5E")
OBJECTIVE_NAME = (
    "zero_contour_symmetric_chamfer_px(4M,5M)"
    "+zero_contour_symmetric_chamfer_px(4E,5E)"
)


def add_src_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return root


def resolve_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else base_dir / path


def times_from_multipliers(
    base_times: Sequence[float],
    multipliers: Sequence[float],
    lower_scale: float,
    upper_scale: float,
) -> Tuple[float, ...]:
    if len(base_times) != len(ROLLOUT_STEPS) or len(multipliers) != len(ROLLOUT_STEPS):
        raise ValueError("base_times and multipliers must each contain eight values")
    if not 0.0 < lower_scale < upper_scale:
        raise ValueError("time scale bounds must satisfy 0 < lower < upper")

    result = []
    for label, base, multiplier in zip(STEP_LABELS, base_times, multipliers):
        base_value = float(base)
        scale = float(multiplier)
        if not math.isfinite(base_value) or base_value <= 0.0:
            raise ValueError(f"base time for {label} must be positive and finite")
        if not math.isfinite(scale) or not lower_scale <= scale <= upper_scale:
            raise ValueError(
                f"time multiplier for {label} must be within [{lower_scale}, {upper_scale}]"
            )
        result.append(base_value * scale)
    return tuple(result)


def duration_reference_for(config: Mapping[str, Any], process_name: str) -> float:
    processes = config.get("processes", {})
    if process_name not in processes:
        raise KeyError(f"Missing process config for {process_name}")
    value = float(processes[process_name].get("duration_reference_s", 0.0))
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"duration_reference_s for {process_name} must be positive")
    return value


def fixed_rate_reference_for(
    config: Mapping[str, Any],
    process_name: str,
    process_rates: Sequence[float],
) -> float:
    process_config = config.get("processes", {}).get(process_name, {})
    configured = process_config.get("rate_reference")
    if configured is not None:
        value = float(configured)
    else:
        finite_rates = sorted(float(rate) for rate in process_rates if math.isfinite(float(rate)))
        if not finite_rates:
            raise ValueError(f"No finite rate is available for {process_name}")
        midpoint = len(finite_rates) // 2
        if len(finite_rates) % 2:
            value = finite_rates[midpoint]
        else:
            value = 0.5 * (finite_rates[midpoint - 1] + finite_rates[midpoint])
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"rate reference for {process_name} must be positive")
    return value


def combined_objective(score_4m: float, score_4e: float, invalid_value: float) -> float:
    values = (float(score_4m), float(score_4e))
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        return float(invalid_value)
    return values[0] + values[1]


def validate_target_sources(config: Mapping[str, Any], explicit_workbook: bool) -> None:
    if explicit_workbook:
        return
    state_sources = config.get("data", {}).get("state_sources", {})
    missing = [state for state in ("init", "5M", "5E") if state not in state_sources]
    if missing:
        raise KeyError("Missing configured state sources: " + ", ".join(missing))
    source_5m = state_sources["5M"]
    source_5e = state_sources["5E"]
    identity_5m = (str(source_5m.get("workbook")), str(source_5m.get("sheet")))
    identity_5e = (str(source_5e.get("workbook")), str(source_5e.get("sheet")))
    if identity_5m == identity_5e:
        raise ValueError(
            "5M and 5E resolve to the same configured workbook/sheet. "
            "Pass --workbook with distinct 5M and 5E sheets or fix state_sources."
        )


def build_features_with_fixed_references(
    build_features_fn: Callable[..., Any],
    phi_initial: Any,
    contour: Any,
    xi: Any,
    eta: Any,
    tau: Any,
    duration_s: float,
    average_rate: float,
    duration_reference_s: float,
    rate_reference: float,
    clip_distance: float,
    process_sign: float,
) -> Any:
    return build_features_fn(
        phi_initial,
        contour,
        xi,
        eta,
        tau,
        duration_s,
        average_rate,
        duration_reference_s,
        rate_reference,
        clip_distance,
        process_sign,
    )


@dataclass(frozen=True)
class CandidateEvaluation:
    multipliers: Tuple[float, ...]
    times_s: Tuple[float, ...]
    score_4m: float
    score_4e: float
    objective: float
    status: str
    error: str
    elapsed_s: float


@dataclass
class OptimizationRuntime:
    config: Mapping[str, Any]
    config_path: Path
    workbook_path: Optional[Path]
    states: Mapping[str, Any]
    raw_states: Mapping[str, Any]
    models: Mapping[str, Any]
    rates: Tuple[float, ...]
    duration_references: Mapping[str, float]
    rate_references: Mapping[str, float]
    target_points: Mapping[str, Any]
    device: str
    dtype: Any
    reinitialize_sdf: bool
    smoothing_sigma_px: float
    allow_baseline_fallback: bool
    checkpoint_paths: Mapping[str, Path]
    xi: Any
    eta: Any
    tau: Any


def evaluate_candidate(
    multipliers: Sequence[float],
    base_times: Sequence[float],
    lower_scale: float,
    upper_scale: float,
    invalid_objective: float,
    rollout_fn: Callable[[Sequence[float]], Any],
    score_fn: Callable[[Any], Tuple[float, float]],
) -> CandidateEvaluation:
    start = time.perf_counter()
    multiplier_tuple = tuple(float(value) for value in multipliers)
    try:
        times_s = times_from_multipliers(
            base_times, multiplier_tuple, lower_scale, upper_scale
        )
        predictions = rollout_fn(times_s)
        score_4m, score_4e = score_fn(predictions)
        objective = combined_objective(score_4m, score_4e, invalid_objective)
        if objective >= invalid_objective:
            raise ValueError("candidate produced a non-finite or invalid contour score")
        status = "ok"
        error = ""
    except Exception as exc:
        times_s = tuple(
            float(base) * float(scale)
            for base, scale in zip(base_times, multiplier_tuple)
        )
        score_4m = float("nan")
        score_4e = float("nan")
        objective = float(invalid_objective)
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
    return CandidateEvaluation(
        multipliers=multiplier_tuple,
        times_s=times_s,
        score_4m=float(score_4m),
        score_4e=float(score_4e),
        objective=float(objective),
        status=status,
        error=error,
        elapsed_s=time.perf_counter() - start,
    )


def _load_raw_states(
    config: Mapping[str, Any],
    config_root: Path,
    workbook_path: Optional[Path],
) -> Mapping[str, Any]:
    from epi_pinn.excel_io import load_state_arrays, read_excel_array

    validate_target_sources(config, explicit_workbook=workbook_path is not None)
    if workbook_path is None:
        return load_state_arrays(config, base_dir=config_root)

    data_config = config.get("data", {})
    expected_shape = (
        int(data_config.get("expected_height", 350)),
        int(data_config.get("expected_width", 200)),
    )
    allow_transpose = bool(data_config.get("allow_transpose", True))
    return {
        state: read_excel_array(
            str(workbook_path),
            state,
            expected_shape=expected_shape,
            allow_transpose=allow_transpose,
        )
        for state in WORKBOOK_STATES
    }


def prepare_runtime(args: argparse.Namespace) -> OptimizationRuntime:
    import numpy as np

    add_src_path()
    from epi_pinn.config import (
        average_rate,
        device_name,
        load_config,
        output_dir,
        process_config,
        project_root_from_config_path,
    )
    from epi_pinn.evaluate import _zero_contour_points
    from epi_pinn.rollout import _infer_process_rate, _load_model
    from epi_pinn.sampling import full_grid_query
    from epi_pinn.sdf import ensure_signed_distance
    from epi_pinn.train import torch_dtype

    config = load_config(args.config)
    config_root = project_root_from_config_path(args.config)
    config_path = Path(args.config).resolve()
    workbook_path = (
        resolve_path(args.workbook, config_root).resolve() if args.workbook else None
    )
    raw_states = _load_raw_states(config, config_root, workbook_path)
    missing = [state for state in WORKBOOK_STATES if state not in raw_states]
    if missing:
        raise KeyError("Missing required states: " + ", ".join(missing))

    level_config = config.get("level_set", {})
    states = {
        state: ensure_signed_distance(raw_states[state], level_config)
        for state in ("init", "5M", "5E")
    }
    if states["5M"].shape != states["5E"].shape:
        raise ValueError("5M and 5E target arrays must have the same shape")
    if np.array_equal(states["5M"], states["5E"]):
        raise ValueError("5M and 5E target arrays are identical; objective targets must differ")

    target_points = {
        "5M": _zero_contour_points(states["5M"]),
        "5E": _zero_contour_points(states["5E"]),
    }
    for state, points in target_points.items():
        if points.size == 0:
            raise ValueError(f"Target {state} does not contain a zero contour")

    base_output_dir = output_dir(config, config_root)
    checkpoint_dir = base_output_dir / "checkpoints"
    checkpoint_paths = {
        process: checkpoint_dir / f"{process}_best.pt"
        for process in ("deposition", "etch")
    }
    device = device_name(config)
    dtype = torch_dtype(config.get("training", {}).get("dtype", "float64"))
    models = {
        process: _load_model(
            config, process, checkpoint_paths[process], device, dtype
        )
        for process in ("deposition", "etch")
    }
    for process, model in models.items():
        if model is None and not args.allow_baseline_fallback:
            raise FileNotFoundError(
                f"Missing checkpoint for {process}: {checkpoint_paths[process]}"
            )

    inferred_rates = {
        process: (
            _infer_process_rate(config, raw_states, process)
            if args.infer_missing_rates
            else None
        )
        for process in ("deposition", "etch")
    }
    rates = tuple(
        average_rate(config, process, cycle, fallback=inferred_rates[process])
        for process, cycle, _state in ROLLOUT_STEPS
    )
    rates_by_process = {
        process: [
            rate
            for rate, (step_process, _cycle, _state) in zip(rates, ROLLOUT_STEPS)
            if step_process == process
        ]
        for process in ("deposition", "etch")
    }
    duration_references = {
        process: duration_reference_for(config, process)
        for process in ("deposition", "etch")
    }
    rate_references = {
        process: fixed_rate_reference_for(config, process, rates_by_process[process])
        for process in ("deposition", "etch")
    }

    rollout_config = config.get("rollout", {})
    reinitialize_sdf = bool(rollout_config.get("reinitialize_sdf_each_step", True))
    smoothing_sigma_px = float(rollout_config.get("interface_smoothing_sigma_px", 0.0))
    if smoothing_sigma_px != 0.0 and not 0.5 <= smoothing_sigma_px <= 1.0:
        raise ValueError("interface_smoothing_sigma_px must be 0 or within [0.5, 1.0]")
    if smoothing_sigma_px > 0.0 and not reinitialize_sdf:
        raise ValueError("interface smoothing requires SDF reinitialization")

    for process in ("deposition", "etch"):
        process_config(config, process)

    height, width = states["init"].shape
    xi, eta, _x, _y = full_grid_query(height, width)
    tau = np.ones_like(xi)

    return OptimizationRuntime(
        config=config,
        config_path=config_path,
        workbook_path=workbook_path,
        states=states,
        raw_states=raw_states,
        models=models,
        rates=rates,
        duration_references=duration_references,
        rate_references=rate_references,
        target_points=target_points,
        device=device,
        dtype=dtype,
        reinitialize_sdf=reinitialize_sdf,
        smoothing_sigma_px=smoothing_sigma_px,
        allow_baseline_fallback=bool(args.allow_baseline_fallback),
        checkpoint_paths=checkpoint_paths,
        xi=xi,
        eta=eta,
        tau=tau,
    )


def predict_next_with_fixed_references(
    runtime: OptimizationRuntime,
    phi_initial: Any,
    process_name: str,
    duration_s: float,
    average_rate: float,
) -> Any:
    import numpy as np

    from epi_pinn.baseline import known_average_rate_baseline
    from epi_pinn.config import process_config
    from epi_pinn.contour import extract_contour20
    from epi_pinn.sampling import build_features

    process_sign = float(process_config(runtime.config, process_name)["sign"])
    model = runtime.models[process_name]
    if model is None:
        return known_average_rate_baseline(
            phi_initial, duration_s, average_rate, process_sign
        )

    contour_config = runtime.config.get("contour", {})
    contour = extract_contour20(
        phi_initial,
        num_points=int(contour_config.get("num_points", 20)),
        min_valid_points=int(contour_config.get("min_valid_points", 10)),
        crossing_policy=str(
            contour_config.get("crossing_policy", "closest_to_previous")
        ),
        first_crossing_policy=str(
            contour_config.get("first_crossing_policy", "topmost")
        ),
    )
    height, width = phi_initial.shape
    spatial_config = runtime.config.get("spatial", {})
    pixel_size_y = float(spatial_config.get("pixel_size_y", 1.0))
    length_y = max(pixel_size_y, (height - 1) * pixel_size_y)
    clip_distance = float(
        runtime.config.get("level_set", {}).get("phi_clip_distance", 32.0)
    )
    features, raw_phi0 = build_features_with_fixed_references(
        build_features,
        phi_initial,
        contour,
        runtime.xi,
        runtime.eta,
        runtime.tau,
        duration_s,
        average_rate,
        runtime.duration_references[process_name],
        runtime.rate_references[process_name],
        clip_distance,
        process_sign,
    )
    prediction = model.predict_numpy(
        features,
        contour.as_features(),
        raw_phi0,
        duration_s,
        average_rate,
        clip_distance,
        device=runtime.device,
        dtype=runtime.dtype,
        length_y=length_y,
    )
    return np.ascontiguousarray(prediction.reshape(height, width), dtype=np.float64)


def rollout_times(
    runtime: OptimizationRuntime, durations_s: Sequence[float]
) -> Dict[str, Any]:
    from epi_pinn.sdf import smooth_and_rebuild_sdf

    if len(durations_s) != len(ROLLOUT_STEPS):
        raise ValueError("durations_s must contain eight values")
    phi = runtime.states["init"]
    predictions: Dict[str, Any] = {}
    for (process, _cycle, output_state), duration, rate in zip(
        ROLLOUT_STEPS, durations_s, runtime.rates
    ):
        phi = predict_next_with_fixed_references(
            runtime, phi, process, float(duration), float(rate)
        )
        if runtime.reinitialize_sdf:
            phi = smooth_and_rebuild_sdf(phi, runtime.smoothing_sigma_px)
        predictions[output_state] = phi
    return predictions


def score_predictions(
    runtime: OptimizationRuntime, predictions: Mapping[str, Any]
) -> Tuple[float, float]:
    from epi_pinn.evaluate import _symmetric_chamfer, _zero_contour_points

    scores = []
    for prediction_state, target_state in (("4M", "5M"), ("4E", "5E")):
        points = _zero_contour_points(predictions[prediction_state])
        if points.size == 0:
            raise ValueError(f"Prediction {prediction_state} has no zero contour")
        score = _symmetric_chamfer(points, runtime.target_points[target_state])
        if not math.isfinite(score):
            raise ValueError(f"Prediction {prediction_state} has an invalid Chamfer score")
        scores.append(float(score))
    return scores[0], scores[1]


def _history_fieldnames() -> Tuple[str, ...]:
    return (
        "evaluation",
        "generation",
        "candidate",
        "status",
        "error",
        "elapsed_s",
        *(f"multiplier_{label}" for label in STEP_LABELS),
        *(f"time_{label}_s" for label in STEP_LABELS),
        "score_4M_vs_5M",
        "score_4E_vs_5E",
        "objective",
    )


def evaluation_to_row(
    evaluation_index: int,
    generation: int,
    candidate: str,
    result: CandidateEvaluation,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "evaluation": evaluation_index,
        "generation": generation,
        "candidate": candidate,
        "status": result.status,
        "error": result.error,
        "elapsed_s": result.elapsed_s,
        "score_4M_vs_5M": result.score_4m,
        "score_4E_vs_5E": result.score_4e,
        "objective": result.objective,
    }
    row.update(
        {f"multiplier_{label}": value for label, value in zip(STEP_LABELS, result.multipliers)}
    )
    row.update(
        {f"time_{label}_s": value for label, value in zip(STEP_LABELS, result.times_s)}
    )
    return row


def write_history(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_history_fieldnames()))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_history(path: Path) -> list[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_cma_state(path: Path, strategy: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(strategy.pickle_dumps())
    temporary.replace(path)


def load_cma_state(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"CMA-ES state does not exist: {path}")
    return pickle.loads(path.read_bytes())


def _file_sha256(path: Optional[Path]) -> Optional[str]:
    if path is None or not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_identity(
    runtime: OptimizationRuntime,
    base_times: Sequence[float],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        "config_path": str(runtime.config_path),
        "config_sha256": _file_sha256(runtime.config_path),
        "workbook_path": str(runtime.workbook_path) if runtime.workbook_path else None,
        "workbook_sha256": _file_sha256(runtime.workbook_path),
        "checkpoint_sha256": {
            process: _file_sha256(path)
            for process, path in runtime.checkpoint_paths.items()
        },
        "base_times_s": list(map(float, base_times)),
        "time_scale_bounds": list(map(float, args.time_scale_bounds)),
        "cma_sigma": float(args.cma_sigma),
        "cma_popsize": int(args.cma_popsize),
        "cma_seed": int(args.cma_seed),
        "duration_references_s": dict(runtime.duration_references),
        "rate_references": dict(runtime.rate_references),
        "fixed_rates": list(map(float, runtime.rates)),
        "objective": OBJECTIVE_NAME,
        "normalization": "fixed_training_references",
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _row_float(row: Mapping[str, Any], key: str) -> float:
    return float(row[key])


def best_valid_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    valid = [
        row
        for row in rows
        if row.get("status") == "ok" and math.isfinite(_row_float(row, "objective"))
    ]
    if not valid:
        raise RuntimeError("CMA-ES did not produce any valid candidate")
    return min(valid, key=lambda row: _row_float(row, "objective"))


def _row_multipliers(row: Mapping[str, Any]) -> Tuple[float, ...]:
    return tuple(_row_float(row, f"multiplier_{label}") for label in STEP_LABELS)


def _row_times(row: Mapping[str, Any]) -> Tuple[float, ...]:
    return tuple(_row_float(row, f"time_{label}_s") for label in STEP_LABELS)


def _save_overlay(
    runtime: OptimizationRuntime,
    predictions: Mapping[str, Any],
    output_path: Path,
    tem_paths: Mapping[str, Optional[Path]],
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    from epi_pinn.visualize import _plot_zero_contours

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 7.0), constrained_layout=True)
    fig.suptitle("CMA-ES best 4M/4E contours against 5M/5E targets", fontsize=15)
    for axis, prediction_state, target_state in zip(
        axes, ("4M", "4E"), ("5M", "5E")
    ):
        prediction = np.asarray(predictions[prediction_state], dtype=np.float64)
        target = np.asarray(runtime.states[target_state], dtype=np.float64)
        height, width = target.shape
        tem_path = tem_paths.get(target_state)
        if tem_path is not None:
            image = plt.imread(str(tem_path))
            axis.imshow(
                image,
                cmap="gray" if image.ndim == 2 else None,
                origin="upper",
                extent=(0, width - 1, height - 1, 0),
            )
        else:
            max_abs = max(float(np.nanmax(np.abs(target))), 1.0)
            axis.imshow(
                target,
                cmap="coolwarm",
                origin="upper",
                vmin=-max_abs,
                vmax=max_abs,
            )
        _plot_zero_contours(
            axis,
            prediction,
            color="black",
            linewidth=2.2,
            linestyle="solid",
            mode="main",
            min_points=25,
            border_margin=2.0,
        )
        _plot_zero_contours(
            axis,
            target,
            color="#ef4444",
            linewidth=2.0,
            linestyle="dashed",
            mode="main",
            min_points=25,
            border_margin=2.0,
        )
        axis.set_title(f"Pred {prediction_state} vs GT {target_state}")
        axis.set_xlim(0, width - 1)
        axis.set_ylim(height - 1, 0)
        axis.set_xlabel("x pixel")
        axis.set_ylabel("y pixel")
        axis.legend(
            handles=[
                Line2D([0], [0], color="black", linewidth=2.2, label="prediction"),
                Line2D(
                    [0], [0], color="#ef4444", linewidth=2.0,
                    linestyle="dashed", label="target"
                ),
            ],
            loc="upper right",
        )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_best_artifacts(
    runtime: OptimizationRuntime,
    predictions: Mapping[str, Any],
    best_row: Mapping[str, Any],
    baseline_row: Mapping[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
    stop_reasons: Mapping[str, Any],
    evaluation_count: int,
) -> None:
    import numpy as np

    from epi_pinn.contour import extract_contour20, save_contour_csv
    from epi_pinn.evaluate import evaluate_pair
    from epi_pinn.excel_io import write_prediction_workbook

    best_dir = output_dir / "best"
    contour_dir = best_dir / "contours"
    best_dir.mkdir(parents=True, exist_ok=True)
    contour_dir.mkdir(parents=True, exist_ok=True)
    for state, phi in predictions.items():
        np.save(best_dir / f"{state}.npy", phi)
        contour = extract_contour20(
            phi,
            num_points=int(runtime.config.get("contour", {}).get("num_points", 20)),
            min_valid_points=int(
                runtime.config.get("contour", {}).get("min_valid_points", 10)
            ),
        )
        save_contour_csv(contour, str(contour_dir / f"{state}_contour20.csv"))
    write_prediction_workbook(
        predictions, str(best_dir / "predictions_1m_to_4e.xlsx")
    )

    metrics = {
        "4M_vs_5M": evaluate_pair(
            predictions["4M"], runtime.states["5M"], runtime.config
        ),
        "4E_vs_5E": evaluate_pair(
            predictions["4E"], runtime.states["5E"], runtime.config
        ),
    }
    write_json(best_dir / "comparison_metrics.json", metrics)

    best_objective_from_artifacts = (
        metrics["4M_vs_5M"]["zero_contour_symmetric_chamfer_px"]
        + metrics["4E_vs_5E"]["zero_contour_symmetric_chamfer_px"]
    )
    if not math.isclose(
        best_objective_from_artifacts,
        _row_float(best_row, "objective"),
        rel_tol=1.0e-10,
        abs_tol=1.0e-10,
    ):
        raise RuntimeError("Saved best predictions do not reproduce the optimized objective")

    tem_paths = {
        "5M": resolve_path(args.tem_5m, runtime.config_path.parent.parent).resolve()
        if args.tem_5m else None,
        "5E": resolve_path(args.tem_5e, runtime.config_path.parent.parent).resolve()
        if args.tem_5e else None,
    }
    for state, path in tem_paths.items():
        if path is not None and not path.exists():
            raise FileNotFoundError(f"TEM image for {state} does not exist: {path}")
    figure_path = best_dir / "4M_4E_on_5M_5E.png"
    if not args.no_plot:
        _save_overlay(runtime, predictions, figure_path, tem_paths)

    baseline_objective = _row_float(baseline_row, "objective")
    best_objective = _row_float(best_row, "objective")
    improvement_percent = (
        100.0 * (baseline_objective - best_objective) / baseline_objective
        if baseline_objective > 0.0 else 0.0
    )
    result = {
        "objective": OBJECTIVE_NAME,
        "normalization": "fixed_training_references",
        "baseline": {
            "times_s": dict(zip(STEP_LABELS, _row_times(baseline_row))),
            "score_4M_vs_5M": _row_float(baseline_row, "score_4M_vs_5M"),
            "score_4E_vs_5E": _row_float(baseline_row, "score_4E_vs_5E"),
            "objective": baseline_objective,
        },
        "best": {
            "multipliers": dict(zip(STEP_LABELS, _row_multipliers(best_row))),
            "times_s": dict(zip(STEP_LABELS, _row_times(best_row))),
            "score_4M_vs_5M": metrics["4M_vs_5M"][
                "zero_contour_symmetric_chamfer_px"
            ],
            "score_4E_vs_5E": metrics["4E_vs_5E"][
                "zero_contour_symmetric_chamfer_px"
            ],
            "objective": best_objective_from_artifacts,
        },
        "improvement_percent": improvement_percent,
        "evaluations": evaluation_count,
        "stop_reasons": stop_reasons,
        "calibration_warning": (
            "5M and 5E were optimized directly and are not unbiased holdout targets."
        ),
    }
    write_json(output_dir / "best_result.json", result)
    write_json(
        best_dir / "manifest.json",
        {
            "config": str(runtime.config_path),
            "workbook": str(runtime.workbook_path) if runtime.workbook_path else None,
            "times_s": dict(zip(STEP_LABELS, _row_times(best_row))),
            "fixed_rates": dict(zip(STEP_LABELS, runtime.rates)),
            "duration_references_s": runtime.duration_references,
            "rate_references": runtime.rate_references,
            "reinitialize_sdf_each_step": runtime.reinitialize_sdf,
            "interface_smoothing_sigma_px": runtime.smoothing_sigma_px,
            "figure": str(figure_path) if not args.no_plot else None,
        },
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize 1M through 4E process durations with CMA-ES to minimize "
            "the 4M-vs-5M plus 4E-vs-5E full zero-contour Chamfer score."
        )
    )
    parser.add_argument("--config", default="configs/ablation_full_physics.yaml")
    parser.add_argument(
        "--workbook",
        default=None,
        help="Workbook containing exact init, 1M, 1E, 2M, 2E, 5M, and 5E sheets",
    )
    parser.add_argument("--infer-missing-rates", action="store_true")
    parser.add_argument("--allow-baseline-fallback", action="store_true")
    parser.add_argument(
        "--times",
        nargs=8,
        type=float,
        default=None,
        metavar="SECONDS",
        help="Initial times ordered as 1M 1E 2M 2E 3M 3E 4M 4E",
    )
    parser.add_argument(
        "--time-scale-bounds", nargs=2, type=float, default=(0.5, 1.5),
        metavar=("LOWER", "UPPER")
    )
    parser.add_argument("--cma-sigma", type=float, default=0.2)
    parser.add_argument("--cma-generations", type=int, default=30)
    parser.add_argument("--cma-popsize", type=int, default=10)
    parser.add_argument("--cma-seed", type=int, default=42)
    parser.add_argument("--invalid-objective", type=float, default=1.0e9)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tem-5m", default=None)
    parser.add_argument("--tem-5e", default=None)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    lower, upper = args.time_scale_bounds
    if not 0.0 < lower < upper:
        parser.error("--time-scale-bounds must satisfy 0 < LOWER < UPPER")
    if args.cma_sigma <= 0.0:
        parser.error("--cma-sigma must be positive")
    if args.cma_generations <= 0 or args.cma_popsize < 2:
        parser.error("CMA generations must be positive and population must be at least 2")
    if not math.isfinite(args.invalid_objective) or args.invalid_objective <= 0.0:
        parser.error("--invalid-objective must be positive and finite")
    if args.times is not None and any(
        not math.isfinite(value) or value <= 0.0 for value in args.times
    ):
        parser.error("all initial times must be positive and finite")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = add_src_path()
    try:
        import cma
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'cma'. Install the project with: python -m pip install -e ."
        ) from exc

    from epi_pinn.config import load_config, output_dir, project_root_from_config_path, schedule_seconds

    config = load_config(args.config)
    config_root = project_root_from_config_path(args.config)
    base_times = tuple(
        float(value) for value in args.times
    ) if args.times is not None else tuple(
        schedule_seconds(config, process, cycle)
        for process, cycle, _state in ROLLOUT_STEPS
    )
    runtime = prepare_runtime(args)
    base_output = output_dir(config, config_root)
    optimization_dir = (
        resolve_path(args.output_dir, root).resolve()
        if args.output_dir else (base_output / "time_optimization").resolve()
    )
    optimization_dir.mkdir(parents=True, exist_ok=True)
    history_path = optimization_dir / "optimization_history.csv"
    state_path = optimization_dir / "cmaes_state.pkl"
    identity_path = optimization_dir / "run_identity.json"
    identity = build_run_identity(runtime, base_times, args)

    if args.resume:
        if not identity_path.exists():
            raise FileNotFoundError(f"Resume identity does not exist: {identity_path}")
        stored_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if stored_identity != _json_safe(identity):
            raise ValueError("Resume inputs do not match the saved CMA-ES run identity")
        strategy = load_cma_state(state_path)
        strategy.opts.set({"maxiter": int(args.cma_generations)})
        # A strategy pickled at maxiter retains the prior stop result until
        # the stop dictionary is cleared after extending the iteration limit.
        strategy.stop().clear()
        rows: list[Dict[str, Any]] = [
            row for row in read_history(history_path)
            if int(row["generation"]) <= int(strategy.countiter)
        ]
        write_history(history_path, rows)
    else:
        if history_path.exists() or state_path.exists() or identity_path.exists():
            raise FileExistsError(
                f"Optimization output already exists in {optimization_dir}; "
                "use --resume or choose --output-dir"
            )
        options = {
            "bounds": [
                [float(args.time_scale_bounds[0])] * len(ROLLOUT_STEPS),
                [float(args.time_scale_bounds[1])] * len(ROLLOUT_STEPS),
            ],
            "popsize": int(args.cma_popsize),
            "seed": int(args.cma_seed),
            "maxiter": int(args.cma_generations),
            "verb_log": 0,
        }
        strategy = cma.CMAEvolutionStrategy(
            [1.0] * len(ROLLOUT_STEPS), float(args.cma_sigma), options
        )
        rows = []
        baseline = evaluate_candidate(
            [1.0] * len(ROLLOUT_STEPS),
            base_times,
            float(args.time_scale_bounds[0]),
            float(args.time_scale_bounds[1]),
            float(args.invalid_objective),
            lambda times: rollout_times(runtime, times),
            lambda predictions: score_predictions(runtime, predictions),
        )
        rows.append(evaluation_to_row(0, 0, "baseline", baseline))
        if baseline.status != "ok":
            raise RuntimeError(f"Baseline candidate failed: {baseline.error}")
        write_history(history_path, rows)
        write_json(identity_path, identity)
        save_cma_state(state_path, strategy)

    evaluation_index = max((int(row["evaluation"]) for row in rows), default=0)
    cache: Dict[Tuple[float, ...], CandidateEvaluation] = {}
    for row in rows:
        key = tuple(round(value, 12) for value in _row_multipliers(row))
        cache[key] = CandidateEvaluation(
            multipliers=_row_multipliers(row),
            times_s=_row_times(row),
            score_4m=_row_float(row, "score_4M_vs_5M"),
            score_4e=_row_float(row, "score_4E_vs_5E"),
            objective=_row_float(row, "objective"),
            status=str(row["status"]),
            error=str(row.get("error", "")),
            elapsed_s=_row_float(row, "elapsed_s"),
        )

    while strategy.countiter < args.cma_generations and not strategy.stop():
        generation = int(strategy.countiter) + 1
        solutions = strategy.ask()
        generation_results = []
        generation_rows = []
        for candidate_index, solution in enumerate(solutions):
            key = tuple(round(float(value), 12) for value in solution)
            if key in cache:
                cached = cache[key]
                result = CandidateEvaluation(
                    multipliers=cached.multipliers,
                    times_s=cached.times_s,
                    score_4m=cached.score_4m,
                    score_4e=cached.score_4e,
                    objective=cached.objective,
                    status="ok" if cached.status == "ok" else "failed",
                    error=cached.error,
                    elapsed_s=0.0,
                )
            else:
                result = evaluate_candidate(
                    solution,
                    base_times,
                    float(args.time_scale_bounds[0]),
                    float(args.time_scale_bounds[1]),
                    float(args.invalid_objective),
                    lambda times: rollout_times(runtime, times),
                    lambda predictions: score_predictions(runtime, predictions),
                )
                cache[key] = result
            evaluation_index += 1
            generation_results.append(result)
            generation_rows.append(
                evaluation_to_row(
                    evaluation_index, generation, str(candidate_index), result
                )
            )
            print(
                f"generation={generation}/{args.cma_generations} "
                f"candidate={candidate_index + 1}/{len(solutions)} "
                f"objective={result.objective:.6g} "
                f"4M={result.score_4m:.6g} 4E={result.score_4e:.6g} "
                f"status={result.status}",
                flush=True,
            )
        strategy.tell(solutions, [result.objective for result in generation_results])
        strategy.disp()
        rows.extend(generation_rows)
        write_history(history_path, rows)
        save_cma_state(state_path, strategy)

    best_row = best_valid_row(rows)
    baseline_row = next(row for row in rows if str(row["candidate"]) == "baseline")
    best_times = _row_times(best_row)
    best_predictions = rollout_times(runtime, best_times)
    final_score_4m, final_score_4e = score_predictions(runtime, best_predictions)
    final_objective = combined_objective(
        final_score_4m, final_score_4e, float(args.invalid_objective)
    )
    if not math.isclose(
        final_objective,
        _row_float(best_row, "objective"),
        rel_tol=1.0e-10,
        abs_tol=1.0e-10,
    ):
        raise RuntimeError("Best candidate is not deterministic when re-evaluated")

    save_best_artifacts(
        runtime,
        best_predictions,
        best_row,
        baseline_row,
        optimization_dir,
        args,
        strategy.stop(),
        evaluation_index,
    )
    print(f"Best objective: {final_objective:.6g}")
    print("Best times (s):")
    for label, value in zip(STEP_LABELS, best_times):
        print(f"  {label}: {value:.6g}")
    print(f"Saved optimization artifacts: {optimization_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
