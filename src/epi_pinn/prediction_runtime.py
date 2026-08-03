"""Reusable prediction runtime for legacy and training-reference inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from epi_pinn.config import (
    average_rate,
    device_name,
    load_config,
    output_dir,
    process_config,
    project_root_from_config_path,
)
from epi_pinn.excel_io import load_state_arrays, read_excel_array
from epi_pinn.rollout import _infer_process_rate, _load_model, predict_next_levelset
from epi_pinn.sdf import ensure_signed_distance, gaussian_smooth_interface, rebuild_sdf_from_mask
from epi_pinn.train import torch_dtype


BASE_COMMIT = "2ab4856c44d4fe1ce7c189f0701cc190bd1b5397"
NORMALIZATION_MODES = ("legacy", "training-reference")
PREDICT_1M_TO_4E_STEPS: Tuple[Tuple[str, int, str], ...] = (
    ("deposition", 1, "1M"),
    ("etch", 1, "1E"),
    ("deposition", 2, "2M"),
    ("etch", 2, "2E"),
    ("deposition", 3, "3M"),
    ("etch", 3, "3E"),
    ("deposition", 4, "4M"),
    ("etch", 4, "4E"),
)
ROLLOUT_2E_TO_5E_STEPS: Tuple[Tuple[str, int, str], ...] = (
    ("deposition", 3, "3M"),
    ("etch", 3, "3E"),
    ("deposition", 4, "4M"),
    ("etch", 4, "4E"),
    ("deposition", 5, "5M"),
    ("etch", 5, "5E"),
)
WORKBOOK_STATES = ("init", "1M", "1E", "2M", "2E", "5M", "5E")


@dataclass
class PredictionRuntime:
    config_path: Path
    config: Mapping[str, Any]
    config_root: Path
    workbook_path: Optional[Path]
    checkpoint_dir: Path
    checkpoint_paths: Mapping[str, Path]
    raw_states: Mapping[str, np.ndarray]
    states: Mapping[str, np.ndarray]
    models: Mapping[str, Optional[torch.nn.Module]]
    inferred_rates: Mapping[str, Optional[float]]
    duration_references: Mapping[str, float]
    rate_references: Mapping[str, float]
    normalization_mode: str
    device: str
    dtype: torch.dtype
    reinitialize_sdf_each_step: bool


@dataclass
class PredictionStepResult:
    process_name: str
    cycle: int
    output_state: str
    input_state: str
    duration_s: float
    average_rate: float
    input_phi: np.ndarray
    model_input_phi: np.ndarray
    prediction_phi: np.ndarray


@dataclass
class PredictionSequenceResult:
    start_state: str
    normalization_mode: str
    prediction_gaussian_sigma: float
    steps: list[PredictionStepResult]
    predictions: Dict[str, np.ndarray]


def _resolve_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _load_workbook_states(config: Mapping[str, Any], workbook_path: Path) -> Dict[str, np.ndarray]:
    data_cfg = config.get("data", {})
    expected_shape = (
        int(data_cfg.get("expected_height", 350)),
        int(data_cfg.get("expected_width", 200)),
    )
    allow_transpose = bool(data_cfg.get("allow_transpose", True))
    return {
        state: read_excel_array(
            str(workbook_path),
            state,
            expected_shape=expected_shape,
            allow_transpose=allow_transpose,
        )
        for state in WORKBOOK_STATES
    }


def _positive_reference(value: Any, label: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    return number


def _fixed_rate_reference(
    config: Mapping[str, Any], process_name: str, inferred_rate: Optional[float]
) -> float:
    proc = process_config(config, process_name)
    configured = proc.get("rate_reference")
    if configured is None:
        configured = proc.get("average_rate_default")
    if configured is not None:
        return _positive_reference(configured, f"{process_name} rate_reference")
    if inferred_rate is not None:
        return _positive_reference(inferred_rate, f"{process_name} inferred rate reference")

    rates = []
    for cycle in range(1, 6):
        try:
            rates.append(average_rate(config, process_name, cycle))
        except (KeyError, ValueError):
            pass
    if not rates:
        raise ValueError(
            f"Cannot determine fixed rate reference for {process_name}; configure a rate "
            "or pass --infer-missing-rates"
        )
    return _positive_reference(float(np.median(rates)), f"{process_name} median rate reference")


def load_prediction_runtime(
    config_path: str,
    workbook_path: str | None = None,
    checkpoint_dir: str | None = None,
    infer_missing_rates: bool = False,
    allow_baseline_fallback: bool = False,
    normalization_mode: str = "legacy",
) -> PredictionRuntime:
    if normalization_mode not in NORMALIZATION_MODES:
        raise ValueError(f"normalization_mode must be one of {NORMALIZATION_MODES}")

    config_file = Path(config_path).resolve()
    config = load_config(str(config_file))
    config_root = project_root_from_config_path(str(config_file))
    workbook = _resolve_path(workbook_path, config_root) if workbook_path else None
    raw_states = (
        _load_workbook_states(config, workbook)
        if workbook is not None
        else load_state_arrays(config, base_dir=config_root)
    )
    missing = [state for state in WORKBOOK_STATES if state not in raw_states]
    if missing:
        raise KeyError("Missing required states: " + ", ".join(missing))

    level_cfg = config.get("level_set", {})
    states = {
        state: ensure_signed_distance(raw_states[state], level_cfg)
        for state in WORKBOOK_STATES
    }
    artifact_dir = output_dir(config, config_root)
    checkpoints = (
        _resolve_path(checkpoint_dir, config_root)
        if checkpoint_dir else (artifact_dir / "checkpoints").resolve()
    )
    device = device_name(config)
    dtype = torch_dtype(config.get("training", {}).get("dtype", "float64"))
    checkpoint_paths = {
        process: checkpoints / f"{process}_best.pt"
        for process in ("deposition", "etch")
    }
    models = {
        process: _load_model(config, process, checkpoint_paths[process], device, dtype)
        for process in ("deposition", "etch")
    }
    for process, model in models.items():
        if model is None and not allow_baseline_fallback:
            raise FileNotFoundError(
                f"Missing checkpoint for {process}: {checkpoint_paths[process]}"
            )
        if model is not None:
            model.eval()

    inferred_rates = {
        process: (
            _infer_process_rate(config, raw_states, process)
            if infer_missing_rates else None
        )
        for process in ("deposition", "etch")
    }
    duration_references = {
        process: _positive_reference(
            process_config(config, process).get("duration_reference_s", 1.0),
            f"{process} duration_reference_s",
        )
        for process in ("deposition", "etch")
    }
    rate_references = {
        process: _fixed_rate_reference(config, process, inferred_rates[process])
        for process in ("deposition", "etch")
    }
    return PredictionRuntime(
        config_path=config_file,
        config=config,
        config_root=config_root,
        workbook_path=workbook,
        checkpoint_dir=checkpoints,
        checkpoint_paths=checkpoint_paths,
        raw_states=raw_states,
        states=states,
        models=models,
        inferred_rates=inferred_rates,
        duration_references=duration_references,
        rate_references=rate_references,
        normalization_mode=normalization_mode,
        device=device,
        dtype=dtype,
        reinitialize_sdf_each_step=bool(
            config.get("rollout", {}).get("reinitialize_sdf_each_step", True)
        ),
    )


def _predict_training_reference(
    runtime: PredictionRuntime,
    phi_initial: np.ndarray,
    process_name: str,
    duration_s: float,
    average_rate_value: float,
) -> np.ndarray:
    from epi_pinn.contour import extract_contour20
    from epi_pinn.sampling import build_features, full_grid_query

    model = runtime.models[process_name]
    process_sign = float(process_config(runtime.config, process_name)["sign"])
    if model is None:
        return predict_next_levelset(
            phi_initial,
            duration_s,
            average_rate_value,
            process_sign,
            runtime.config,
            None,
            device=runtime.device,
            dtype=runtime.dtype,
        )

    contour_cfg = runtime.config.get("contour", {})
    level_cfg = runtime.config.get("level_set", {})
    clip_distance = float(level_cfg.get("phi_clip_distance", 32.0))
    contour = extract_contour20(
        phi_initial,
        num_points=int(contour_cfg.get("num_points", 20)),
        min_valid_points=int(contour_cfg.get("min_valid_points", 10)),
        crossing_policy=str(contour_cfg.get("crossing_policy", "closest_to_previous")),
        first_crossing_policy=str(contour_cfg.get("first_crossing_policy", "topmost")),
    )
    height, width = phi_initial.shape
    pixel_size_y = float(runtime.config.get("spatial", {}).get("pixel_size_y", 1.0))
    length_y = max(pixel_size_y, (height - 1) * pixel_size_y)
    xi, eta, _x, _y = full_grid_query(height, width)
    tau = np.ones_like(xi)
    features, raw_phi0 = build_features(
        phi_initial,
        contour,
        xi,
        eta,
        tau,
        duration_s,
        average_rate_value,
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
        average_rate_value,
        clip_distance,
        device=runtime.device,
        dtype=runtime.dtype,
        length_y=length_y,
    )
    return np.ascontiguousarray(prediction.reshape(height, width), dtype=np.float64)


def run_prediction_sequence(
    runtime: PredictionRuntime,
    start_phi: np.ndarray,
    steps: Sequence[tuple[str, int, str]],
    durations_s: Sequence[float],
    prediction_gaussian_sigma: float = 0.0,
    start_state: str = "init",
) -> PredictionSequenceResult:
    if len(steps) != len(durations_s):
        raise ValueError("steps and durations_s must have the same length")
    durations = tuple(float(value) for value in durations_s)
    if any(not np.isfinite(value) or value <= 0.0 for value in durations):
        raise ValueError("all durations must be positive and finite")

    phi = np.ascontiguousarray(np.asarray(start_phi, dtype=np.float64)).copy()
    results: list[PredictionStepResult] = []
    predictions: Dict[str, np.ndarray] = {}
    input_state = start_state
    with torch.inference_mode():
        for (process_name, cycle, output_state), duration in zip(steps, durations):
            input_phi = phi.copy()
            model_input_phi = gaussian_smooth_interface(
                input_phi, prediction_gaussian_sigma
            )
            rate = average_rate(
                runtime.config,
                process_name,
                cycle,
                fallback=runtime.inferred_rates[process_name],
            )
            process_sign = float(process_config(runtime.config, process_name)["sign"])
            if runtime.normalization_mode == "legacy":
                prediction = predict_next_levelset(
                    model_input_phi,
                    duration,
                    rate,
                    process_sign,
                    runtime.config,
                    runtime.models[process_name],
                    device=runtime.device,
                    dtype=runtime.dtype,
                )
            else:
                prediction = _predict_training_reference(
                    runtime, model_input_phi, process_name, duration, rate
                )
            if not np.isfinite(prediction).all():
                raise ValueError(f"prediction {output_state} contains NaN or Inf")
            if runtime.reinitialize_sdf_each_step:
                prediction = rebuild_sdf_from_mask(prediction < 0.0)
            phi = np.ascontiguousarray(prediction, dtype=np.float64)
            predictions[output_state] = phi.copy()
            results.append(
                PredictionStepResult(
                    process_name=process_name,
                    cycle=cycle,
                    output_state=output_state,
                    input_state=input_state,
                    duration_s=duration,
                    average_rate=float(rate),
                    input_phi=input_phi,
                    model_input_phi=model_input_phi.copy(),
                    prediction_phi=phi.copy(),
                )
            )
            input_state = output_state
    return PredictionSequenceResult(
        start_state=start_state,
        normalization_mode=runtime.normalization_mode,
        prediction_gaussian_sigma=float(prediction_gaussian_sigma),
        steps=results,
        predictions=predictions,
    )


def normalization_metadata(runtime: PredictionRuntime) -> Dict[str, Any]:
    if runtime.normalization_mode == "legacy":
        return {
            "mode": "legacy",
            "legacy_self_reference": True,
            "duration_normalized_behavior": "duration / duration = 1",
            "rate_normalized_behavior": "rate / rate = 1",
            "base_commit": BASE_COMMIT,
        }
    return {
        "mode": "training-reference",
        "legacy_self_reference": False,
        "duration_references_s": dict(runtime.duration_references),
        "rate_references": dict(runtime.rate_references),
        "base_commit": BASE_COMMIT,
    }