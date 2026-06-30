"""Rollout from 2E through predicted 5E."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch

from epi_pinn.baseline import estimate_average_rate_from_pair, known_average_rate_baseline
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
from epi_pinn.models import DepositionPINN, EtchPINN
from epi_pinn.sampling import build_features, full_grid_query
from epi_pinn.sdf import ensure_signed_distance
from epi_pinn.train import torch_dtype


ROLL_OUT_STEPS = [
    ("deposition", 3, "3M"),
    ("etch", 3, "3E"),
    ("deposition", 4, "4M"),
    ("etch", 4, "4E"),
    ("deposition", 5, "5M"),
    ("etch", 5, "5E"),
]


def _load_model(config: Mapping[str, Any], process_name: str, checkpoint_path: Path, device: str, dtype: torch.dtype) -> Optional[torch.nn.Module]:
    if not checkpoint_path.exists():
        return None
    model = DepositionPINN(config.get("model", {})) if process_name == "deposition" else EtchPINN(config.get("model", {}))
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    allowed_keys = {"raw_curvature_velocity_weight"}
    missing = [key for key in load_result.missing_keys if key not in allowed_keys]
    unexpected = [key for key in load_result.unexpected_keys if key not in allowed_keys]
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} is incompatible with the model; "
            f"missing={missing}, unexpected={unexpected}"
        )
    model.to(device=device, dtype=dtype)
    model.eval()
    return model


def _infer_process_rate(config: Mapping[str, Any], states: Mapping[str, np.ndarray], process_name: str) -> Optional[float]:
    transitions_key = "deposition_train" if process_name == "deposition" else "etch_train"
    transitions = config.get("transitions", {}).get(transitions_key, [])
    proc = process_config(config, process_name)
    sign = float(proc["sign"])
    level_cfg = config.get("level_set", {})
    narrow = float(level_cfg.get("narrow_band_distance", 8.0))
    rates = []
    for transition in transitions:
        try:
            duration = schedule_seconds(config, process_name, int(transition["cycle"]))
            phi0 = ensure_signed_distance(states[str(transition["input_state"])], level_cfg)
            phi1 = ensure_signed_distance(states[str(transition["target_state"])], level_cfg)
            rates.append(estimate_average_rate_from_pair(phi0, phi1, duration, sign, narrow))
        except Exception:
            continue
    if not rates:
        return None
    return float(np.median(rates))


def predict_next_levelset(
    phi_initial: np.ndarray,
    duration_s: float,
    average_rate_value: float,
    process_sign: float,
    config: Mapping[str, Any],
    model: Optional[torch.nn.Module] = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> np.ndarray:
    if model is None:
        return known_average_rate_baseline(phi_initial, duration_s, average_rate_value, process_sign)

    contour_cfg = config.get("contour", {})
    level_cfg = config.get("level_set", {})
    clip_distance = float(level_cfg.get("phi_clip_distance", 32.0))
    contour = extract_contour20(
        phi_initial,
        num_points=int(contour_cfg.get("num_points", 20)),
        min_valid_points=int(contour_cfg.get("min_valid_points", 10)),
        crossing_policy=str(contour_cfg.get("crossing_policy", "closest_to_previous")),
        first_crossing_policy=str(contour_cfg.get("first_crossing_policy", "topmost")),
    )
    height, width = phi_initial.shape
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
        duration_s,
        average_rate_value,
        clip_distance,
        process_sign,
    )
    pred = model.predict_numpy(
        features,
        contour.as_features(),
        raw_phi0,
        duration_s,
        average_rate_value,
        clip_distance,
        device=device,
        dtype=dtype,
    )
    return np.ascontiguousarray(pred.reshape(height, width), dtype=np.float64)


def run_rollout(config_path: str, infer_missing_rates: bool = False, allow_baseline_fallback: bool = False) -> Dict[str, np.ndarray]:
    config = load_config(config_path)
    root = project_root_from_config_path(config_path)
    states = load_state_arrays(config, base_dir=root)
    level_cfg = config.get("level_set", {})
    start_state = str(config.get("rollout", {}).get("start_state", "2E"))
    phi = ensure_signed_distance(states[start_state], level_cfg)

    out_dir = output_dir(config, root)
    checkpoint_dir = out_dir / "checkpoints"
    prediction_dir = out_dir / "predictions"
    contour_dir = out_dir / "contours"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    contour_dir.mkdir(parents=True, exist_ok=True)

    device = device_name(config)
    dtype = torch_dtype(config.get("training", {}).get("dtype", "float64"))
    models = {
        "deposition": _load_model(config, "deposition", checkpoint_dir / "deposition_best.pt", device, dtype),
        "etch": _load_model(config, "etch", checkpoint_dir / "etch_best.pt", device, dtype),
    }
    inferred = {
        "deposition": _infer_process_rate(config, states, "deposition") if infer_missing_rates else None,
        "etch": _infer_process_rate(config, states, "etch") if infer_missing_rates else None,
    }

    predictions: Dict[str, np.ndarray] = {}
    for process_name, cycle, output_state in ROLL_OUT_STEPS:
        proc = process_config(config, process_name)
        process_sign = float(proc["sign"])
        duration = schedule_seconds(config, process_name, cycle)
        rate = average_rate(config, process_name, cycle, fallback=inferred[process_name])
        model = models[process_name]
        if model is None and not allow_baseline_fallback:
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

    if bool(config.get("evaluation", {}).get("export_prediction_workbook", True)):
        write_prediction_workbook(predictions, str(prediction_dir / "predictions.xlsx"))
    return predictions