"""Holdout evaluation metrics for 5M and 5E predictions."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd

from epi_pinn.config import load_config, output_dir, project_root_from_config_path
from epi_pinn.contour import ContourCondition, extract_contour20
from epi_pinn.excel_io import load_state_arrays
from epi_pinn.geometry import normalized_to_pixel_xy, spatial_gradients
from epi_pinn.sdf import ensure_signed_distance, material_mask


def dice_iou(pred_mask: np.ndarray, target_mask: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred_mask, dtype=bool)
    target = np.asarray(target_mask, dtype=bool)
    intersection = float(np.logical_and(pred, target).sum())
    pred_sum = float(pred.sum())
    target_sum = float(target.sum())
    union = float(np.logical_or(pred, target).sum())
    dice = (2.0 * intersection + 1.0e-8) / (pred_sum + target_sum + 1.0e-8)
    iou = (intersection + 1.0e-8) / (union + 1.0e-8)
    return {"dice": dice, "iou": iou}


def _contour_pixels(contour: ContourCondition, height: int, width: int) -> np.ndarray:
    x, y = normalized_to_pixel_xy(contour.points_xy[:, 0], contour.points_xy[:, 1], height, width)
    return np.stack([x, y], axis=1)


def _symmetric_chamfer(points_a: np.ndarray, points_b: np.ndarray) -> float:
    if points_a.size == 0 or points_b.size == 0:
        return float("nan")
    diff = points_a[:, None, :] - points_b[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    return float(np.mean(np.min(dist, axis=1)) + np.mean(np.min(dist, axis=0)))


def _contour_metrics(phi_pred: np.ndarray, phi_target: np.ndarray, contour_config: Mapping[str, Any]) -> Dict[str, float]:
    pred_contour = extract_contour20(
        phi_pred,
        num_points=int(contour_config.get("num_points", 20)),
        min_valid_points=int(contour_config.get("min_valid_points", 10)),
    )
    target_contour = extract_contour20(
        phi_target,
        num_points=int(contour_config.get("num_points", 20)),
        min_valid_points=int(contour_config.get("min_valid_points", 10)),
    )
    height, width = phi_pred.shape
    shared = (pred_contour.valid_mask > 0.0) & (target_contour.valid_mask > 0.0)
    if shared.any():
        _x_pred, y_pred = normalized_to_pixel_xy(
            pred_contour.points_xy[shared, 0], pred_contour.points_xy[shared, 1], height, width
        )
        _x_true, y_true = normalized_to_pixel_xy(
            target_contour.points_xy[shared, 0], target_contour.points_xy[shared, 1], height, width
        )
        y_mae = float(np.mean(np.abs(y_pred - y_true)))
    else:
        y_mae = float("nan")
    pred_points = _contour_pixels(pred_contour, height, width)[pred_contour.valid_mask > 0.0]
    true_points = _contour_pixels(target_contour, height, width)[target_contour.valid_mask > 0.0]
    return {
        "contour20_y_mae_px": y_mae,
        "zero_contour_symmetric_chamfer_px": _symmetric_chamfer(pred_points, true_points),
    }


def evaluate_pair(phi_pred: np.ndarray, phi_target: np.ndarray, config: Mapping[str, Any]) -> Dict[str, float]:
    pred = np.asarray(phi_pred, dtype=np.float64)
    target = np.asarray(phi_target, dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {pred.shape} vs {target.shape}")

    narrow = float(config.get("level_set", {}).get("narrow_band_distance", 8.0))
    narrow_mask = np.abs(target) <= narrow
    if not narrow_mask.any():
        narrow_mask = np.ones(target.shape, dtype=bool)
    mask_metrics = dice_iou(material_mask(pred), material_mask(target))
    area_target = float(material_mask(target).sum())
    area_pred = float(material_mask(pred).sum())
    phi_x, phi_y = spatial_gradients(pred)
    eikonal_error = np.mean(np.abs(np.sqrt(phi_x * phi_x + phi_y * phi_y + 1.0e-12) - 1.0))

    metrics = {
        "full_field_levelset_mae": float(np.mean(np.abs(pred - target))),
        "narrow_band_levelset_mae": float(np.mean(np.abs(pred[narrow_mask] - target[narrow_mask]))),
        "material_mask_dice": mask_metrics["dice"],
        "material_mask_iou": mask_metrics["iou"],
        "material_area_percent_error": float(100.0 * (area_pred - area_target) / max(area_target, 1.0)),
        "mean_eikonal_error": float(eikonal_error),
    }
    metrics.update(_contour_metrics(pred, target, config.get("contour", {})))
    return metrics


def evaluate_holdout(config_path: str) -> Dict[str, Dict[str, float]]:
    config = load_config(config_path)
    root = project_root_from_config_path(config_path)
    out_dir = output_dir(config, root)
    states = load_state_arrays(config, base_dir=root)
    prediction_dir = out_dir / "predictions"
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    targets = {
        "5M": str(config.get("evaluation", {}).get("holdout_deposition_state", "5M")),
        "5E": str(config.get("evaluation", {}).get("holdout_etch_state", "5E")),
    }
    results: Dict[str, Dict[str, float]] = {}
    for prediction_state, target_state in targets.items():
        pred_path = prediction_dir / f"{prediction_state}.npy"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing prediction file: {pred_path}")
        pred = np.load(pred_path)
        target = ensure_signed_distance(states[target_state], config.get("level_set", {}))
        metrics = evaluate_pair(pred, target, config)
        results[prediction_state] = metrics
        with (metrics_dir / f"{prediction_state}_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

    summary = pd.DataFrame.from_dict(results, orient="index")
    summary.index.name = "state"
    summary.to_csv(metrics_dir / "summary.csv")
    report_lines = ["# Holdout Evaluation Report", ""]
    for state, metrics in results.items():
        report_lines.append(f"## {state}")
        for key, value in metrics.items():
            report_lines.append(f"- {key}: {value:.6g}")
        report_lines.append("")
    (out_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return results