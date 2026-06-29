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


def _material_mask(phi: np.ndarray) -> np.ndarray:
    return np.asarray(phi) < 0


def _contour_pixels(contour: ContourCondition, height: int, width: int) -> np.ndarray:
    x, y = normalized_to_pixel_xy(contour.points_xy[:, 0], contour.points_xy[:, 1], height, width)
    return np.stack([x, y], axis=1)


def _find_zero_contours(phi: np.ndarray) -> list:
    from skimage import measure

    return measure.find_contours(np.asarray(phi, dtype=np.float64), level=0.0)


def _zero_contour_points(phi: np.ndarray) -> np.ndarray:
    contours = _find_zero_contours(phi)
    if not contours:
        return np.empty((0, 2), dtype=np.float64)

    point_sets = []
    for contour in contours:
        contour_array = np.asarray(contour, dtype=np.float64)
        if contour_array.size == 0:
            continue
        # skimage returns (row, col), while Chamfer uses pixel (x, y).
        point_sets.append(np.stack([contour_array[:, 1], contour_array[:, 0]], axis=1))
    if not point_sets:
        return np.empty((0, 2), dtype=np.float64)
    return np.ascontiguousarray(np.concatenate(point_sets, axis=0), dtype=np.float64)


def _mean_min_distance(points_a: np.ndarray, points_b: np.ndarray, chunk_size: int = 4096) -> float:
    minima = []
    for start in range(0, points_a.shape[0], chunk_size):
        chunk = points_a[start : start + chunk_size]
        diff = chunk[:, None, :] - points_b[None, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        minima.append(np.sqrt(np.min(dist_sq, axis=1)))
    return float(np.mean(np.concatenate(minima)))


def _symmetric_chamfer(points_a: np.ndarray, points_b: np.ndarray) -> float:
    a = np.asarray(points_a, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(points_b, dtype=np.float64).reshape(-1, 2)
    if a.size == 0 or b.size == 0:
        return float("nan")
    return _mean_min_distance(a, b) + _mean_min_distance(b, a)


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

    pred_contour20_points = _contour_pixels(pred_contour, height, width)[pred_contour.valid_mask > 0.0]
    true_contour20_points = _contour_pixels(target_contour, height, width)[target_contour.valid_mask > 0.0]
    pred_zero_points = _zero_contour_points(phi_pred)
    true_zero_points = _zero_contour_points(phi_target)
    return {
        "contour20_y_mae_px": y_mae,
        "contour20_symmetric_chamfer_px": _symmetric_chamfer(pred_contour20_points, true_contour20_points),
        "zero_contour_symmetric_chamfer_px": _symmetric_chamfer(pred_zero_points, true_zero_points),
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
    pred_mask = _material_mask(pred)
    target_mask = _material_mask(target)
    mask_metrics = dice_iou(pred_mask, target_mask)
    area_target = float(target_mask.sum())
    area_pred = float(pred_mask.sum())
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
    from epi_pinn.sdf import ensure_signed_distance

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