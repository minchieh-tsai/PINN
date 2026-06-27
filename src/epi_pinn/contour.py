"""Twenty-point zero-contour extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from epi_pinn.geometry import pixel_to_normalized_xy


@dataclass(frozen=True)
class ContourCondition:
    points_xy: np.ndarray
    valid_mask: np.ndarray

    def as_features(self) -> np.ndarray:
        points = np.asarray(self.points_xy, dtype=np.float64)
        mask = np.asarray(self.valid_mask, dtype=np.float64).reshape(-1, 1)
        return np.concatenate([points, mask], axis=1)


def _column_crossings(column: np.ndarray, epsilon: float = 1.0e-12) -> List[float]:
    values = np.asarray(column, dtype=np.float64)
    crossings: List[float] = []
    exact = np.where(values == 0.0)[0]
    crossings.extend(float(index) for index in exact.tolist())
    for y_index in range(values.size - 1):
        a = values[y_index]
        b = values[y_index + 1]
        if a == 0.0 or b == 0.0:
            continue
        if a * b < 0.0:
            crossings.append(float(y_index) - a / (b - a + epsilon))
    return sorted(crossings)


def extract_contour20(
    phi: np.ndarray,
    num_points: int = 20,
    min_valid_points: int = 10,
    crossing_policy: str = "closest_to_previous",
    first_crossing_policy: str = "topmost",
) -> ContourCondition:
    array = np.asarray(phi, dtype=np.float64)
    height, width = array.shape
    x_positions = np.linspace(0.0, float(width - 1), num_points, dtype=np.float64)
    y_positions = np.zeros(num_points, dtype=np.float64)
    valid_mask = np.zeros(num_points, dtype=np.float64)
    previous_y: Optional[float] = None

    for index, x_value in enumerate(x_positions):
        x_index = int(round(float(x_value)))
        crossings = _column_crossings(array[:, x_index])
        if not crossings:
            continue

        if previous_y is None:
            if first_crossing_policy != "topmost":
                raise ValueError(f"Unsupported first_crossing_policy: {first_crossing_policy}")
            y_value = crossings[0]
        elif crossing_policy == "closest_to_previous":
            y_value = min(crossings, key=lambda item: abs(item - previous_y))
        else:
            raise ValueError(f"Unsupported crossing_policy: {crossing_policy}")

        y_positions[index] = y_value
        valid_mask[index] = 1.0
        previous_y = y_value

    valid_count = int(valid_mask.sum())
    if valid_count < min_valid_points:
        raise ValueError(
            f"Only {valid_count} valid contour points found; "
            f"minimum required is {min_valid_points}"
        )

    xi, eta = pixel_to_normalized_xy(x_positions, y_positions, height, width)
    return ContourCondition(
        points_xy=np.ascontiguousarray(np.stack([xi, eta], axis=1), dtype=np.float64),
        valid_mask=np.ascontiguousarray(valid_mask, dtype=np.float64),
    )


def interpolate_contour_eta(contour: ContourCondition, xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    features = contour.as_features()
    valid = features[:, 2] > 0.0
    query = np.asarray(xi, dtype=np.float64)
    if valid.sum() < 2:
        return np.zeros_like(query), np.zeros_like(query)
    eta = np.interp(query, features[valid, 0], features[valid, 1])
    mask = (query >= features[valid, 0].min()) & (query <= features[valid, 0].max())
    return eta, mask.astype(np.float64)


def save_contour_csv(contour: ContourCondition, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "xi": contour.points_xy[:, 0],
            "eta": contour.points_xy[:, 1],
            "valid_mask": contour.valid_mask,
        }
    )
    frame.to_csv(path, index=False)
