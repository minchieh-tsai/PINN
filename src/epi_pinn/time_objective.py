"""CMA-ES time objective with user-extensible penalty placeholders."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Mapping

import numpy as np

from epi_pinn.evaluate import evaluate_pair


PENALTY_PLACEHOLDERS_ACTIVE = True


@dataclass(frozen=True)
class ScoreWeights:
    chamfer: float = 1.0
    merge: float = 1.0
    down: float = 1.0

    def validated(self) -> "ScoreWeights":
        for name, value in asdict(self).items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} weight must be finite and nonnegative")
        return self


@dataclass(frozen=True)
class ScoreBreakdown:
    chamfer_4m_vs_5m: float
    chamfer_4e_vs_5e: float
    chamfer_score: float
    merge_penalty: float
    down_penalty: float
    chamfer_weight: float
    merge_weight: float
    down_weight: float
    weighted_chamfer: float
    weighted_merge: float
    weighted_down: float
    total_score: float

    def to_dict(self) -> Dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


def merge_penalty(
    predictions: Mapping[str, np.ndarray],
    states: Mapping[str, np.ndarray],
    times_s: Mapping[str, float],
    config: Mapping[str, Any],
) -> float:
    # TO-DO: Replace this placeholder with the user-defined merge penalty.
    return 0.0


def down_penalty(
    predictions: Mapping[str, np.ndarray],
    states: Mapping[str, np.ndarray],
    times_s: Mapping[str, float],
    config: Mapping[str, Any],
) -> float:
    # TO-DO: Replace this placeholder with the user-defined down penalty.
    return 0.0


def _nonnegative_finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return number


def evaluate_time_objective(
    predictions: Mapping[str, np.ndarray],
    states: Mapping[str, np.ndarray],
    times_s: Mapping[str, float],
    config: Mapping[str, Any],
    weights: ScoreWeights,
) -> ScoreBreakdown:
    weights = weights.validated()
    for state in ("4M", "4E"):
        if state not in predictions:
            raise KeyError(f"Missing prediction state: {state}")
        if not np.isfinite(np.asarray(predictions[state], dtype=np.float64)).all():
            raise ValueError(f"prediction {state} contains NaN or Inf")
    for state in ("5M", "5E"):
        if state not in states:
            raise KeyError(f"Missing target state: {state}")

    metrics_4m = evaluate_pair(predictions["4M"], states["5M"], config)
    metrics_4e = evaluate_pair(predictions["4E"], states["5E"], config)
    chamfer_4m = _nonnegative_finite(
        metrics_4m["zero_contour_symmetric_chamfer_px"], "4M-vs-5M Chamfer"
    )
    chamfer_4e = _nonnegative_finite(
        metrics_4e["zero_contour_symmetric_chamfer_px"], "4E-vs-5E Chamfer"
    )
    merge_value = _nonnegative_finite(
        merge_penalty(predictions, states, times_s, config), "merge penalty"
    )
    down_value = _nonnegative_finite(
        down_penalty(predictions, states, times_s, config), "down penalty"
    )
    chamfer_score = chamfer_4m + chamfer_4e
    weighted_chamfer = weights.chamfer * chamfer_score
    weighted_merge = weights.merge * merge_value
    weighted_down = weights.down * down_value
    total = _nonnegative_finite(
        weighted_chamfer + weighted_merge + weighted_down, "total score"
    )
    return ScoreBreakdown(
        chamfer_4m_vs_5m=chamfer_4m,
        chamfer_4e_vs_5e=chamfer_4e,
        chamfer_score=chamfer_score,
        merge_penalty=merge_value,
        down_penalty=down_value,
        chamfer_weight=weights.chamfer,
        merge_weight=weights.merge,
        down_weight=weights.down,
        weighted_chamfer=weighted_chamfer,
        weighted_merge=weighted_merge,
        weighted_down=weighted_down,
        total_score=total,
    )