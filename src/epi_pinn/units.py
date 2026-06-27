"""Constants and small validation helpers."""

from __future__ import annotations

EXPECTED_HEIGHT = 350
EXPECTED_WIDTH = 200
EXPECTED_SHAPE = (EXPECTED_HEIGHT, EXPECTED_WIDTH)
EPS = 1.0e-12


def require_positive(value: float, name: str) -> float:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return value


def require_finite_positive(value: float, name: str) -> float:
    import math

    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return value
