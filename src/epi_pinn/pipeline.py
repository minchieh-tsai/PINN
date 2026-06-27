"""Convenience pipeline functions used by CLI scripts."""

from __future__ import annotations

from epi_pinn.evaluate import evaluate_holdout
from epi_pinn.preprocess import inspect_xlsx, run_preprocess
from epi_pinn.rollout import run_rollout
from epi_pinn.train import train_process


__all__ = [
    "evaluate_holdout",
    "inspect_xlsx",
    "run_preprocess",
    "run_rollout",
    "train_process",
]