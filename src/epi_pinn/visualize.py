"""Small plotting helpers for prediction artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def save_levelset_triplet(phi_true: np.ndarray, phi_pred: np.ndarray, output_path: str, title: Optional[str] = None) -> None:
    import matplotlib.pyplot as plt

    true = np.asarray(phi_true, dtype=np.float64)
    pred = np.asarray(phi_pred, dtype=np.float64)
    error = np.abs(pred - true)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    if title:
        fig.suptitle(title)
    for axis, data, label in zip(axes, [true, pred, error], ["ground truth", "prediction", "absolute error"]):
        image = axis.imshow(data, cmap="coolwarm")
        axis.set_title(label)
        axis.set_axis_off()
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=150)
    plt.close(fig)