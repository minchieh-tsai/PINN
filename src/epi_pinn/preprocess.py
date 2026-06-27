"""Preprocessing pipeline for configured level-set states."""

from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np
import pandas as pd

from epi_pinn.config import load_config, output_dir, project_root_from_config_path, save_config
from epi_pinn.contour import extract_contour20, save_contour_csv
from epi_pinn.excel_io import inspect_configured_workbooks, load_state_arrays
from epi_pinn.sdf import ensure_signed_distance


def inspect_xlsx(config_path: str) -> Dict[str, Any]:
    config = load_config(config_path)
    root = project_root_from_config_path(config_path)
    return inspect_configured_workbooks(config, base_dir=root)


def run_preprocess(config_path: str, split: str = "train") -> Dict[str, Any]:
    if split not in ("train", "holdout", "all"):
        raise ValueError("split must be one of: train, holdout, all")

    config = load_config(config_path)
    root = project_root_from_config_path(config_path)
    out_dir = output_dir(config, root)
    preprocess_dir = out_dir / "preprocess"
    contours_dir = out_dir / "contours"
    processed_dir = root / "data" / "processed"
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    contours_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    states = load_state_arrays(config, base_dir=root)
    level_config = config.get("level_set", {})
    contour_config = config.get("contour", {})

    if split == "train":
        selected = ["init", "1M", "1E", "2M", "2E"]
    elif split == "holdout":
        selected = ["5M", "5E"]
    else:
        selected = list(states.keys())

    rows = []
    for state_name in selected:
        phi = ensure_signed_distance(states[state_name], level_config)
        np.save(processed_dir / f"{state_name}.npy", phi)
        row = {
            "state": state_name,
            "shape": list(phi.shape),
            "min": float(np.min(phi)),
            "max": float(np.max(phi)),
            "mean": float(np.mean(phi)),
            "finite": bool(np.isfinite(phi).all()),
        }
        try:
            contour = extract_contour20(
                phi,
                num_points=int(contour_config.get("num_points", 20)),
                min_valid_points=int(contour_config.get("min_valid_points", 10)),
                crossing_policy=str(contour_config.get("crossing_policy", "closest_to_previous")),
                first_crossing_policy=str(contour_config.get("first_crossing_policy", "topmost")),
            )
            save_contour_csv(contour, str(contours_dir / f"{state_name}_contour20.csv"))
            row["valid_contour_points"] = int(contour.valid_mask.sum())
        except Exception as exc:
            row["contour_error"] = str(exc)
            row["valid_contour_points"] = 0
        rows.append(row)

    summary = {"split": split, "states": rows}
    with (preprocess_dir / "data_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    pd.DataFrame(rows).to_csv(preprocess_dir / "contour_summary.csv", index=False)
    save_config(config, str(out_dir / "resolved_config.yaml"))
    return summary