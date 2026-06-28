#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROLLOUT_STATES = ["3M", "3E", "4M", "4E", "5M", "5E"]


def add_src_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    return root


def load_predictions(prediction_dir: Path, states: list[str]):
    import numpy as np

    predictions = {}
    missing = []
    for state in states:
        path = prediction_dir / f"{state}.npy"
        if path.exists():
            predictions[state] = np.load(path)
        else:
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing prediction files:\n" + "\n".join(missing))
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot rollout phi=0 contours for 3M through 5E.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--prediction-dir", default=None, help="Directory containing 3M.npy ... 5E.npy")
    parser.add_argument("--output", default=None, help="Output PNG path")
    args = parser.parse_args()

    root = add_src_path()
    from epi_pinn.config import load_config, output_dir, project_root_from_config_path
    from epi_pinn.excel_io import load_state_arrays
    from epi_pinn.sdf import ensure_signed_distance
    from epi_pinn.visualize import save_zero_contour_grid

    config = load_config(args.config)
    config_root = project_root_from_config_path(args.config)
    out_dir = output_dir(config, config_root)
    prediction_dir = Path(args.prediction_dir) if args.prediction_dir else out_dir / "predictions"
    if not prediction_dir.is_absolute():
        prediction_dir = root / prediction_dir
    output = Path(args.output) if args.output else out_dir / "figures" / "rollout_zero_contours_3M_to_5E.png"
    if not output.is_absolute():
        output = root / output

    predictions = load_predictions(prediction_dir, ROLLOUT_STATES)
    states = load_state_arrays(config, base_dir=config_root)
    level_cfg = config.get("level_set", {})
    gt_arrays = {}
    for state in ("5M", "5E"):
        if state in states:
            gt_arrays[state] = ensure_signed_distance(states[state], level_cfg)

    save_zero_contour_grid(
        predictions,
        str(output),
        ROLLOUT_STATES,
        gt_arrays=gt_arrays,
        title="Rollout prediction phi=0 contours (GT overlay for 5M/5E)",
    )
    print(f"Saved figure: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())