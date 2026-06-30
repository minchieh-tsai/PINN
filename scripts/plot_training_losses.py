#!/usr/bin/env python
"""Plot loss curves and weighted loss ratios from training logs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def add_src_path() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot raw loss curves, weighted loss curves, and weighted loss contribution ratios."
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config file.")
    parser.add_argument("--output", default=None, help="Output figure path. Defaults to artifacts/figures/training_loss_breakdown.png.")
    parser.add_argument("--log-dir", default=None, help="Directory containing *_training.csv logs. Defaults to artifacts/logs.")
    parser.add_argument(
        "--process",
        action="append",
        choices=["deposition", "etch"],
        dest="processes",
        help="Process to plot. Repeat to include both; omit to plot all available logs.",
    )
    args = parser.parse_args()

    add_src_path()
    from epi_pinn.training_plots import plot_training_losses

    output = plot_training_losses(
        args.config,
        output_path=args.output,
        processes=args.processes,
        log_dir=args.log_dir,
    )
    print(f"Saved training loss figure: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
