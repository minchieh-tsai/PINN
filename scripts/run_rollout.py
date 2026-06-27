#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def add_src_path() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 2E-to-5E rollout.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--infer-missing-rates", action="store_true")
    parser.add_argument("--allow-baseline-fallback", action="store_true")
    args = parser.parse_args()
    add_src_path()
    from epi_pinn.rollout import run_rollout

    predictions = run_rollout(
        args.config,
        infer_missing_rates=args.infer_missing_rates,
        allow_baseline_fallback=args.allow_baseline_fallback,
    )
    print("Saved predictions: " + ", ".join(sorted(predictions.keys())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())