#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def add_src_path() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the etch level-set PINN.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--infer-missing-rates", action="store_true")
    args = parser.parse_args()
    add_src_path()
    from epi_pinn.train import train_process

    checkpoint = train_process(args.config, "etch", infer_missing_rates=args.infer_missing_rates)
    print(f"Saved etch checkpoint: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())