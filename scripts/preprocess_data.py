#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def add_src_path() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess configured level-set states.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", choices=["train", "holdout", "all"], default="train")
    args = parser.parse_args()
    add_src_path()
    from epi_pinn.preprocess import run_preprocess

    summary = run_preprocess(args.config, split=args.split)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())