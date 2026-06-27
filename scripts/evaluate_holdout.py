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
    parser = argparse.ArgumentParser(description="Evaluate 5M/5E holdout predictions.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    add_src_path()
    from epi_pinn.evaluate import evaluate_holdout

    metrics = evaluate_holdout(args.config)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())