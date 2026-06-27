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
    parser = argparse.ArgumentParser(description="Inspect configured EPI level-set XLSX workbooks.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    add_src_path()
    from epi_pinn.preprocess import inspect_xlsx

    report = inspect_xlsx(args.config)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())