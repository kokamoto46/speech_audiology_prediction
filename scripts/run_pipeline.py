#!/usr/bin/env python
"""Thin wrapper to run the FILS pipeline.

Usage:
  python scripts/run_pipeline.py --data path/to/your_data.csv

If --data is omitted, the pipeline searches common paths defined in src/fils_pipeline.py (DATA_PATHS).
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.fils_pipeline import main as pipeline_main  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=None, help="Path to input CSV/XLS/XLSX.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.data:
        sys.argv = [sys.argv[0], args.data]
    else:
        sys.argv = [sys.argv[0]]
    pipeline_main()


if __name__ == "__main__":
    main()
