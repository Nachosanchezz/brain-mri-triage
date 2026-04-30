"""
summarize_processed.py
----------------------
Reconstruye data/processed/preprocessing_summary.json leyendo los .npz
existentes y separando los conteos por dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .base_preprocessing import DEFAULT_OUTPUT_DIR, PreprocessingConfig, scan_processed, write_preprocessing_summary
except ImportError:
    from base_preprocessing import DEFAULT_OUTPUT_DIR, PreprocessingConfig, scan_processed, write_preprocessing_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume los .npz de data/processed por dataset y clase.")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--write", action="store_true", help="Actualiza preprocessing_summary.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scan = scan_processed(args.processed_dir)

    print(json.dumps(scan, indent=2))

    if args.write:
        summary_path = write_preprocessing_summary(args.processed_dir, config=PreprocessingConfig())
        print(f"\nResumen actualizado: {summary_path}")


if __name__ == "__main__":
    main()
