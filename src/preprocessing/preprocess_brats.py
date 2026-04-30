"""
preprocess_brats.py
-------------------
Preprocesa BraTS a data/processed/positives/*.npz con label=1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .base_preprocessing import (
        DEFAULT_OUTPUT_DIR,
        REPO_ROOT,
        DatasetSample,
        PreprocessingConfig,
        process_dataset,
        write_preprocessing_summary,
    )
except ImportError:
    from base_preprocessing import (
        DEFAULT_OUTPUT_DIR,
        REPO_ROOT,
        DatasetSample,
        PreprocessingConfig,
        process_dataset,
        write_preprocessing_summary,
    )


DEFAULT_BRATS_DIR = REPO_ROOT / "data" / "raw" / "brats"


def first_existing(patient_dir: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(patient_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def build_brats_samples(input_dir: Path = DEFAULT_BRATS_DIR, limit: int | None = None) -> list[DatasetSample]:
    patient_dirs = sorted(path for path in Path(input_dir).iterdir() if path.is_dir())
    if limit is not None:
        patient_dirs = patient_dirs[:limit]

    samples: list[DatasetSample] = []
    missing: list[str] = []

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        t1_path = first_existing(
            patient_dir,
            [f"{patient_id}_t1w.nii.gz", f"{patient_id}_t1w.nii", f"{patient_id}_t1.nii.gz", f"{patient_id}_t1.nii", "*_t1w.nii*"],
        )
        t2_path = first_existing(
            patient_dir,
            [f"{patient_id}_t2w.nii.gz", f"{patient_id}_t2w.nii", f"{patient_id}_t2.nii.gz", f"{patient_id}_t2.nii", "*_t2w.nii*"],
        )

        if t1_path is None or t2_path is None:
            missing.append(patient_id)
            continue

        samples.append(
            DatasetSample(
                dataset="brats",
                subject_id=patient_id,
                t1_path=t1_path,
                t2_path=t2_path,
                label=1,
                output_subdir="positives",
            )
        )

    if missing:
        print(f"BraTS: {len(missing)} pacientes sin T1/T2 completos. Primeros: {missing[:10]}")
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocesa BraTS a .npz positivos.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_BRATS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_dir.exists():
        print(f"ERROR: no existe {args.input_dir}")
        sys.exit(1)

    config = PreprocessingConfig()
    samples = build_brats_samples(args.input_dir, limit=args.limit)
    print(f"BraTS: {len(samples)} pacientes con T1/T2")

    result = process_dataset(
        samples,
        output_root=args.output_dir,
        config=config,
        overwrite=args.overwrite,
        desc="BraTS (positivos)",
    )
    summary_path = write_preprocessing_summary(args.output_dir, config=config, run_results=[result])

    print(f"Procesados nuevos: {result.written}")
    print(f"Ya existentes    : {result.existing}")
    print(f"Errores          : {result.skipped}")
    print(f"Resumen          : {summary_path}")


if __name__ == "__main__":
    main()
