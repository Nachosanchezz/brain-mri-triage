"""
preprocess_ixi.py
-----------------
Preprocesa IXI skull-stripped a data/processed/negatives/*.npz con label=0.
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


DEFAULT_IXI_T1_DIR = REPO_ROOT / "data" / "raw" / "ixi_stripped" / "t1"
DEFAULT_IXI_T2_DIR = REPO_ROOT / "data" / "raw" / "ixi_stripped" / "t2"


def subject_id_from_modality_file(path: Path, modality: str) -> str:
    stem = path.name
    for token in [f"-{modality}", f"_{modality}", f"-{modality.lower()}", f"_{modality.lower()}"]:
        if token in stem:
            return stem.split(token)[0]
    return path.name.replace(".nii.gz", "").replace(".nii", "")


def build_ixi_samples(
    t1_dir: Path = DEFAULT_IXI_T1_DIR,
    t2_dir: Path = DEFAULT_IXI_T2_DIR,
    limit: int | None = None,
) -> list[DatasetSample]:
    t1_files = {subject_id_from_modality_file(path, "T1"): path for path in Path(t1_dir).glob("*.nii*")}
    t2_files = {subject_id_from_modality_file(path, "T2"): path for path in Path(t2_dir).glob("*.nii*")}

    common_subjects = sorted(set(t1_files) & set(t2_files))
    if limit is not None:
        common_subjects = common_subjects[:limit]

    missing_t2 = sorted(set(t1_files) - set(t2_files))
    missing_t1 = sorted(set(t2_files) - set(t1_files))
    if missing_t1 or missing_t2:
        print(f"IXI: {len(missing_t1)} sujetos sin T1 y {len(missing_t2)} sujetos sin T2.")

    return [
        DatasetSample(
            dataset="ixi",
            subject_id=subject_id,
            t1_path=t1_files[subject_id],
            t2_path=t2_files[subject_id],
            label=0,
            output_subdir="negatives",
        )
        for subject_id in common_subjects
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocesa IXI a .npz negativos.")
    parser.add_argument("--t1-dir", type=Path, default=DEFAULT_IXI_T1_DIR)
    parser.add_argument("--t2-dir", type=Path, default=DEFAULT_IXI_T2_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path, name in [(args.t1_dir, "IXI T1"), (args.t2_dir, "IXI T2")]:
        if not path.exists():
            print(f"ERROR: no existe {name}: {path}")
            sys.exit(1)

    config = PreprocessingConfig()
    samples = build_ixi_samples(args.t1_dir, args.t2_dir, limit=args.limit)
    print(f"IXI: {len(samples)} sujetos con T1/T2")

    result = process_dataset(
        samples,
        output_root=args.output_dir,
        config=config,
        overwrite=args.overwrite,
        desc="IXI (negativos)",
    )
    summary_path = write_preprocessing_summary(args.output_dir, config=config, run_results=[result])

    print(f"Procesados nuevos: {result.written}")
    print(f"Ya existentes    : {result.existing}")
    print(f"Errores          : {result.skipped}")
    print(f"Resumen          : {summary_path}")


if __name__ == "__main__":
    main()
