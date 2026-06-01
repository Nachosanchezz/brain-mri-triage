"""
preprocess_nki_rockland.py
--------------------------
Preprocesa NKI Rockland descargado en BIDS a .npz negativos.

Espera archivos:
  data/raw/nki_rockland/sub-*/ses-*/anat/*_T1w.nii.gz
  data/raw/nki_rockland/sub-*/ses-*/anat/*_T2w.nii.gz
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

# Silencia warnings de nnU-Net cuando se usa HD-BET en inferencia.
os.environ.setdefault("nnUNet_raw", "tmp_unused")
os.environ.setdefault("nnUNet_preprocessed", "tmp_unused")
os.environ.setdefault("nnUNet_results", "tmp_unused")


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


DEFAULT_NKI_DIR = REPO_ROOT / "data" / "raw" / "nki_rockland"
DEFAULT_NKI_STRIPPED_ROOT = REPO_ROOT / "data" / "raw" / "nki_rockland_stripped"


def subject_session_id(path: Path) -> str:
    match = re.search(r"sub-(A\d+)_ses-([A-Z0-9]+)", path.name)
    if match:
        return f"NKI-{match.group(1)}-{match.group(2)}"

    parts = path.parts
    subject = next((part.removeprefix("sub-") for part in parts if part.startswith("sub-")), path.stem)
    session = next((part.removeprefix("ses-") for part in parts if part.startswith("ses-")), "unknown")
    return f"NKI-{subject}-{session}"


def build_nki_samples(input_dir: Path = DEFAULT_NKI_DIR, limit: int | None = None) -> list[DatasetSample]:
    t1_files = {subject_session_id(path): path for path in Path(input_dir).glob("sub-*/ses-*/anat/*_T1w.nii.gz")}
    t2_files = {subject_session_id(path): path for path in Path(input_dir).glob("sub-*/ses-*/anat/*_T2w.nii.gz")}

    common_subjects = sorted(set(t1_files) & set(t2_files))
    if limit is not None:
        common_subjects = common_subjects[:limit]

    missing_t2 = sorted(set(t1_files) - set(t2_files))
    missing_t1 = sorted(set(t2_files) - set(t1_files))
    if missing_t1 or missing_t2:
        print(f"NKI Rockland: {len(missing_t1)} sesiones sin T1 y {len(missing_t2)} sesiones sin T2.")

    return [
        DatasetSample(
            dataset="nki_rockland",
            subject_id=subject_id,
            t1_path=t1_files[subject_id],
            t2_path=t2_files[subject_id],
            label=0,
            output_subdir="negatives",
        )
        for subject_id in common_subjects
    ]


def build_nki_stripped_samples(
    stripped_root: Path = DEFAULT_NKI_STRIPPED_ROOT,
    limit: int | None = None,
) -> list[DatasetSample]:
    t1_dir = stripped_root / "t1"
    t2_dir = stripped_root / "t2"
    t1_files = {path.name.replace("_T1w.nii.gz", ""): path for path in t1_dir.glob("*_T1w.nii.gz")}
    t2_files = {path.name.replace("_T2w.nii.gz", ""): path for path in t2_dir.glob("*_T2w.nii.gz")}

    common_subjects = sorted(set(t1_files) & set(t2_files))
    if limit is not None:
        common_subjects = common_subjects[:limit]

    return [
        DatasetSample(
            dataset="nki_rockland",
            subject_id=subject_id,
            t1_path=t1_files[subject_id],
            t2_path=t2_files[subject_id],
            label=0,
            output_subdir="negatives",
        )
        for subject_id in common_subjects
    ]


def skull_strip_nki(
    samples: list[DatasetSample],
    stripped_root: Path = DEFAULT_NKI_STRIPPED_ROOT,
) -> None:
    import torch
    from HD_BET.checkpoint_download import maybe_download_parameters
    from HD_BET.hd_bet_prediction import get_hdbet_predictor, hdbet_predict
    from tqdm import tqdm

    if not torch.cuda.is_available():
        print("ERROR: no hay GPU CUDA. HD-BET sobre NKI es demasiado lento sin GPU.")
        sys.exit(1)

    print("=" * 60)
    print("SKULL-STRIPPING NKI ROCKLAND CON HD-BET")
    print("=" * 60)
    maybe_download_parameters()
    predictor = get_hdbet_predictor(
        use_tta=False,
        device=torch.device("cuda"),
        verbose=False,
    )

    out_t1_dir = stripped_root / "t1"
    out_t2_dir = stripped_root / "t2"
    out_t1_dir.mkdir(parents=True, exist_ok=True)
    out_t2_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    skipped = 0
    for sample in tqdm(samples, desc="NKI HD-BET", unit="vol"):
        outputs = [
            (sample.t1_path, out_t1_dir / f"{sample.subject_id}_T1w.nii.gz"),
            (sample.t2_path, out_t2_dir / f"{sample.subject_id}_T2w.nii.gz"),
        ]
        for in_path, out_path in outputs:
            if out_path.exists():
                continue
            try:
                hdbet_predict(
                    str(in_path),
                    str(out_path),
                    predictor,
                    keep_brain_mask=False,
                    compute_brain_extracted_image=True,
                )
            except Exception as exc:
                skipped += 1
                print(f"\n  Error en {in_path.name}: {exc}")

    elapsed = time.time() - t0
    print(f"Volumenes stripped guardados en: {stripped_root}")
    print(f"Errores HD-BET: {skipped}")
    print(f"Tiempo total: {elapsed / 60:.1f} min")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocesa NKI Rockland a .npz negativos.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_NKI_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skull-strip", action="store_true", help="Ejecuta HD-BET antes de crear los .npz.")
    parser.add_argument("--stripped-root", type=Path, default=DEFAULT_NKI_STRIPPED_ROOT)
    parser.add_argument("--use-stripped", action="store_true", help="Usa NKI ya skull-stripped en stripped-root.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_dir.exists():
        print(f"ERROR: no existe {args.input_dir}")
        print("Ejecuta primero: python src/preprocessing/download_nki_rockland.py --aws-links <aws_links.csv>")
        sys.exit(1)

    config = PreprocessingConfig()
    raw_samples = build_nki_samples(args.input_dir, limit=args.limit)

    if args.skull_strip:
        if not raw_samples:
            print("ERROR: no hay pares T1/T2 raw para skull-strip.")
            sys.exit(1)
        skull_strip_nki(raw_samples, stripped_root=args.stripped_root)
        args.use_stripped = True

    samples = build_nki_stripped_samples(args.stripped_root, limit=args.limit) if args.use_stripped else raw_samples
    print(f"NKI Rockland: {len(samples)} sesiones con T1/T2")
    if not samples:
        print("ERROR: no hay pares T1/T2 para preprocesar.")
        sys.exit(1)

    result = process_dataset(
        samples,
        output_root=args.output_dir,
        config=config,
        overwrite=args.overwrite,
        desc="NKI Rockland (negativos)",
    )
    summary_path = write_preprocessing_summary(args.output_dir, config=config, run_results=[result])

    print(f"Procesados nuevos: {result.written}")
    print(f"Ya existentes    : {result.existing}")
    print(f"Errores          : {result.skipped}")
    print(f"Resumen          : {summary_path}")


if __name__ == "__main__":
    main()
