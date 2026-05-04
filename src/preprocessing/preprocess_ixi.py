"""
preprocess_ixi.py
-----------------
Skull-strip opcional de IXI con HD-BET y preprocesado final a
data/processed/negatives/*.npz con label=0.
"""

from __future__ import annotations

import argparse
import os
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


DEFAULT_IXI_T1_DIR = REPO_ROOT / "data" / "raw" / "ixi_stripped" / "t1"
DEFAULT_IXI_T2_DIR = REPO_ROOT / "data" / "raw" / "ixi_stripped" / "t2"
DEFAULT_IXI_RAW_T1_DIR = Path(r"C:\Trabajo Fin Grado\IXI-T1")
DEFAULT_IXI_RAW_T2_DIR = Path(r"C:\Trabajo Fin Grado\IXI-T2")
DEFAULT_IXI_STRIPPED_ROOT = REPO_ROOT / "data" / "raw" / "ixi_stripped"


def skull_strip_folder(input_dir: Path, output_dir: Path, predictor, name: str) -> tuple[int, int]:
    """Aplica HD-BET a un directorio IXI, saltando salidas ya existentes."""
    from HD_BET.hd_bet_prediction import hdbet_predict
    from tqdm import tqdm

    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(input_dir).glob("*.nii.gz"))
    print(f"\n{name}: {len(files)} volumenes encontrados")

    already_done = sum(1 for path in files if (output_dir / path.name).exists())
    if already_done:
        print(f"  {already_done} ya procesados, se saltan")

    skipped: list[str] = []
    for path in tqdm(files, desc=name, unit="vol"):
        out_path = output_dir / path.name
        if out_path.exists():
            continue
        try:
            hdbet_predict(
                str(path),
                str(out_path),
                predictor,
                keep_brain_mask=False,
                compute_brain_extracted_image=True,
            )
        except Exception as exc:
            print(f"\n  Error procesando {path.name}: {exc}")
            skipped.append(path.name)

    processed = len(files) - len(skipped)
    print(f"  Procesados: {processed}/{len(files)}")
    if skipped:
        print(f"  Saltados: {len(skipped)} -> {skipped[:5]}")
    return processed, len(skipped)


def skull_strip_ixi(
    raw_t1_dir: Path = DEFAULT_IXI_RAW_T1_DIR,
    raw_t2_dir: Path = DEFAULT_IXI_RAW_T2_DIR,
    stripped_root: Path = DEFAULT_IXI_STRIPPED_ROOT,
) -> None:
    """Genera data/raw/ixi_stripped/{t1,t2} usando HD-BET."""
    import torch
    from HD_BET.checkpoint_download import maybe_download_parameters
    from HD_BET.hd_bet_prediction import get_hdbet_predictor

    print("=" * 60)
    print("SKULL-STRIPPING IXI CON HD-BET")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: no hay GPU CUDA. HD-BET en IXI es demasiado lento sin GPU.")
        sys.exit(1)

    for path, name in [(raw_t1_dir, "IXI raw T1"), (raw_t2_dir, "IXI raw T2")]:
        if not path.exists():
            print(f"ERROR: no existe {name}: {path}")
            sys.exit(1)

    print("Descargando parametros HD-BET si hace falta...")
    maybe_download_parameters()

    print("Creando predictor HD-BET...")
    predictor = get_hdbet_predictor(
        use_tta=False,
        device=torch.device("cuda"),
        verbose=False,
    )

    t0 = time.time()
    skull_strip_folder(raw_t1_dir, stripped_root / "t1", predictor, "IXI T1")
    skull_strip_folder(raw_t2_dir, stripped_root / "t2", predictor, "IXI T2")
    elapsed = time.time() - t0

    print()
    print("=" * 60)
    print(f"Tiempo total: {elapsed / 60:.1f} min")
    print(f"Volumenes stripped guardados en: {stripped_root}")
    print("=" * 60)


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
    parser.add_argument("--skull-strip", action="store_true", help="Ejecuta HD-BET antes de crear los .npz.")
    parser.add_argument("--raw-t1-dir", type=Path, default=DEFAULT_IXI_RAW_T1_DIR)
    parser.add_argument("--raw-t2-dir", type=Path, default=DEFAULT_IXI_RAW_T2_DIR)
    parser.add_argument("--stripped-root", type=Path, default=DEFAULT_IXI_STRIPPED_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.skull_strip:
        skull_strip_ixi(args.raw_t1_dir, args.raw_t2_dir, args.stripped_root)
        args.t1_dir = args.stripped_root / "t1"
        args.t2_dir = args.stripped_root / "t2"

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
