"""
preprocess_volumes.py
---------------------
Lanzador conjunto para el preprocesado base de entrenamiento:

  - BraTS -> data/processed/positives/*.npz, label=1
  - IXI   -> data/processed/negatives/*.npz, label=0

Para anadir datasets nuevos, crea otro preprocess_<dataset>.py que construya
DatasetSample y llama a process_dataset desde base_preprocessing.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .base_preprocessing import DEFAULT_OUTPUT_DIR, PreprocessingConfig, process_dataset, write_preprocessing_summary
    from .preprocess_brats import DEFAULT_BRATS_DIR, build_brats_samples
    from .preprocess_ixi import DEFAULT_IXI_T1_DIR, DEFAULT_IXI_T2_DIR, build_ixi_samples
except ImportError:
    from base_preprocessing import DEFAULT_OUTPUT_DIR, PreprocessingConfig, process_dataset, write_preprocessing_summary
    from preprocess_brats import DEFAULT_BRATS_DIR, build_brats_samples
    from preprocess_ixi import DEFAULT_IXI_T1_DIR, DEFAULT_IXI_T2_DIR, build_ixi_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocesa BraTS+IXI a .npz homogeno.")
    parser.add_argument("--brats-dir", type=Path, default=DEFAULT_BRATS_DIR)
    parser.add_argument("--ixi-t1-dir", type=Path, default=DEFAULT_IXI_T1_DIR)
    parser.add_argument("--ixi-t2-dir", type=Path, default=DEFAULT_IXI_T2_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--datasets", nargs="+", choices=["brats", "ixi"], default=["brats", "ixi"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Limite por dataset para pruebas rapidas.")
    return parser.parse_args()


def ensure_exists(path: Path, name: str) -> None:
    if not path.exists():
        print(f"ERROR: no existe {name}: {path}")
        sys.exit(1)


def main() -> None:
    args = parse_args()
    config = PreprocessingConfig()
    results = []

    print("=" * 60)
    print("PREPROCESADO HOMOGENEO")
    print(f"  Shape objetivo  : {config.target_shape}")
    print(f"  Spacing objetivo: {config.target_spacing} mm")
    print(f"  Salida          : {args.output_dir}")
    print(f"  Datasets        : {', '.join(args.datasets)}")
    print("=" * 60)

    if "brats" in args.datasets:
        ensure_exists(args.brats_dir, "BraTS")
        samples = build_brats_samples(args.brats_dir, limit=args.limit)
        print(f"\nBraTS: {len(samples)} pacientes con T1/T2")
        results.append(
            process_dataset(
                samples,
                output_root=args.output_dir,
                config=config,
                overwrite=args.overwrite,
                desc="BraTS (positivos)",
            )
        )

    if "ixi" in args.datasets:
        ensure_exists(args.ixi_t1_dir, "IXI T1")
        ensure_exists(args.ixi_t2_dir, "IXI T2")
        samples = build_ixi_samples(args.ixi_t1_dir, args.ixi_t2_dir, limit=args.limit)
        print(f"\nIXI: {len(samples)} sujetos con T1/T2")
        results.append(
            process_dataset(
                samples,
                output_root=args.output_dir,
                config=config,
                overwrite=args.overwrite,
                desc="IXI (negativos)",
            )
        )

    summary_path = write_preprocessing_summary(args.output_dir, config=config, run_results=results)

    print()
    print("=" * 60)
    print("RESUMEN FINAL")
    for result in results:
        print(
            f"  {result.dataset:8s} label={result.label}: "
            f"{result.available} disponibles ({result.written} nuevos, {result.existing} existentes, {result.skipped} errores)"
        )
    print(f"  Summary: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
