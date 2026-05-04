"""
preprocess_upenn.py
-------------------
Filtra pacientes UPENN-GBM con T1/T2 completos y los preprocesa a
data/processed/positives/*.npz con label=1.
"""

from __future__ import annotations

import argparse
from collections import Counter
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


DEFAULT_UPENN_DIR = REPO_ROOT / "data" / "raw" / "upenn"
DEFAULT_UPENN_MANIFEST_IN = Path(r"C:\Trabajo Fin Grado\manifest-1777282737360.tcia")
DEFAULT_UPENN_MANIFEST_OUT = Path(r"C:\Trabajo Fin Grado\manifest-upenn-t1-t2-only.tcia")
UPENN_REQUIRE_CAPTK = True
UPENN_T1_EXCLUDE = ["post", " gd", "c+", "gadolinium", "stealth-post", "flair"]
UPENN_T2_EXCLUDE = ["flair", "c+", "post", "tirm", "stir", "dark-fluid"]


def first_existing(patient_dir: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(patient_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def read_manifest(path: Path) -> tuple[list[str], list[str]]:
    """Lee un manifest TCIA y devuelve cabecera + UIDs."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    header: list[str] = []
    uids: list[str] = []
    in_list = False

    for line in lines:
        if line.startswith("ListOfSeriesToDownload="):
            header.append(line)
            in_list = True
        elif in_list:
            if line.strip():
                uids.append(line.strip())
        else:
            header.append(line)

    return header, uids


def write_manifest(path: Path, header: list[str], uids: list[str]) -> None:
    """Escribe un manifest TCIA filtrado."""
    Path(path).write_text("\n".join(header + uids) + "\n", encoding="utf-8")


def classify_upenn_series_description(description: str) -> str | None:
    """Devuelve 'T1', 'T2' o None segun SeriesDescription."""
    desc = description.lower()

    if UPENN_REQUIRE_CAPTK and "processed_captk" not in desc:
        return None

    has_t1 = "t1" in desc
    has_t2 = "t2" in desc

    if has_t1 and not any(excluded in desc for excluded in UPENN_T1_EXCLUDE):
        return "T1"
    if has_t2 and not any(excluded in desc for excluded in UPENN_T2_EXCLUDE):
        return "T2"
    return None


def filter_upenn_manifest(manifest_in: Path, manifest_out: Path, assume_yes: bool = False) -> None:
    """Filtra un manifest TCIA UPENN-GBM para conservar solo series T1/T2 CaPTk."""
    try:
        from tcia_utils import nbia
    except ImportError:
        print("ERROR: tcia_utils no instalado. Ejecuta: pip install tcia_utils")
        sys.exit(1)

    header, all_uids = read_manifest(manifest_in)
    print(f"Series en manifest original: {len(all_uids)}")

    print("Consultando metadata de series UPENN-GBM...")
    all_series = nbia.getSeries(collection="UPENN-GBM")

    uid_to_desc: dict[str, str] = {}
    for series in all_series:
        uid = series.get("SeriesInstanceUID", "")
        desc = series.get("SeriesDescription", series.get("Modality", "?"))
        uid_to_desc[uid] = desc

    desc_counts = Counter(uid_to_desc.values())
    print(f"\nDescripciones de series en UPENN-GBM ({len(desc_counts)} unicas):")
    for desc, count in sorted(desc_counts.items(), key=lambda item: -item[1]):
        keep = classify_upenn_series_description(desc)
        mark = f"  [MANTENER como {keep}]" if keep else ""
        print(f"  {count:4d}  {desc}{mark}")

    kept_t1: list[str] = []
    kept_t2: list[str] = []
    discarded: list[tuple[str, str]] = []

    for uid in all_uids:
        desc = uid_to_desc.get(uid, "")
        tag = classify_upenn_series_description(desc)
        if tag == "T1":
            kept_t1.append(uid)
        elif tag == "T2":
            kept_t2.append(uid)
        else:
            discarded.append((uid, desc))

    print("\nResumen del filtrado:")
    print(f"  T1 mantenidas : {len(kept_t1)}")
    print(f"  T2 mantenidas : {len(kept_t2)}")
    print(f"  Descartadas   : {len(discarded)}")
    if discarded[:5]:
        print(f"  Ejemplos descartados: {[desc for _, desc in discarded[:5]]}")

    filtered_uids = kept_t1 + kept_t2
    if not filtered_uids:
        print("\nATENCION: no se encontraron series T1/T2.")
        sys.exit(1)

    if not assume_yes:
        confirm = input(f"\nGuardar manifest con {len(filtered_uids)} series en {manifest_out}? [s/n]: ")
        if confirm.strip().lower() != "s":
            print("Cancelado.")
            return

    write_manifest(manifest_out, header, filtered_uids)
    print(f"\nManifest guardado: {manifest_out}")
    print("Ahora abre TCIA Data Retriever y carga ese archivo.")


def filter_upenn_patient_dirs(input_dir: Path = DEFAULT_UPENN_DIR) -> tuple[list[tuple[str, Path, Path]], list[str]]:
    patient_dirs = sorted(path for path in Path(input_dir).iterdir() if path.is_dir())
    complete: list[tuple[str, Path, Path]] = []
    missing: list[str] = []

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        t1_path = first_existing(patient_dir, [f"{patient_id}_t1.nii.gz", f"{patient_id}_t1.nii", "*_t1.nii*", "*T1*.nii*"])
        t2_path = first_existing(patient_dir, [f"{patient_id}_t2.nii.gz", f"{patient_id}_t2.nii", "*_t2.nii*", "*T2*.nii*"])

        if t1_path is None or t2_path is None:
            missing.append(patient_id)
            continue

        complete.append((patient_id, t1_path, t2_path))

    return complete, missing


def build_upenn_samples(input_dir: Path = DEFAULT_UPENN_DIR, limit: int | None = None) -> list[DatasetSample]:
    complete, missing = filter_upenn_patient_dirs(input_dir)
    if limit is not None:
        complete = complete[:limit]

    samples: list[DatasetSample] = []
    for patient_id, t1_path, t2_path in complete:
        samples.append(
            DatasetSample(
                dataset="upenn",
                subject_id=patient_id,
                t1_path=t1_path,
                t2_path=t2_path,
                label=1,
                output_subdir="positives",
            )
        )

    if missing:
        print(f"UPENN: {len(missing)} pacientes sin T1/T2 completos. Primeros: {missing[:10]}")
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocesa UPENN-GBM a .npz positivos.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_UPENN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--filter-manifest", action="store_true", help="Filtra el manifest TCIA de UPENN y termina.")
    parser.add_argument("--manifest-in", type=Path, default=DEFAULT_UPENN_MANIFEST_IN)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_UPENN_MANIFEST_OUT)
    parser.add_argument("--yes", action="store_true", help="No pide confirmacion al filtrar manifest.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.filter_manifest:
        filter_upenn_manifest(args.manifest_in, args.manifest_out, assume_yes=args.yes)
        return

    if not args.input_dir.exists():
        print(f"ERROR: no existe {args.input_dir}")
        print("Ejecuta primero: python src/preprocessing/conversion/dicom_to_nifti.py upenn")
        sys.exit(1)

    config = PreprocessingConfig()
    samples = build_upenn_samples(args.input_dir, limit=args.limit)
    print(f"UPENN: {len(samples)} pacientes con T1/T2")

    result = process_dataset(
        samples,
        output_root=args.output_dir,
        config=config,
        overwrite=args.overwrite,
        desc="UPENN (test externo)",
    )
    summary_path = write_preprocessing_summary(args.output_dir, config=config, run_results=[result])

    print(f"Procesados nuevos: {result.written}")
    print(f"Ya existentes    : {result.existing}")
    print(f"Errores          : {result.skipped}")
    print(f"Resumen          : {summary_path}")


if __name__ == "__main__":
    main()
