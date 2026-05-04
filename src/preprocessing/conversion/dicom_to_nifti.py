"""
dicom_to_nifti.py
-----------------
Conversor comun DICOM -> NIfTI para datasets con estructura distinta.

Uso:
    python src/preprocessing/conversion/dicom_to_nifti.py brats
    python src/preprocessing/conversion/dicom_to_nifti.py upenn

BraTS espera carpetas de modalidad T1w/T2w dentro de cada paciente.
UPENN identifica T1/T2 leyendo SeriesDescription y usando series Processed_CaPTk.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pydicom
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_BRATS_DICOM_ROOT = Path(
    r"C:\Trabajo Fin Grado\PKG - RSNA-ASNR-MICCAI-BraTS-2021"
    r"\RSNA-ASNR-MICCAI-BraTS-2021\BraTS2021_TrainingSet_dcm"
)
DEFAULT_BRATS_OUTPUT_ROOT = REPO_ROOT / "data" / "raw" / "brats"

DEFAULT_UPENN_DICOM_ROOT = Path(r"C:\Users\1cnac\Downloads\upenn_gbm")
DEFAULT_UPENN_OUTPUT_ROOT = REPO_ROOT / "data" / "raw" / "upenn"

BRATS_MODALITIES = ("T1w", "T2w")


def find_dcm2niix() -> str:
    """Busca dcm2niix en PATH, en el venv local o en tools/."""
    path = shutil.which("dcm2niix")
    if path is not None:
        return path

    candidates = [
        REPO_ROOT / "env" / "Scripts" / "dcm2niix.exe",
        REPO_ROOT / "tools" / "dcm2niix.exe",
        REPO_ROOT / "env" / "Lib" / "site-packages" / "dcm2niix" / "dcm2niix.exe",
        REPO_ROOT / "env" / "Lib" / "site-packages" / "bin" / "dcm2niix.exe",
        Path(r"C:\Tools\dcm2niix\dcm2niix.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    print("ERROR: dcm2niix no encontrado en PATH ni en rutas locales.")
    print("Instalalo con: pip install dcm2niix")
    sys.exit(1)


def run_dcm2niix(
    dcm2niix_path: str,
    dicom_dir: Path,
    output_dir: Path,
    output_name: str,
    keep_sidecars: bool = False,
) -> tuple[bool, str]:
    """Ejecuta dcm2niix y normaliza el nombre de salida esperado."""
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_file = output_dir / f"{output_name}.nii.gz"
    if expected_file.exists():
        return True, "exists"

    tmp_dir = output_dir / f"_tmp_{output_name}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        dcm2niix_path,
        "-z",
        "y",
        "-f",
        output_name,
        "-o",
        str(tmp_dir),
    ]
    if not keep_sidecars:
        cmd.extend(["-b", "n"])
    cmd.append(str(dicom_dir))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False, f"dcm2niix fallo: {result.stderr[:300]}"

    nifti_files = sorted(tmp_dir.glob(f"{output_name}*.nii.gz"))
    if not nifti_files:
        nifti_files = sorted(tmp_dir.glob("*.nii.gz"))
    if not nifti_files:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False, "dcm2niix no genero NIfTI"

    shutil.move(str(nifti_files[0]), str(expected_file))
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return True, "ok"


def convert_brats_patient(
    dcm2niix_path: str,
    patient_id: str,
    dicom_patient_dir: Path,
    output_root: Path,
) -> list[str]:
    converted: list[str] = []
    output_dir = output_root / patient_id

    for modality in BRATS_MODALITIES:
        dicom_mod_dir = dicom_patient_dir / modality
        if not dicom_mod_dir.exists():
            print(f"  WARNING {patient_id}/{modality}: carpeta no encontrada")
            continue

        output_name = f"{patient_id}_{modality.lower()}"
        ok, msg = run_dcm2niix(
            dcm2niix_path=dcm2niix_path,
            dicom_dir=dicom_mod_dir,
            output_dir=output_dir,
            output_name=output_name,
            keep_sidecars=False,
        )
        if ok:
            converted.append(modality)
        else:
            print(f"  ERROR {patient_id}/{modality}: {msg}")

    return converted


def convert_brats(dicom_root: Path, output_root: Path, dcm2niix_path: str) -> None:
    if not dicom_root.exists():
        print(f"ERROR: no existe {dicom_root}")
        sys.exit(1)

    output_root.mkdir(parents=True, exist_ok=True)
    institutions = sorted(path for path in dicom_root.iterdir() if path.is_dir())

    print(f"BraTS DICOM root: {dicom_root}")
    print(f"BraTS NIfTI root: {output_root}")
    print(f"Instituciones: {[path.name for path in institutions]}")

    total_patients = 0
    complete = 0
    failed: list[str] = []

    for institution_dir in institutions:
        patients = sorted(path for path in institution_dir.iterdir() if path.is_dir())
        print(f"\n--- {institution_dir.name}: {len(patients)} pacientes ---")

        for patient_dir in patients:
            patient_id = f"BraTS2021_{patient_dir.name}"
            converted = convert_brats_patient(dcm2niix_path, patient_id, patient_dir, output_root)

            if len(converted) == len(BRATS_MODALITIES):
                complete += 1
                print(f"  OK {patient_id}: {', '.join(converted)}")
            else:
                failed.append(patient_id)
                print(f"  WARNING {patient_id}: solo {converted}")

            total_patients += 1

    print_summary("BraTS", total_patients, complete, failed, output_root)


def first_dicom_file(series_dir: Path) -> Path | None:
    for path in sorted(series_dir.glob("*.dcm")):
        if path.is_file():
            return path
    return None


def classify_upenn_series(series_dir: Path) -> str | None:
    """Devuelve 't1', 't2' o None segun SeriesDescription."""
    dcm_file = first_dicom_file(series_dir)
    if dcm_file is None:
        return None

    ds = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
    desc = getattr(ds, "SeriesDescription", "").lower()

    if "processed_captk" not in desc:
        return None
    if "t1" in desc and "t2" not in desc:
        return "t1"
    if "t2" in desc and "t1" not in desc:
        return "t2"
    return None


def find_upenn_series(patient_dir: Path) -> dict[str, Path]:
    series_map: dict[str, Path] = {}

    for study_dir in sorted(path for path in patient_dir.iterdir() if path.is_dir()):
        for series_dir in sorted(path for path in study_dir.iterdir() if path.is_dir()):
            tag = classify_upenn_series(series_dir)
            if tag and tag not in series_map:
                series_map[tag] = series_dir

    return series_map


def convert_upenn_patient(
    dcm2niix_path: str,
    patient_id: str,
    patient_dir: Path,
    output_root: Path,
) -> tuple[bool, str]:
    output_dir = output_root / patient_id
    output_dir.mkdir(parents=True, exist_ok=True)

    t1_ok = (output_dir / f"{patient_id}_t1.nii.gz").exists()
    t2_ok = (output_dir / f"{patient_id}_t2.nii.gz").exists()
    if t1_ok and t2_ok:
        return True, "exists"

    series_map = find_upenn_series(patient_dir)
    if "t1" not in series_map or "t2" not in series_map:
        return False, f"serie T1 o T2 no encontrada (encontradas: {list(series_map.keys())})"

    for modality in ("t1", "t2"):
        ok, msg = run_dcm2niix(
            dcm2niix_path=dcm2niix_path,
            dicom_dir=series_map[modality],
            output_dir=output_dir,
            output_name=f"{patient_id}_{modality}",
            keep_sidecars=True,
        )
        if not ok:
            return False, f"{modality}: {msg}"

    return True, "ok"


def convert_upenn(dicom_root: Path, output_root: Path, dcm2niix_path: str) -> None:
    if not dicom_root.exists():
        print(f"ERROR: no existe {dicom_root}")
        sys.exit(1)

    patient_dirs = sorted(path for path in dicom_root.iterdir() if path.is_dir())
    print(f"UPENN DICOM root: {dicom_root}")
    print(f"UPENN NIfTI root: {output_root}")
    print(f"Pacientes encontrados: {len(patient_dirs)}")

    complete = 0
    failed: list[str] = []

    for patient_dir in tqdm(patient_dirs, desc="UPENN-GBM conversion"):
        patient_id = patient_dir.name
        ok, msg = convert_upenn_patient(dcm2niix_path, patient_id, patient_dir, output_root)
        if ok:
            complete += 1
        else:
            print(f"\n  Saltado {patient_id}: {msg}")
            failed.append(patient_id)

    print_summary("UPENN", len(patient_dirs), complete, failed, output_root)


def print_summary(dataset: str, total: int, complete: int, failed: list[str], output_root: Path) -> None:
    print()
    print("=" * 60)
    print(f"RESUMEN {dataset}")
    print(f"  Pacientes encontrados : {total}")
    print(f"  Conversiones completas: {complete}")
    print(f"  Pacientes con problema: {len(failed)}")
    if failed:
        print(f"  Primeros problemas    : {failed[:20]}")
    print(f"  NIfTI en              : {output_root}")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convierte DICOM a NIfTI para BraTS o UPENN.")
    parser.add_argument("dataset", choices=["brats", "upenn"], help="Dataset a convertir.")
    parser.add_argument("--dicom-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dcm2niix_path = find_dcm2niix()
    print(f"Usando dcm2niix: {dcm2niix_path}")

    if args.dataset == "brats":
        convert_brats(
            dicom_root=args.dicom_root or DEFAULT_BRATS_DICOM_ROOT,
            output_root=args.output_root or DEFAULT_BRATS_OUTPUT_ROOT,
            dcm2niix_path=dcm2niix_path,
        )
    elif args.dataset == "upenn":
        convert_upenn(
            dicom_root=args.dicom_root or DEFAULT_UPENN_DICOM_ROOT,
            output_root=args.output_root or DEFAULT_UPENN_OUTPUT_ROOT,
            dcm2niix_path=dcm2niix_path,
        )


if __name__ == "__main__":
    main()
