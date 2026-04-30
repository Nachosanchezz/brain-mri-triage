"""
convert_dicom_to_nifti_upenn.py
--------------------------------
Convierte los DICOMs de UPENN-GBM (descargados via TCIA Data Retriever)
a NIfTI, identificando T1 y T2 por el campo SeriesDescription del header DICOM.

Los DICOMs están organizados en carpetas con nombres de UID (StudyUID/SeriesUID),
por eso se necesita leer el header para saber qué modalidad es cada serie.
Solo se convierten series con "Processed_CaPTk" en la descripción (skull-stripped).

Uso:
    python src/preprocessing/convert_dicom_to_nifti_upenn.py

Requiere dcm2niix en el PATH y pydicom instalado.
"""

import sys
import shutil
import subprocess
import pydicom
from pathlib import Path
from tqdm import tqdm


# ============================================================
# CONFIGURACIÓN — Ajusta estas rutas a tu máquina
# ============================================================
DICOM_ROOT  = Path(r"C:\Users\1cnac\Downloads\upenn_gbm")
OUTPUT_ROOT = Path(r"C:\Trabajo Fin Grado\brain-mri-triage\data\raw\upenn")
# ============================================================


def find_dcm2niix():
    """Busca el ejecutable dcm2niix en el PATH."""
    path = shutil.which("dcm2niix")
    if path is not None:
        return path

    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "env" / "Scripts" / "dcm2niix.exe",
        repo_root / "tools" / "dcm2niix.exe",
        repo_root / "env" / "Lib" / "site-packages" / "dcm2niix" / "dcm2niix.exe",
        repo_root / "env" / "Lib" / "site-packages" / "bin" / "dcm2niix.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    print("ERROR: dcm2niix no encontrado en el PATH ni en el entorno local.")
    print("Instalalo en el venv con: pip install dcm2niix")
    sys.exit(1)


def classify_series(series_dir):
    """Devuelve 't1', 't2' o None según el SeriesDescription del primer DICOM."""
    dcm_files = sorted(series_dir.glob("*.dcm"))
    if not dcm_files:
        return None
    ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
    desc = getattr(ds, "SeriesDescription", "").lower()
    # Solo series procesadas por CaPTk (skull-stripped y registradas)
    if "processed_captk" not in desc:
        return None
    if "t1" in desc and "t2" not in desc:
        return "t1"
    if "t2" in desc and "t1" not in desc:
        return "t2"
    return None


def convert_patient(dcm2niix_path, patient_id, patient_dir, output_dir):
    """Convierte las series T1 y T2 de un paciente a NIfTI."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Identificar qué carpeta de serie corresponde a T1 y cuál a T2
    series_map = {}  # {'t1': Path, 't2': Path}
    for study_dir in sorted(patient_dir.iterdir()):
        if not study_dir.is_dir():
            continue
        for series_dir in sorted(study_dir.iterdir()):
            if not series_dir.is_dir():
                continue
            tag = classify_series(series_dir)
            if tag and tag not in series_map:
                series_map[tag] = series_dir

    if "t1" not in series_map or "t2" not in series_map:
        return False, f"serie T1 o T2 no encontrada (encontradas: {list(series_map.keys())})"

    for modality, series_dir in series_map.items():
        out_file = output_dir / f"{patient_id}_{modality}.nii.gz"
        if out_file.exists():
            continue

        tmp_dir = output_dir / f"_tmp_{modality}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(exist_ok=True)

        cmd = [
            dcm2niix_path,
            "-z", "y",           # compresión gzip
            "-f", f"{patient_id}_{modality}",
            "-o", str(tmp_dir),
            str(series_dir),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            shutil.rmtree(tmp_dir)
            return False, f"dcm2niix fallo para {modality}: {result.stderr[:300]}"

        nifti_files = list(tmp_dir.glob("*.nii.gz"))
        if not nifti_files:
            shutil.rmtree(tmp_dir)
            return False, f"dcm2niix no generó NIfTI para {modality}"

        # Si dcm2niix genera más de un fichero (multi-echo, etc.), usar el primero
        shutil.move(str(nifti_files[0]), str(out_file))
        shutil.rmtree(tmp_dir)

    return True, "ok"


def main():
    dcm2niix_path = find_dcm2niix()

    if not DICOM_ROOT.exists():
        print(f"ERROR: No se encuentra {DICOM_ROOT}")
        sys.exit(1)

    patient_dirs = sorted([d for d in DICOM_ROOT.iterdir() if d.is_dir()])
    print(f"Pacientes encontrados: {len(patient_dirs)}")
    print(f"Salida: {OUTPUT_ROOT}")

    processed = 0
    skipped = []

    for patient_dir in tqdm(patient_dirs, desc="UPENN-GBM (conversión)"):
        patient_id = patient_dir.name
        output_dir = OUTPUT_ROOT / patient_id

        t1_ok = (output_dir / f"{patient_id}_t1.nii.gz").exists()
        t2_ok = (output_dir / f"{patient_id}_t2.nii.gz").exists()
        if t1_ok and t2_ok:
            processed += 1
            continue

        ok, msg = convert_patient(dcm2niix_path, patient_id, patient_dir, output_dir)
        if ok:
            processed += 1
        else:
            print(f"\n  Saltado {patient_id}: {msg}")
            skipped.append(patient_id)

    print(f"\nConvertidos : {processed}")
    if skipped:
        print(f"Saltados    : {len(skipped)} → {skipped[:10]}")
    print(f"NIfTI en    : {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
