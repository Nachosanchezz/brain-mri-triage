"""
convert_dicom_to_nifti.py
-------------------------
Convierte los DICOMs del BraTS 2021 Training Set a NIfTI (.nii.gz),
extrayendo solo las modalidades T1w y T2w.

Uso:
    python convert_dicom_to_nifti.py

Antes de ejecutar:
    - Asegúrate de tener dcm2niix instalado (pip install dcm2niix o descarga manual)
    - Ajusta las rutas de abajo si es necesario
"""

import os
import subprocess
import sys
from pathlib import Path

# ============================================================
# CONFIGURACIÓN — Ajusta estas rutas a tu máquina
# ============================================================

# Carpeta raíz donde están los DICOMs del Training Set
DICOM_ROOT = Path(r"C:\Trabajo Fin Grado\PKG - RSNA-ASNR-MICCAI-BraTS-2021\RSNA-ASNR-MICCAI-BraTS-2021\BraTS2021_TrainingSet_dcm")

# Carpeta de salida para los NIfTI convertidos
OUTPUT_ROOT = Path(r"C:\Trabajo Fin Grado\brain-mri-triage\data\raw\brats")

# Modalidades que queremos extraer
MODALITIES = ["T1w", "T2w"]

# ============================================================


def find_dcm2niix():
    """Busca dcm2niix en el PATH o en ubicaciones comunes."""
    # Primero intenta encontrarlo en el PATH
    for name in ["dcm2niix", "dcm2niix.exe"]:
        result = subprocess.run(["where", name], capture_output=True, text=True)
        if result.returncode == 0:
            return name

    # Busca en ubicaciones comunes
    common_paths = [
        Path(r"C:\Trabajo Fin Grado\brain-mri-triage\tools\dcm2niix.exe"),
        Path(r"C:\Tools\dcm2niix\dcm2niix.exe"),
    ]
    for p in common_paths:
        if p.exists():
            return str(p)

    print("ERROR: No se encontró dcm2niix.")
    print("Instálalo con: pip install dcm2niix")
    print("O descárgalo de: https://github.com/rordenlab/dcm2niix/releases")
    sys.exit(1)


def convert_patient(dcm2niix_path, patient_id, institution, dicom_patient_dir):
    """Convierte las modalidades T1w y T2w de un paciente."""
    converted = []

    for modality in MODALITIES:
        dicom_mod_dir = dicom_patient_dir / modality
        if not dicom_mod_dir.exists():
            print(f"  ⚠ {patient_id}/{modality}: carpeta no encontrada, saltando")
            continue

        # Crear carpeta de salida: data/raw/brats/PATIENT_ID/
        output_dir = OUTPUT_ROOT / patient_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Nombre del fichero de salida: BraTS2021_XXXXX_t1w.nii.gz
        output_name = f"{patient_id}_{modality.lower()}"

        # Ejecutar dcm2niix
        cmd = [
            dcm2niix_path,
            "-z", "y",          # Comprimir con gzip (.nii.gz)
            "-f", output_name,  # Nombre del fichero de salida
            "-o", str(output_dir),  # Carpeta de salida
            "-b", "n",          # No generar fichero .bson/.json
            str(dicom_mod_dir)  # Carpeta DICOM de entrada
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Verificar que se creó el fichero
        expected_file = output_dir / f"{output_name}.nii.gz"
        if expected_file.exists():
            converted.append(modality)
        else:
            # A veces dcm2niix añade sufijos, buscar cualquier .nii.gz
            nii_files = list(output_dir.glob(f"{output_name}*.nii.gz"))
            if nii_files:
                # Renombrar al nombre esperado
                nii_files[0].rename(expected_file)
                converted.append(modality)
            else:
                print(f"  ✗ {patient_id}/{modality}: conversión fallida")
                if result.stderr:
                    print(f"    Error: {result.stderr[:200]}")

    return converted


def main():
    dcm2niix_path = find_dcm2niix()
    print(f"Usando dcm2niix: {dcm2niix_path}")
    print(f"DICOM root: {DICOM_ROOT}")
    print(f"Output root: {OUTPUT_ROOT}")
    print()

    # Verificar que la carpeta DICOM existe
    if not DICOM_ROOT.exists():
        print(f"ERROR: No se encuentra la carpeta {DICOM_ROOT}")
        sys.exit(1)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Recorrer todas las instituciones
    institutions = [d for d in DICOM_ROOT.iterdir() if d.is_dir()]
    print(f"Instituciones encontradas: {[i.name for i in institutions]}")
    print()

    total_patients = 0
    total_converted = 0
    failed_patients = []

    for institution_dir in sorted(institutions):
        institution = institution_dir.name
        patients = sorted([d for d in institution_dir.iterdir() if d.is_dir()])
        print(f"--- {institution}: {len(patients)} pacientes ---")

        for patient_dir in patients:
            # El ID del paciente en BraTS es el nombre de la carpeta (ej: "00000")
            # Lo prefijamos para que sea único
            patient_id = f"BraTS2021_{patient_dir.name}"

            converted = convert_patient(dcm2niix_path, patient_id, institution, patient_dir)

            if len(converted) == len(MODALITIES):
                total_converted += 1
                print(f"  ✓ {patient_id}: {', '.join(converted)}")
            else:
                failed_patients.append(patient_id)
                print(f"  ⚠ {patient_id}: solo {converted}")

            total_patients += 1

    # Resumen final
    print()
    print("=" * 60)
    print(f"RESUMEN")
    print(f"  Pacientes procesados: {total_patients}")
    print(f"  Conversiones completas (T1w + T2w): {total_converted}")
    print(f"  Pacientes con problemas: {len(failed_patients)}")
    if failed_patients:
        print(f"  IDs con problemas: {failed_patients[:20]}...")
    print("=" * 60)


if __name__ == "__main__":
    main()