"""
skull_strip_ixi.py
------------------
Aplica HD-BET (brain extraction con deep learning, Isensee et al. 2019) a todos
los volúmenes de IXI (T1 y T2) para homogeneizar el skull-stripping con BraTS,
que ya viene sin cráneo de fábrica.

Este paso es crítico para eliminar el sesgo de dominio: sin skull-stripping,
el modelo aprende trivialmente a distinguir IXI (con cráneo) de BraTS
(sin cráneo) en lugar de detectar señales tumorales.

Referencia:
  Isensee F, et al. Automated brain extraction of multi-sequence MRI using
  artificial neural networks. Hum Brain Mapp. 2019;40:4952–4964.

Uso:
    python src/preprocessing/skull_strip_ixi.py

Requiere: HD-BET (pip install HD-BET) y GPU CUDA disponible.
El script es reanudable: si se interrumpe, al relanzarlo salta los ya procesados.
"""

# Silenciar warnings de nnU-Net sobre variables de entorno no definidas
# (son innocuas para inferencia, pero llenan la salida)
import os
os.environ.setdefault("nnUNet_raw",          "tmp_unused")
os.environ.setdefault("nnUNet_preprocessed", "tmp_unused")
os.environ.setdefault("nnUNet_results",      "tmp_unused")

import sys
import time
import torch
from pathlib import Path
from tqdm import tqdm
from HD_BET.checkpoint_download import maybe_download_parameters
from HD_BET.hd_bet_prediction import get_hdbet_predictor, hdbet_predict


# ============================================================
# CONFIGURACIÓN — Ajusta estas rutas a tu máquina
# ============================================================

IXI_T1_DIR  = Path(r"C:\Trabajo Fin Grado\IXI-T1")
IXI_T2_DIR  = Path(r"C:\Trabajo Fin Grado\IXI-T2")
OUTPUT_ROOT = Path(r"C:\Trabajo Fin Grado\brain-mri-triage\data\raw\ixi_stripped")

# ============================================================


def strip_folder(input_dir, output_dir, predictor, nombre):
    """Aplica HD-BET a todos los .nii.gz de un directorio, saltando los ya procesados."""
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob("*.nii.gz"))
    print(f"\n{nombre}: {len(files)} volúmenes encontrados")

    already_done = sum(1 for f in files if (output_dir / f.name).exists())
    if already_done:
        print(f"  {already_done} ya procesados — se saltarán")

    skipped = []
    for f in tqdm(files, desc=nombre, unit="vol"):
        out = output_dir / f.name
        if out.exists():
            continue
        try:
            hdbet_predict(
                str(f), str(out), predictor,
                keep_brain_mask=False,
                compute_brain_extracted_image=True,
            )
        except Exception as e:
            print(f"\n  Error procesando {f.name}: {e}")
            skipped.append(f.name)

    processed = len(files) - len(skipped)
    print(f"  Procesados: {processed}/{len(files)}")
    if skipped:
        print(f"  Saltados: {len(skipped)} -> {skipped[:5]}...")


def main():
    print("=" * 60)
    print("SKULL-STRIPPING DE IXI CON HD-BET")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: No hay GPU CUDA. Este script requiere GPU para ir a una velocidad razonable.")
        sys.exit(1)

    for path, name in [(IXI_T1_DIR, "IXI T1"), (IXI_T2_DIR, "IXI T2")]:
        if not path.exists():
            print(f"ERROR: No se encuentra {name}: {path}")
            sys.exit(1)

    print("Descargando modelo HD-BET si es necesario...")
    maybe_download_parameters()

    print("Creando predictor (use_tta=False para priorizar velocidad)...")
    predictor = get_hdbet_predictor(
        use_tta=False,
        device=torch.device("cuda"),
        verbose=False,
    )

    t0 = time.time()
    strip_folder(IXI_T1_DIR, OUTPUT_ROOT / "t1", predictor, "IXI T1")
    strip_folder(IXI_T2_DIR, OUTPUT_ROOT / "t2", predictor, "IXI T2")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Tiempo total: {elapsed / 60:.1f} min")
    print(f"Volúmenes stripped guardados en: {OUTPUT_ROOT}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()