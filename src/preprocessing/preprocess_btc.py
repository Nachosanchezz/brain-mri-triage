"""
preprocess_btc.py
-----------------
Preprocesado de BTC_preop (OpenNeuro ds001226) a .npz **T1-only**.

Pipeline (identico al resto de datasets, sin T2):
  HD-BET skull-strip (opcional, --skull-strip) ->
  reorient RAS -> resample 1mm iso -> crop/pad 192x224x192 ->
  normalize_intensity (Otsu sobre voxels positivos).

Etiquetado:
  sub-CON*  -> label 0 (sano)   -> data/processed_btc/negatives/
  sub-PAT*  -> label 1 (tumor)  -> data/processed_btc/positives/

Salida: data/processed_btc/{positives,negatives}/sub-<ID>.npz con:
  t1, label, dataset='btc', subject_id, source_t1.

(No se guarda t2: ds001226 no lo trae. El experimento intra-dominio sera
 1-canal y se hace en src/audit/btc_*.py, sin tocar el pipeline 2-canales.)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Silencia warnings de nnU-Net cuando se usa HD-BET en inferencia (igual que IXI).
os.environ.setdefault("nnUNet_raw", "tmp_unused")
os.environ.setdefault("nnUNet_preprocessed", "tmp_unused")
os.environ.setdefault("nnUNet_results", "tmp_unused")

try:
    from .base_preprocessing import (
        REPO_ROOT,
        PreprocessingConfig,
        preprocess_single_volume,
    )
except ImportError:
    from base_preprocessing import (
        REPO_ROOT,
        PreprocessingConfig,
        preprocess_single_volume,
    )


DEFAULT_BTC_RAW_DIR = REPO_ROOT / "data" / "raw" / "btc_preop" / "raw"
DEFAULT_BTC_STRIPPED_DIR = REPO_ROOT / "data" / "raw" / "btc_preop" / "stripped"
DEFAULT_BTC_OUTPUT_DIR = REPO_ROOT / "data" / "processed_btc"


def label_for_subject(subject_id: str) -> int | None:
    """sub-CON* -> 0 (sano), sub-PAT* -> 1 (tumor), otros -> None."""
    sid = subject_id.upper()
    if "CON" in sid:
        return 0
    if "PAT" in sid:
        return 1
    return None


def subject_id_from_filename(path: Path) -> str:
    """sub-CON01_T1w.nii.gz -> sub-CON01"""
    name = path.name
    for sep in ("_T1w", "_ses-", "."):
        if sep in name:
            name = name.split(sep)[0]
            break
    return name


def skull_strip_folder(input_dir: Path, output_dir: Path) -> tuple[int, int]:
    """Aplica HD-BET sobre input_dir y escribe a output_dir (mismo nombre)."""
    import torch
    from HD_BET.checkpoint_download import maybe_download_parameters
    from HD_BET.hd_bet_prediction import get_hdbet_predictor, hdbet_predict
    from tqdm import tqdm

    if not torch.cuda.is_available():
        print("ERROR: no hay GPU CUDA/ROCm. HD-BET sin GPU es demasiado lento.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    maybe_download_parameters()
    predictor = get_hdbet_predictor(use_tta=False, device=torch.device("cuda"), verbose=False)

    files = sorted(Path(input_dir).glob("*.nii.gz"))
    print(f"HD-BET sobre {len(files)} volumenes en {input_dir}")
    already = sum(1 for p in files if (output_dir / p.name).exists())
    if already:
        print(f"  {already} ya stripped, se saltan")

    errors: list[str] = []
    for path in tqdm(files, desc="HD-BET BTC", unit="vol"):
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
            print(f"\n  Error en {path.name}: {exc}")
            errors.append(path.name)

    ok = len(files) - len(errors)
    print(f"  Procesados: {ok}/{len(files)}")
    if errors:
        print(f"  Errores: {errors}")
    return ok, len(errors)


def preprocess_btc(
    stripped_dir: Path,
    output_dir: Path,
    config: PreprocessingConfig,
    overwrite: bool = False,
) -> dict:
    """Para cada T1 stripped: pipeline -> .npz."""
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        def tqdm(it, **kw):
            return it

    files = sorted(Path(stripped_dir).glob("*.nii.gz"))
    if not files:
        raise SystemExit(f"No hay .nii.gz en {stripped_dir}. ¿Hiciste --skull-strip?")

    written = 0
    existing = 0
    skipped: list[dict] = []

    for path in tqdm(files, desc="BTC preprocess", unit="vol"):
        sid = subject_id_from_filename(path)
        label = label_for_subject(sid)
        if label is None:
            skipped.append({"subject_id": sid, "error": "sin etiqueta CON/PAT"})
            continue
        subdir = "positives" if label == 1 else "negatives"
        out_path = output_dir / subdir / f"{sid}.npz"

        if out_path.exists() and not overwrite:
            existing += 1
            continue

        try:
            t1 = preprocess_single_volume(path, config)
        except Exception as exc:
            print(f"\n  Error procesando {sid}: {exc}")
            skipped.append({"subject_id": sid, "error": str(exc)})
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(out_path),
            t1=t1.astype(np.float32),
            label=np.int64(label),
            dataset=np.array("btc"),
            subject_id=np.array(sid),
            source_t1=np.array(str(path)),
        )
        written += 1

    print()
    print(f"Procesados nuevos : {written}")
    print(f"Ya existentes     : {existing}")
    print(f"Errores           : {len(skipped)}")
    return {"written": written, "existing": existing, "skipped": skipped}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocesa BTC_preop (ds001226) a .npz T1-only.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_BTC_RAW_DIR,
                        help="Directorio con los T1w originales (no stripped).")
    parser.add_argument("--stripped-dir", type=Path, default=DEFAULT_BTC_STRIPPED_DIR,
                        help="Directorio con los T1w skull-stripped (input del preprocesado).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BTC_OUTPUT_DIR,
                        help="Salida .npz: <output>/{positives,negatives}/")
    parser.add_argument("--skull-strip", action="store_true",
                        help="Ejecuta HD-BET sobre raw-dir antes del preprocesado.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    if args.skull_strip:
        if not args.raw_dir.exists():
            print(f"ERROR: no existe {args.raw_dir}")
            sys.exit(1)
        skull_strip_folder(args.raw_dir, args.stripped_dir)

    if not args.stripped_dir.exists():
        print(f"ERROR: no existe {args.stripped_dir}. Lanza con --skull-strip primero.")
        sys.exit(1)

    config = PreprocessingConfig()
    result = preprocess_btc(args.stripped_dir, args.output_dir, config, overwrite=args.overwrite)

    elapsed = time.time() - t0
    print(f"\nTiempo total: {elapsed / 60:.1f} min")
    print(f"Salidas: {args.output_dir}/{{positives,negatives}}")


if __name__ == "__main__":
    main()
