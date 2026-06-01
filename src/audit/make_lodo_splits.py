"""
make_lodo_splits.py
-------------------
Genera un splits.json para el experimento LODO (leave-one-dataset-out
por clase) sin tocar la logica del pipeline.

Reutiliza el formato de data/splits.json que ya espera src/data/dataset_3d.py,
asi que train_3d.py y evaluate_3d.py funcionan sin cambios siempre que
configs/train_lodo.yaml tenga `recreate_splits: false`.

Configuraciones:
  A: train+val={brats,ixi}        test={upenn,nki_rockland}
  B: train+val={upenn,nki_rockland} test={brats,ixi}
  C: train+val={brats,nki_rockland} test={upenn,ixi}
  D: train+val={upenn,ixi}        test={brats,nki_rockland}

Escribe DOS ficheros:
  data/splits.json                          (el que lee el pipeline)
  data/splits_lodo_<CONFIG>.json            (registro permanente de cada config)

Uso:
    python -m src.audit.make_lodo_splits --config A
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"
SPLITS_FILE = REPO_ROOT / "data" / "splits.json"

CONFIGS = {
    "A": (("brats", "ixi"),           ("upenn", "nki_rockland")),
    "B": (("upenn", "nki_rockland"),  ("brats", "ixi")),
    "C": (("brats", "nki_rockland"),  ("upenn", "ixi")),
    "D": (("upenn", "ixi"),           ("brats", "nki_rockland")),
}


def _scalar(v):
    if isinstance(v, np.ndarray):
        v = v.item()
    if isinstance(v, bytes):
        return v.decode("utf-8")
    return v


def index_by_dataset_and_subject() -> dict[str, dict[str, list[Path]]]:
    """Devuelve {dataset: {subject_id: [path, ...]}}."""
    out: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for sub in ("positives", "negatives"):
        for p in sorted((PROCESSED / sub).glob("*.npz")):
            with np.load(p) as s:
                ds = str(_scalar(s["dataset"])) if "dataset" in s.files else "unknown"
                sid = str(_scalar(s["subject_id"])) if "subject_id" in s.files else p.stem
            out[ds][sid].append(p)
    return out


def split_train_val_by_subject(
    subjects: list[str], val_ratio: float, rng: np.random.Generator
) -> tuple[list[str], list[str]]:
    subjects = sorted(subjects)
    rng.shuffle(subjects)
    n_val = int(round(len(subjects) * val_ratio))
    return subjects[n_val:], subjects[:n_val]  # train, val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=sorted(CONFIGS), required=True,
                    help="Configuracion LODO (A/B/C/D)")
    ap.add_argument("--val-ratio", type=float, default=0.15,
                    help="Fraccion de sujetos del pool train+val que van a val")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_ds_names, test_ds_names = CONFIGS[args.config]
    print(f"Config {args.config}: train+val={train_ds_names}  test={test_ds_names}")

    index = index_by_dataset_and_subject()
    missing = [d for d in train_ds_names + test_ds_names if d not in index]
    if missing:
        raise SystemExit(f"ERROR: no se encontraron datasets en data/processed: {missing}\n"
                         f"Disponibles: {sorted(index.keys())}")

    rng = np.random.default_rng(args.seed)

    # TRAIN/VAL: split por sujeto dentro de cada dataset del pool de entrenamiento
    train_files: list[str] = []
    val_files: list[str] = []
    for ds in train_ds_names:
        subjects = list(index[ds].keys())
        tr_subs, vl_subs = split_train_val_by_subject(subjects, args.val_ratio, rng)
        for sid in tr_subs:
            train_files.extend(str(p) for p in index[ds][sid])
        for sid in vl_subs:
            val_files.extend(str(p) for p in index[ds][sid])

    # TEST: TODOS los ficheros de los datasets de test (sujetos completos)
    test_files: list[str] = []
    for ds in test_ds_names:
        for sid in index[ds]:
            test_files.extend(str(p) for p in index[ds][sid])

    rng.shuffle(train_files)
    rng.shuffle(val_files)
    rng.shuffle(test_files)

    splits = {
        "seed": args.seed,
        "config": args.config,
        "train_datasets": list(train_ds_names),
        "test_datasets": list(test_ds_names),
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }

    # Registro permanente por config
    record_path = REPO_ROOT / "data" / f"splits_lodo_{args.config}.json"
    record_path.write_text(json.dumps(splits, indent=2), encoding="utf-8")

    # El fichero que lee el pipeline
    SPLITS_FILE.write_text(json.dumps(splits, indent=2), encoding="utf-8")

    # Resumen
    def summarize(name: str, files: list[str]) -> None:
        by = defaultdict(lambda: [0, 0])  # dataset -> [n_pos, n_neg]
        for f in files:
            p = Path(f)
            lbl = 1 if p.parent.name == "positives" else 0
            with np.load(p) as s:
                ds = str(_scalar(s["dataset"])) if "dataset" in s.files else "?"
            by[ds][0 if lbl == 1 else 1] += 1  # idx0 -> pos, idx1 -> neg
        total = len(files)
        print(f"  {name:5s} (n={total}): " + ", ".join(
            f"{ds}: {p}+/{n}-" for ds, (p, n) in sorted(by.items())))

    print(f"Escrito: {record_path}")
    print(f"Escrito: {SPLITS_FILE}  (lo leera train_3d.py / evaluate_3d.py)")
    print("\nComposicion:")
    summarize("train", train_files)
    summarize("val",   val_files)
    summarize("test",  test_files)


if __name__ == "__main__":
    main()
