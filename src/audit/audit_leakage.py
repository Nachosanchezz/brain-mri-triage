"""
audit_leakage.py
----------------
Pruebas negativas para diagnosticar resultados sospechosamente altos en el
pipeline de triaje de RM cerebral.

NO toca el pipeline de entrenamiento/evaluacion. Solo lee data/splits.json y
los .npz de data/processed, calcula features baratas (no clinicas) por volumen
y comprueba si la etiqueta y/o el dataset de origen son triviales de predecir
a partir de esas features. Tambien detecta duplicados exactos entre splits.

Pruebas implementadas:
  B) Tiny baseline       : LogReg/RandomForest sobre features de intensidad
                           -> AUC alto = la etiqueta se decodifica sin senal clinica.
  C) Dataset-origin clf  : predecir el dataset de origen (4 clases) desde features
                           -> trivial = fuerte sesgo de dominio.
  Duplicados             : sha1 de (t1,t2) por fichero, solapamiento entre splits.

Salida: docs/audit/  (CSV de features+hashes y JSON con metricas).

Uso:
    python -m src.audit.audit_leakage                # muestreo por defecto
    python -m src.audit.audit_leakage --per-dataset-train 0   # usar todo train
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SPLITS_FILE = REPO_ROOT / "data" / "splits.json"
OUT_DIR = REPO_ROOT / "docs" / "audit"

FEATURE_NAMES = [
    "nz_frac_t1", "mean_t1", "std_t1", "p01_t1", "p25_t1", "p50_t1", "p75_t1", "p99_t1",
    "nz_frac_t2", "mean_t2", "std_t2", "p01_t2", "p25_t2", "p50_t2", "p75_t2", "p99_t2",
]


def _scalar(value):
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def volume_features(arr: np.ndarray) -> list[float]:
    """Features baratas y NO clinicas: fraccion de voxels no-cero y estadisticos
    de intensidad sobre el tejido. Si estas separan la clase, la senal es de
    dominio/protocolo, no de tumor."""
    nz = arr[arr != 0]
    frac = float(nz.size) / float(arr.size)
    if nz.size == 0:
        return [frac, 0, 0, 0, 0, 0, 0, 0]
    p01, p25, p50, p75, p99 = np.percentile(nz, [1, 25, 50, 75, 99])
    return [frac, float(nz.mean()), float(nz.std()),
            float(p01), float(p25), float(p50), float(p75), float(p99)]


def extract(path: str) -> dict:
    p = Path(path)
    with np.load(p) as s:
        t1 = s["t1"].astype(np.float32, copy=False)
        t2 = s["t2"].astype(np.float32, copy=False)
        label = int(s["label"]) if "label" in s.files else (1 if p.parent.name == "positives" else 0)
        ds = str(_scalar(s["dataset"])) if "dataset" in s.files else "unknown"
        sid = str(_scalar(s["subject_id"])) if "subject_id" in s.files else p.stem
        h = hashlib.sha1()
        h.update(np.ascontiguousarray(t1).tobytes())
        h.update(np.ascontiguousarray(t2).tobytes())
    feats = volume_features(t1) + volume_features(t2)
    return {"file": str(p), "dataset": ds, "subject_id": sid, "label": label,
            "sha1": h.hexdigest(), "features": feats}


def subsample(files: list[str], rows_meta: dict, per_dataset: int, seed: int) -> list[str]:
    if per_dataset <= 0:
        return files
    rng = np.random.default_rng(seed)
    by_ds: dict[str, list[str]] = defaultdict(list)
    for f in files:
        by_ds[rows_meta[f]].append(f)
    out: list[str] = []
    for ds, fs in by_ds.items():
        fs = sorted(fs)
        rng.shuffle(fs)
        out.extend(fs[:per_dataset])
    return out


def fit_eval_label(Xtr, ytr, Xte, yte) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    out = {}
    lr = LogisticRegression(max_iter=2000).fit(Xtr_s, ytr)
    out["logreg_test_auc"] = float(roc_auc_score(yte, lr.predict_proba(Xte_s)[:, 1]))
    out["logreg_test_acc"] = float(lr.score(Xte_s, yte))

    rf = RandomForestClassifier(n_estimators=300, random_state=0).fit(Xtr, ytr)
    out["rf_test_auc"] = float(roc_auc_score(yte, rf.predict_proba(Xte)[:, 1]))
    out["rf_test_acc"] = float(rf.score(Xte, yte))
    out["rf_feature_importance"] = {
        name: float(imp) for name, imp in sorted(
            zip(FEATURE_NAMES, rf.feature_importances_), key=lambda kv: -kv[1])
    }
    return out


def fit_eval_dataset(Xtr, dtr, Xte, dte) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=300, random_state=0).fit(Xtr, dtr)
    return {"rf_dataset_test_acc": float(rf.score(Xte, dte)),
            "n_classes": len(set(dtr))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-dataset-train", type=int, default=150,
                    help="muestras/dataset en train (0 = todo)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    splits = json.loads(SPLITS_FILE.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Metadatos ligeros para poder submuestrear train por dataset sin abrir todo.
    print("Indexando metadatos por fichero...")
    file_ds: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for f in splits[split]:
            with np.load(f) as s:
                file_ds[f] = str(_scalar(s["dataset"])) if "dataset" in s.files else "unknown"

    train_files = subsample(splits["train"], file_ds, args.per_dataset_train, args.seed)
    test_files = splits["test"]  # test completo, evaluacion honesta
    print(f"train usados: {len(train_files)} | test usados: {len(test_files)}")

    print("Extrayendo features y hashes...")
    rows = {}
    for split, files in (("train", train_files), ("test", test_files), ("val", splits["val"])):
        for i, f in enumerate(files):
            if f in rows:
                rows[f]["splits"].add(split)
                continue
            r = extract(f)
            r["splits"] = {split}
            rows[f] = r
        print(f"  {split}: hecho")

    # --- Duplicados exactos por sha1, cruzando splits ---
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for r in rows.values():
        by_hash[r["sha1"]].append(r)
    dup_groups = []
    for h, group in by_hash.items():
        if len(group) > 1:
            dup_groups.append({
                "sha1": h,
                "files": [Path(g["file"]).name for g in group],
                "splits": sorted(set().union(*[g["splits"] for g in group])),
                "labels": sorted(set(g["label"] for g in group)),
            })

    # --- Matrices para los clasificadores ---
    def mat(files):
        X = np.array([rows[f]["features"] for f in files], dtype=np.float64)
        y = np.array([rows[f]["label"] for f in files], dtype=np.int64)
        d = np.array([rows[f]["dataset"] for f in files])
        return X, y, d

    Xtr, ytr, dtr = mat(train_files)
    Xte, yte, dte = mat(test_files)

    print("Prueba B (tiny baseline: features de intensidad -> etiqueta)...")
    label_res = fit_eval_label(Xtr, ytr, Xte, yte)
    print("Prueba C (clasificador de dataset de origen)...")
    ds_res = fit_eval_dataset(Xtr, dtr, Xte, dte)

    report = {
        "n_train": len(train_files),
        "n_test": len(test_files),
        "train_label_balance": dict(Counter(int(v) for v in ytr)),
        "test_label_balance": dict(Counter(int(v) for v in yte)),
        "tiny_baseline_label": label_res,
        "dataset_origin_classifier": ds_res,
        "exact_duplicates_cross_split": {
            "n_duplicate_groups": len(dup_groups),
            "n_cross_split_groups": sum(1 for g in dup_groups if len(g["splits"]) > 1),
            "groups": dup_groups[:50],
        },
        "label_equals_dataset": {
            ds: sorted(set(int(rows[f]["label"]) for f in rows if rows[f]["dataset"] == ds))
            for ds in sorted(set(file_ds.values()))
        },
    }

    out_json = OUT_DIR / "audit_leakage.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # CSV de features
    csv_path = OUT_DIR / "audit_features.csv"
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("file,dataset,subject_id,label,sha1," + ",".join(FEATURE_NAMES) + "\n")
        for r in rows.values():
            fh.write(f"{Path(r['file']).name},{r['dataset']},{r['subject_id']},{r['label']},{r['sha1']},"
                     + ",".join(f"{x:.6g}" for x in r["features"]) + "\n")

    print("\n================ RESULTADOS ================")
    print(f"Tiny baseline (LogReg)  test AUC = {label_res['logreg_test_auc']:.4f}  acc = {label_res['logreg_test_acc']:.4f}")
    print(f"Tiny baseline (RandomF) test AUC = {label_res['rf_test_auc']:.4f}  acc = {label_res['rf_test_acc']:.4f}")
    print(f"Dataset-origin classifier test acc = {ds_res['rf_dataset_test_acc']:.4f} ({ds_res['n_classes']} clases)")
    print(f"Grupos duplicados exactos: {report['exact_duplicates_cross_split']['n_duplicate_groups']} "
          f"(cruzando splits: {report['exact_duplicates_cross_split']['n_cross_split_groups']})")
    print("label_equals_dataset:", report["label_equals_dataset"])
    print(f"\nJSON -> {out_json}\nCSV  -> {csv_path}")


if __name__ == "__main__":
    main()
