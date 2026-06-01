"""
audit_lodo.py
-------------
Test de estrés cross-dataset (leave-one-dataset-out por clase) usando el
tiny baseline: un modelo lineal/arbol sobre estadisticos de intensidad.

Pregunta: si entreno a separar (tumorA vs sanoA) y evaluo en (tumorB vs sanoB),
¿transfiere?
  - AUC test alto  -> la "senal" trivial de intensidad cruza datasets (confound
                      de familia-de-dominio); NO prueba deteccion de tumor.
  - AUC test bajo  -> el modelo dependia de firmas especificas de cada dataset
                      que no transfieren (fragilidad por confound).

Tambien incluye el split aleatorio mezclado como referencia (deberia dar ~1.0).

NO toca el pipeline de entrenamiento/evaluacion. Solo lee data/processed.

Uso:
    python -m src.audit.audit_lodo --per-dataset 150
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"
OUT_DIR = REPO_ROOT / "docs" / "audit"

FEATURE_NAMES = [
    "nz_frac_t1", "mean_t1", "std_t1", "p01_t1", "p25_t1", "p50_t1", "p75_t1", "p99_t1",
    "nz_frac_t2", "mean_t2", "std_t2", "p01_t2", "p25_t2", "p50_t2", "p75_t2", "p99_t2",
]


def _scalar(v):
    if isinstance(v, np.ndarray):
        v = v.item()
    if isinstance(v, bytes):
        return v.decode("utf-8")
    return v


def vol_feats(arr: np.ndarray) -> list[float]:
    nz = arr[arr != 0]
    frac = float(nz.size) / float(arr.size)
    if nz.size == 0:
        return [frac, 0, 0, 0, 0, 0, 0, 0]
    p01, p25, p50, p75, p99 = np.percentile(nz, [1, 25, 50, 75, 99])
    return [frac, float(nz.mean()), float(nz.std()),
            float(p01), float(p25), float(p50), float(p75), float(p99)]


def load_index() -> dict[str, list[Path]]:
    """Agrupa los .npz por dataset."""
    by_ds: dict[str, list[Path]] = defaultdict(list)
    for sub in ("positives", "negatives"):
        for p in sorted((PROCESSED / sub).glob("*.npz")):
            with np.load(p) as s:
                ds = str(_scalar(s["dataset"])) if "dataset" in s.files else "unknown"
            by_ds[ds].append(p)
    return by_ds


def extract(files: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for p in files:
        with np.load(p) as s:
            t1 = s["t1"].astype(np.float32, copy=False)
            t2 = s["t2"].astype(np.float32, copy=False)
            lbl = int(s["label"]) if "label" in s.files else (1 if p.parent.name == "positives" else 0)
        X.append(vol_feats(t1) + vol_feats(t2))
        y.append(lbl)
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64)


def evaluate(Xtr, ytr, Xte, yte) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    out = {}
    if len(set(ytr)) < 2 or len(set(yte)) < 2:
        return {"logreg_test_auc": float("nan"), "rf_test_auc": float("nan"),
                "note": "una sola clase en train o test"}
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr)
    out["logreg_test_auc"] = float(roc_auc_score(yte, lr.predict_proba(sc.transform(Xte))[:, 1]))
    rf = RandomForestClassifier(n_estimators=300, random_state=0).fit(Xtr, ytr)
    out["rf_test_auc"] = float(roc_auc_score(yte, rf.predict_proba(Xte)[:, 1]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-dataset", type=int, default=150, help="muestras/dataset (0 = todo)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    by_ds = load_index()
    print("Disponibles por dataset:", {k: len(v) for k, v in by_ds.items()})

    # Submuestreo por dataset
    sampled: dict[str, list[Path]] = {}
    for ds, files in by_ds.items():
        files = sorted(files)
        rng.shuffle(files)
        sampled[ds] = files[: args.per_dataset] if args.per_dataset > 0 else files

    # Cache de features por dataset
    print("Extrayendo features por dataset...")
    feats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ds, files in sampled.items():
        feats[ds] = extract(files)
        print(f"  {ds}: {len(files)} muestras")

    def stack(dsets):
        Xs = [feats[d][0] for d in dsets if d in feats]
        ys = [feats[d][1] for d in dsets if d in feats]
        return np.vstack(Xs), np.concatenate(ys)

    # Detectar nombres reales (pueden variar: nki vs nki_rockland, etc.)
    names = set(feats.keys())
    def pick(*cands):
        for c in cands:
            if c in names:
                return c
        return None
    brats = pick("brats", "BraTS", "brats2021")
    upenn = pick("upenn", "UPENN-GBM", "upenn_gbm")
    ixi = pick("ixi", "IXI")
    nki = pick("nki_rockland", "nki", "NKI")
    print(f"\nMapeo: tumor=({brats},{upenn})  sano=({ixi},{nki})")

    # Configuraciones LODO: train (un tumor + un sano) -> test (los otros)
    configs = [
        ("train BraTS+IXI -> test UPENN+NKI", [brats, ixi], [upenn, nki]),
        ("train UPENN+NKI -> test BraTS+IXI", [upenn, nki], [brats, ixi]),
        ("train BraTS+NKI -> test UPENN+IXI", [brats, nki], [upenn, ixi]),
        ("train UPENN+IXI -> test BraTS+NKI", [upenn, ixi], [brats, nki]),
    ]

    results = {}
    print("\n================ LODO (tiny baseline) ================")
    for name, tr, te in configs:
        if any(x is None for x in tr + te):
            print(f"  [skip] {name}: falta algun dataset")
            continue
        Xtr, ytr = stack(tr)
        Xte, yte = stack(te)
        r = evaluate(Xtr, ytr, Xte, yte)
        results[name] = r
        print(f"  {name}")
        print(f"      LogReg test AUC = {r.get('logreg_test_auc', float('nan')):.4f} | "
              f"RandomForest test AUC = {r.get('rf_test_auc', float('nan')):.4f}")

    # Referencia: split aleatorio mezclado (deberia salir ~1.0)
    allX = np.vstack([feats[d][0] for d in feats])
    allY = np.concatenate([feats[d][1] for d in feats])
    idx = rng.permutation(len(allY))
    cut = int(len(idx) * 0.7)
    ref = evaluate(allX[idx[:cut]], allY[idx[:cut]], allX[idx[cut:]], allY[idx[cut:]])
    results["REFERENCIA random-mix 70/30"] = ref
    print("\n  REFERENCIA (split aleatorio mezclado, deberia ~1.0):")
    print(f"      LogReg test AUC = {ref.get('logreg_test_auc', float('nan')):.4f} | "
          f"RandomForest test AUC = {ref.get('rf_test_auc', float('nan')):.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "audit_lodo.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nJSON -> {out}")


if __name__ == "__main__":
    main()
