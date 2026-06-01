"""
btc_tiny_baseline.py
--------------------
Tiny baseline (modelo lineal/arbol sobre estadisticos de intensidad de T1)
con k-fold por sujeto SOBRE BTC, dataset intra-dominio (mismo escaner).

Pregunta: el atajo trivial de intensidad que daba AUC 1.0 en los datos
confundidos y AUC caotico en LODO, ¿separa tumor vs sano DENTRO del mismo
dominio?

Si AUC ~ azar (0.5) -> el atajo era firma de dominio, no tumor.
Si AUC alto         -> hay intensidad-only que correlaciona con tumor incluso
                       intra-dominio (a discutir si es biologia real o un
                       efecto colateral con n=36).

Salida: docs/audit/btc_intradomain_tinybaseline.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed_btc"
OUT_DIR = REPO_ROOT / "docs" / "audit"

FEATURE_NAMES = [
    "nz_frac_t1", "mean_t1", "std_t1",
    "p01_t1", "p25_t1", "p50_t1", "p75_t1", "p99_t1",
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
        return [frac, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    p01, p25, p50, p75, p99 = np.percentile(nz, [1, 25, 50, 75, 99])
    return [frac, float(nz.mean()), float(nz.std()),
            float(p01), float(p25), float(p50), float(p75), float(p99)]


def load_samples() -> tuple[np.ndarray, np.ndarray, list[str]]:
    files = sorted(list((PROCESSED / "positives").glob("*.npz"))
                   + list((PROCESSED / "negatives").glob("*.npz")))
    X, y, sids = [], [], []
    for p in files:
        with np.load(p) as s:
            t1 = s["t1"].astype(np.float32, copy=False)
            lbl = int(s["label"]) if "label" in s.files else (1 if p.parent.name == "positives" else 0)
            sid = str(_scalar(s["subject_id"])) if "subject_id" in s.files else p.stem
        X.append(vol_feats(t1))
        y.append(lbl)
        sids.append(sid)
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64), sids


def stratified_kfold(y: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for i, ix in enumerate(idx):
            folds[i % k].append(int(ix))
    return [np.array(sorted(f)) for f in folds]


def kfold_eval(X: np.ndarray, y: np.ndarray, k: int, seed: int) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    folds = stratified_kfold(y, k, seed)
    out_scores_lr = np.full(len(y), np.nan)
    out_scores_rf = np.full(len(y), np.nan)
    per_fold = []

    for fi, test_idx in enumerate(folds):
        train_idx = np.array(sorted(set(range(len(y))) - set(test_idx.tolist())))
        Xtr, ytr = X[train_idx], y[train_idx]
        Xte, yte = X[test_idx], y[test_idx]

        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr)
        sc_lr = lr.predict_proba(sc.transform(Xte))[:, 1]
        out_scores_lr[test_idx] = sc_lr

        rf = RandomForestClassifier(n_estimators=300, random_state=0).fit(Xtr, ytr)
        sc_rf = rf.predict_proba(Xte)[:, 1]
        out_scores_rf[test_idx] = sc_rf

        fold_auc_lr = float(roc_auc_score(yte, sc_lr)) if len(set(yte)) > 1 else float("nan")
        fold_auc_rf = float(roc_auc_score(yte, sc_rf)) if len(set(yte)) > 1 else float("nan")
        per_fold.append({"fold": fi, "n_test": int(len(test_idx)),
                         "logreg_auc": fold_auc_lr, "rf_auc": fold_auc_rf})

    auc_lr = float(roc_auc_score(y, out_scores_lr))
    auc_rf = float(roc_auc_score(y, out_scores_rf))
    return {"logreg_auc_overall": auc_lr, "rf_auc_overall": auc_rf,
            "per_fold": per_fold,
            "scores_lr": out_scores_lr.tolist(), "scores_rf": out_scores_rf.tolist(),
            "y_true": y.tolist()}


def bootstrap_auc_ci(y_true: list[int], y_score: list[float], n_boot: int = 2000,
                     seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)
    n = len(y_true_arr)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(set(y_true_arr[idx].tolist())) < 2:
            continue
        aucs.append(roc_auc_score(y_true_arr[idx], y_score_arr[idx]))
    aucs = np.array(aucs)
    return float(np.mean(aucs)), float(np.quantile(aucs, alpha / 2)), float(np.quantile(aucs, 1 - alpha / 2))


def main() -> None:
    X, y, sids = load_samples()
    print(f"Cargados: n={len(y)}  positivos={int(y.sum())}  negativos={int((y == 0).sum())}")
    print(f"  controles: {[s for s, lbl in zip(sids, y) if lbl == 0]}")
    print(f"  pacientes (primeros 5): {[s for s, lbl in zip(sids, y) if lbl == 1][:5]}...")

    if int(y.sum()) < 2 or int((y == 0).sum()) < 2:
        raise SystemExit("ERROR: muy pocas muestras de alguna clase para k-fold.")

    k = min(5, int(min((y == 1).sum(), (y == 0).sum())))
    print(f"k-fold = {k}-fold estratificado por sujeto")

    res = kfold_eval(X, y, k=k, seed=42)

    print(f"\nAUC global (LogReg) = {res['logreg_auc_overall']:.4f}")
    print(f"AUC global (RF)     = {res['rf_auc_overall']:.4f}")
    print("Per-fold:")
    for f in res["per_fold"]:
        print(f"  fold {f['fold']} (n_test={f['n_test']}): LogReg={f['logreg_auc']:.4f}  RF={f['rf_auc']:.4f}")

    print("\nBootstrap IC95% (2000 resamples):")
    mu_lr, lo_lr, hi_lr = bootstrap_auc_ci(res["y_true"], res["scores_lr"])
    mu_rf, lo_rf, hi_rf = bootstrap_auc_ci(res["y_true"], res["scores_rf"])
    print(f"  LogReg AUC = {res['logreg_auc_overall']:.4f}  IC95 [{lo_lr:.4f}, {hi_lr:.4f}]  (boot mean {mu_lr:.4f})")
    print(f"  RF     AUC = {res['rf_auc_overall']:.4f}  IC95 [{lo_rf:.4f}, {hi_rf:.4f}]  (boot mean {mu_rf:.4f})")

    out = {
        "n_samples": int(len(y)),
        "n_positives": int(y.sum()),
        "n_negatives": int((y == 0).sum()),
        "k_fold": k,
        "logreg_auc_overall": res["logreg_auc_overall"],
        "logreg_auc_ci95": [lo_lr, hi_lr],
        "rf_auc_overall": res["rf_auc_overall"],
        "rf_auc_ci95": [lo_rf, hi_rf],
        "per_fold": res["per_fold"],
        "feature_names": FEATURE_NAMES,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "btc_intradomain_tinybaseline.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nJSON -> {out_path}")


if __name__ == "__main__":
    main()
