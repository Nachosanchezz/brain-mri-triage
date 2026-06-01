"""
embeddings_intraclass.py
------------------------
Analisis refinado sobre los embeddings ya extraidos (docs/audit/embeddings.npz):

1. Silhouette INTRA-CLASE: dentro de los sanos, ¿se separan IXI y NKI?
   dentro de los tumores, ¿se separan BraTS y UPENN? Aisla la huella de
   PROCEDENCIA de la señal de clase (que en el silhouette global la tapaba).

2. Clasificador de DATASET sobre los embeddings de la CNN (no sobre pixeles
   crudos): si un modelo simple predice el dataset desde el vector latente,
   la red codifica origen.

NO sustituye al silhouette global (0.37/0.75): lo complementa. Ambos se
reportan juntos por honestidad.

Salida: docs/audit/embeddings_intraclass.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EMB = REPO_ROOT / "docs" / "audit" / "embeddings.npz"
OUT = REPO_ROOT / "docs" / "audit" / "embeddings_intraclass.json"


def main():
    if not EMB.exists():
        raise SystemExit(f"No existe {EMB}. Ejecuta antes src.audit.embeddings_tsne.")
    d = np.load(EMB, allow_pickle=True)
    X = d["X"]
    labels = d["labels"]
    datasets = np.array([str(x) for x in d["datasets"]])
    print(f"Embeddings: {X.shape}, datasets: {sorted(set(datasets))}")

    from sklearn.metrics import silhouette_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict

    Xs = StandardScaler().fit_transform(X)

    result = {}

    # ---- 1. Silhouette intra-clase ----
    def intraclass_silhouette(class_label: int, ds_a: str, ds_b: str):
        mask = (labels == class_label) & np.isin(datasets, [ds_a, ds_b])
        sub_X = Xs[mask]
        sub_ds = datasets[mask]
        if len(set(sub_ds)) < 2:
            return None
        codes = (sub_ds == ds_b).astype(int)
        sil = float(silhouette_score(sub_X, codes))
        # separabilidad por LogReg con CV (AUC de predecir ds_b vs ds_a)
        n = len(codes)
        cv = min(5, int(min(np.bincount(codes))))
        if cv >= 2:
            proba = cross_val_predict(LogisticRegression(max_iter=2000), sub_X, codes,
                                      cv=cv, method="predict_proba")[:, 1]
            auc = float(roc_auc_score(codes, proba))
        else:
            auc = float("nan")
        return {"datasets": [ds_a, ds_b], "n": int(n),
                "silhouette": sil, "logreg_cv_auc": auc}

    # detectar nombres reales
    names = set(datasets)
    def pick(*c):
        for x in c:
            if x in names:
                return x
        return None
    brats = pick("brats"); upenn = pick("upenn")
    ixi = pick("ixi"); nki = pick("nki_rockland", "nki")

    result["intraclass_sanos_IXI_vs_NKI"] = intraclass_silhouette(0, ixi, nki)
    result["intraclass_tumor_BraTS_vs_UPENN"] = intraclass_silhouette(1, brats, upenn)

    # ---- 2. Clasificador de dataset (4 clases) sobre embeddings ----
    ds_codes = np.array([sorted(set(datasets)).index(x) for x in datasets])
    pred = cross_val_predict(LogisticRegression(max_iter=2000), Xs, ds_codes, cv=5)
    acc = float((pred == ds_codes).mean())
    result["dataset_classifier_from_embeddings"] = {
        "n_classes": len(set(datasets)),
        "cv_accuracy": acc,
        "chance_level": round(1.0 / len(set(datasets)), 3),
    }

    # ---- referencia: silhouette global (recordatorio) ----
    sil_glob = json.loads((REPO_ROOT / "docs" / "audit" / "embeddings_silhouette.json").read_text())
    result["global_silhouette_reference"] = sil_glob

    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n================ ANALISIS INTRA-CLASE ================")
    for k in ("intraclass_sanos_IXI_vs_NKI", "intraclass_tumor_BraTS_vs_UPENN"):
        r = result[k]
        if r:
            print(f"{k}: n={r['n']}  silhouette={r['silhouette']:.3f}  LogReg CV-AUC={r['logreg_cv_auc']:.3f}")
    dc = result["dataset_classifier_from_embeddings"]
    print(f"Clasificador de dataset (4 clases) desde embeddings: acc={dc['cv_accuracy']:.3f} (azar={dc['chance_level']})")
    print(f"\n(Referencia global: por etiqueta={sil_glob['silhouette_by_label']:.3f}, por dataset={sil_glob['silhouette_by_dataset']:.3f})")
    print(f"JSON -> {OUT}")


if __name__ == "__main__":
    main()
