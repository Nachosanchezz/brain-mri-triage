"""
make_extra_figures.py  (E2 + E3)
--------------------------------
E2: distribucion de estadisticos de intensidad por dataset (boxplots) desde
    docs/audit/audit_features.csv (ya generado por audit_leakage.py).
E3: matrices de confusion 2x2 por experimento (confounded, LODO A, LODO B,
    Ghent agregado) desde los _results.json / predictions.csv.

Salida:
  docs/audit/figures/intensity_by_dataset.png
  docs/audit/figures/confusion_matrices.png
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_FIG = REPO_ROOT / "docs" / "audit" / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

DS_ORDER = ["brats", "upenn", "ixi", "nki_rockland"]
DS_COLORS = {"brats": "#c62828", "upenn": "#ad1457", "ixi": "#1565c0", "nki_rockland": "#0277bd"}


# ---------------- E2 ----------------
def plot_intensity_by_dataset():
    csv_path = REPO_ROOT / "docs" / "audit" / "audit_features.csv"
    if not csv_path.exists():
        print(f"  AVISO: no existe {csv_path}; salto E2")
        return
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    feats = ["nz_frac_t1", "mean_t1", "p50_t1", "p99_t1"]
    titles = {"nz_frac_t1": "Fracción vóxeles no-cero (T1)",
              "mean_t1": "Intensidad media (T1)",
              "p50_t1": "Mediana intensidad (T1)",
              "p99_t1": "Percentil 99 intensidad (T1)"}
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, feat in zip(axes, feats):
        data, labels, colors = [], [], []
        for ds in DS_ORDER:
            vals = [float(r[feat]) for r in rows if r["dataset"] == ds and r.get(feat)]
            if vals:
                data.append(vals); labels.append(ds); colors.append(DS_COLORS[ds])
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.6)
        ax.set_title(titles[feat], fontsize=10)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("E2 — Distribución de intensidades por dataset (el confound es visible en los píxeles crudos)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "intensity_by_dataset.png", dpi=150)
    plt.close(fig)
    print(f"  guardado: {OUT_FIG / 'intensity_by_dataset.png'}")


# ---------------- E3 ----------------
def confusion_from_predictions(csv_path: Path, thr: float = 0.5):
    if not csv_path.exists():
        return None
    tp = tn = fp = fn = 0
    for r in csv.DictReader(open(csv_path, encoding="utf-8")):
        try:
            y = int(r["label"]); s = float(r["score"])
        except Exception:
            continue
        p = 1 if s >= thr else 0
        if p == 1 and y == 1: tp += 1
        elif p == 0 and y == 0: tn += 1
        elif p == 1 and y == 0: fp += 1
        else: fn += 1
    return np.array([[tn, fp], [fn, tp]])


def confusion_from_kfold(json_path: Path, thr: float = 0.5):
    if not json_path.exists():
        return None
    d = json.loads(json_path.read_text(encoding="utf-8"))
    tp = tn = fp = fn = 0
    for f in d["per_fold"]:
        for s, y in zip(f["scores"], f["labels"]):
            p = 1 if s >= thr else 0
            if p == 1 and y == 1: tp += 1
            elif p == 0 and y == 0: tn += 1
            elif p == 1 and y == 0: fp += 1
            else: fn += 1
    return np.array([[tn, fp], [fn, tp]])


def plot_confusion_matrices():
    panels = [
        ("Confounded (mix)\nAUC≈1.0", confusion_from_predictions(
            REPO_ROOT / "outputs" / "evaluation" / "cnn3d_test_predictions.csv")),
        ("LODO A\nAUC=0.62", confusion_from_predictions(
            REPO_ROOT / "outputs" / "evaluation" / "lodo_A" / "cnn3d_test_predictions.csv")),
        ("LODO B\nAUC=0.20", confusion_from_predictions(
            REPO_ROOT / "outputs" / "evaluation" / "lodo_B" / "cnn3d_test_predictions.csv")),
        ("Ghent intra-dominio\nAUC=0.40", confusion_from_kfold(
            REPO_ROOT / "outputs" / "evaluation" / "btc_intradomain" / "cnn_kfold_results.json")),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, (title, cm) in zip(axes, panels):
        if cm is None:
            ax.set_title(f"{title}\n(sin datos)"); ax.axis("off"); continue
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=14, color="black",
                        fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred sano", "pred tumor"], fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["real sano", "real tumor"], fontsize=8)
        ax.set_title(title, fontsize=10)
    fig.suptitle("E3 — Matrices de confusión por régimen (umbral 0.5)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "confusion_matrices.png", dpi=150)
    plt.close(fig)
    print(f"  guardado: {OUT_FIG / 'confusion_matrices.png'}")


def main():
    print("E2: intensidades por dataset...")
    plot_intensity_by_dataset()
    print("E3: matrices de confusion...")
    plot_confusion_matrices()
    print("Done E2+E3.")


if __name__ == "__main__":
    main()
