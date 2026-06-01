"""
make_plots.py
-------------
Figuras para la memoria, generadas a partir de los JSONs ya producidos.

Salida: docs/audit/figures/
  - roc_curves.png         curvas ROC superpuestas (confounded, LODO A, LODO B, Ghent)
  - auc_summary.png        barras con AUC + IC95% por experimento
  - score_hist_confound.png histogramas de score por dataset (run confundido)
  - score_hist_lodo.png    histogramas de score por dataset (LODO A y B)
  - btc_kfold_bars.png     AUC por fold (Ghent intra-dominio)
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
OUT_DIR = REPO_ROOT / "docs" / "audit" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json_safe(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_predictions_csv(p: Path) -> tuple[list[int], list[float], list[str]] | None:
    if not p.exists():
        return None
    labels: list[int] = []
    scores: list[float] = []
    datasets: list[str] = []
    with open(p, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                labels.append(int(row["label"]))
                scores.append(float(row["score"]))
                datasets.append(row.get("dataset", "?"))
            except Exception:
                continue
    if not labels:
        return None
    return labels, scores, datasets


def roc_curve(y_true: list[int], y_score: list[float]):
    y = np.asarray(y_true)
    s = np.asarray(y_score)
    thresholds = np.unique(np.concatenate(([0.0], s, [1.0 + 1e-9])))[::-1]
    tprs = []
    fprs = []
    P = max((y == 1).sum(), 1)
    N = max((y == 0).sum(), 1)
    for t in thresholds:
        yp = (s >= t).astype(int)
        tp = int(((yp == 1) & (y == 1)).sum())
        fp = int(((yp == 1) & (y == 0)).sum())
        tprs.append(tp / P)
        fprs.append(fp / N)
    return np.array(fprs), np.array(tprs)


def auc_from_roc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    order = np.argsort(fpr)
    return float(np.trapezoid(tpr[order], fpr[order]))


def plot_roc_curves() -> None:
    sources = {
        "Confounded mix (CNN, AUC≈1.0)":
            REPO_ROOT / "outputs" / "evaluation" / "cnn3d_test_predictions.csv",
        "LODO A (CNN, AUC=0.62)":
            REPO_ROOT / "outputs" / "evaluation" / "lodo_A" / "cnn3d_test_predictions.csv",
        "LODO B (CNN, AUC=0.20)":
            REPO_ROOT / "outputs" / "evaluation" / "lodo_B" / "cnn3d_test_predictions.csv",
    }
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, path in sources.items():
        d = load_predictions_csv(path)
        if d is None:
            continue
        y, s, _ = d
        fpr, tpr = roc_curve(y, s)
        auc = auc_from_roc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{label}  (AUC={auc:.3f})", linewidth=2)

    # Ghent CNN k-fold (predicciones agregadas)
    ghent = load_json_safe(REPO_ROOT / "outputs" / "evaluation" / "btc_intradomain" / "cnn_kfold_results.json")
    if ghent:
        all_scores = []
        all_labels = []
        for f in ghent["per_fold"]:
            all_scores.extend(f["scores"])
            all_labels.extend(f["labels"])
        fpr, tpr = roc_curve(all_labels, all_scores)
        auc = auc_from_roc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"Ghent intra-dominio (CNN k-fold, AUC={auc:.3f})", linewidth=2, color="purple")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Azar")
    ax.set_xlabel("False Positive Rate (1 - especificidad)")
    ax.set_ylabel("True Positive Rate (sensibilidad)")
    ax.set_title("ROC: confounded vs LODO vs intra-dominio")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)
    print(f"  guardado: {OUT_DIR / 'roc_curves.png'}")


def plot_auc_summary() -> None:
    """Barras con AUC + IC95% para cada experimento clave."""
    items = [
        ("Mezcla\n(CNN)",            1.0000, None, "#c62828"),
        ("Tiny baseline\nmezcla",    1.0000, None, "#c62828"),
        ("LODO A\n(CNN)",            0.6236, None, "#f57c00"),
        ("LODO B\n(CNN)",            0.2012, None, "#f57c00"),
        ("Ghent intra\n(tiny LR)",   0.5491, (0.3192, 0.7882), "#2e7d32"),
        ("Ghent intra\n(CNN)",       0.4036, (0.2129, 0.6231), "#1565c0"),
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(len(items))
    aucs = [it[1] for it in items]
    cis = [it[2] for it in items]
    colors = [it[3] for it in items]

    bars = ax.bar(xs, aucs, color=colors, alpha=0.85, edgecolor="black")
    # Error bars donde haya IC
    yerr_lo, yerr_hi = [], []
    for auc, ci in zip(aucs, cis):
        if ci is None:
            yerr_lo.append(0); yerr_hi.append(0)
        else:
            yerr_lo.append(auc - ci[0])
            yerr_hi.append(ci[1] - auc)
    ax.errorbar(xs, aucs, yerr=[yerr_lo, yerr_hi], fmt="none", ecolor="black", capsize=4, lw=1.2)

    ax.axhline(0.5, color="black", linestyle="--", alpha=0.5, label="Azar (0.5)")
    ax.set_xticks(xs)
    ax.set_xticklabels([it[0] for it in items], fontsize=9)
    ax.set_ylabel("AUC")
    ax.set_title("AUC por experimento (barras IC95% cuando aplica)")
    ax.set_ylim(0, 1.05)
    for x, auc in zip(xs, aucs):
        ax.text(x, auc + 0.02, f"{auc:.3f}", ha="center", fontsize=9)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "auc_summary.png", dpi=150)
    plt.close(fig)
    print(f"  guardado: {OUT_DIR / 'auc_summary.png'}")


def plot_score_hist_confound() -> None:
    """Histograma de scores por dataset en el run confundido."""
    d = load_predictions_csv(REPO_ROOT / "outputs" / "evaluation" / "cnn3d_test_predictions.csv")
    if d is None:
        return
    labels, scores, datasets = d
    fig, ax = plt.subplots(figsize=(8, 5))
    by_ds: dict[str, list[float]] = {}
    for s, ds in zip(scores, datasets):
        by_ds.setdefault(ds, []).append(s)
    palette = {"brats": "#c62828", "upenn": "#ad1457", "ixi": "#1565c0", "nki_rockland": "#0277bd"}
    bins = np.linspace(0, 1, 41)
    for ds in sorted(by_ds):
        ax.hist(by_ds[ds], bins=bins, alpha=0.6, label=f"{ds} (n={len(by_ds[ds])})",
                color=palette.get(ds, "#666"), edgecolor="black", linewidth=0.5)
    ax.axvline(0.5, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Score del modelo (P(tumor))")
    ax.set_ylabel("Sujetos")
    ax.set_title("Distribución de scores por dataset — run confundido (AUC ≈ 1.0)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "score_hist_confound.png", dpi=150)
    plt.close(fig)
    print(f"  guardado: {OUT_DIR / 'score_hist_confound.png'}")


def plot_score_hist_lodo() -> None:
    """Histogramas LODO A y B lado a lado."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    palette = {"brats": "#c62828", "upenn": "#ad1457", "ixi": "#1565c0", "nki_rockland": "#0277bd"}
    for ax, cfg in zip(axes, ["A", "B"]):
        d = load_predictions_csv(REPO_ROOT / "outputs" / "evaluation" / f"lodo_{cfg}" / "cnn3d_test_predictions.csv")
        if d is None:
            ax.set_title(f"LODO {cfg} (sin datos)")
            continue
        labels, scores, datasets = d
        by_ds: dict[str, list[float]] = {}
        for s, ds in zip(scores, datasets):
            by_ds.setdefault(ds, []).append(s)
        bins = np.linspace(0, 1, 41)
        for ds in sorted(by_ds):
            ax.hist(by_ds[ds], bins=bins, alpha=0.6, label=f"{ds} (n={len(by_ds[ds])})",
                    color=palette.get(ds, "#666"), edgecolor="black", linewidth=0.5)
        ax.axvline(0.5, color="black", linestyle="--", alpha=0.5)
        ax.set_xlabel("Score del modelo (P(tumor))")
        ax.set_title(f"LODO {cfg} — distribución de scores en test (dominios no vistos)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Sujetos")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "score_hist_lodo.png", dpi=150)
    plt.close(fig)
    print(f"  guardado: {OUT_DIR / 'score_hist_lodo.png'}")


def plot_btc_kfold_bars() -> None:
    """Barras AUC por fold del CNN intra-dominio."""
    d = load_json_safe(REPO_ROOT / "outputs" / "evaluation" / "btc_intradomain" / "cnn_kfold_results.json")
    if d is None:
        return
    folds = d["per_fold"]
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [f["fold"] for f in folds]
    aucs = [f["auc"] for f in folds]
    bars = ax.bar(xs, aucs, color="#1565c0", alpha=0.85, edgecolor="black")
    ax.axhline(0.5, color="black", linestyle="--", alpha=0.5, label="Azar")
    ax.axhline(d["overall_auc"], color="purple", linestyle="-", alpha=0.8,
               label=f"Agregado: AUC = {d['overall_auc']:.3f}")
    ci = d.get("overall_auc_ci95")
    if ci:
        ax.axhspan(ci[0], ci[1], color="purple", alpha=0.15, label=f"IC95% [{ci[0]:.2f}, {ci[1]:.2f}]")
    for x, a, f in zip(xs, aucs, folds):
        ax.text(x, a + 0.02, f"{a:.2f}\n(n={f['n_test']})", ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xlabel("Fold")
    ax.set_ylabel("AUC en el fold held-out")
    ax.set_title("CNN 3D 1-canal intra-dominio Ghent (n=36, 5-fold por sujeto)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "btc_kfold_bars.png", dpi=150)
    plt.close(fig)
    print(f"  guardado: {OUT_DIR / 'btc_kfold_bars.png'}")


def main() -> None:
    print(f"Generando figuras en {OUT_DIR}...")
    plot_roc_curves()
    plot_auc_summary()
    plot_score_hist_confound()
    plot_score_hist_lodo()
    plot_btc_kfold_bars()
    print("Done.")


if __name__ == "__main__":
    main()
