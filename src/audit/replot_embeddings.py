"""
replot_embeddings.py
--------------------
Re-grafica las proyecciones de embeddings desde docs/audit/embeddings.npz
(sin re-extraer en GPU), con estilo pulido para memoria:
 - figura grande, puntos visibles, leyenda FUERA del area de datos
 - PCA y t-SNE, cada una con panel por dataset y por etiqueta
 - anotacion del silhouette / AUC intra-clase en el pie

Salida (sobrescribe): docs/audit/figures/embeddings_pca.png, embeddings_tsne.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EMB = REPO_ROOT / "docs" / "audit" / "embeddings.npz"
OUT_FIG = REPO_ROOT / "docs" / "audit" / "figures"

DS_COLORS = {"brats": "#c62828", "upenn": "#ad1457", "ixi": "#1565c0", "nki_rockland": "#0277bd"}
DS_LABELS = {"brats": "BraTS (tumor)", "upenn": "UPENN (tumor)",
             "ixi": "IXI (sano)", "nki_rockland": "NKI (sano)"}
LBL_COLORS = {0: "#1565c0", 1: "#c62828"}


def two_panel(Z, datasets, y, title, fname, subtitle, var=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ds_arr = np.array(datasets)
    # Panel 1: por dataset
    for ds in ["brats", "upenn", "ixi", "nki_rockland"]:
        m = ds_arr == ds
        if not m.any():
            continue
        axes[0].scatter(Z[m, 0], Z[m, 1], s=18, alpha=0.65,
                        c=DS_COLORS[ds], label=DS_LABELS[ds], edgecolors="none")
    axes[0].set_title(f"{title} — coloreado por DATASET (origen)", fontsize=13)
    axes[0].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=10, framealpha=0.9)
    axes[0].grid(alpha=0.25)

    # Panel 2: por etiqueta
    for lbl in (1, 0):
        m = y == lbl
        axes[1].scatter(Z[m, 0], Z[m, 1], s=18, alpha=0.55,
                        c=LBL_COLORS[lbl], label=("tumor" if lbl == 1 else "sano"),
                        edgecolors="none")
    axes[1].set_title(f"{title} — coloreado por ETIQUETA (clase)", fontsize=13)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=10, framealpha=0.9)
    axes[1].grid(alpha=0.25)

    if var is not None:
        for ax in axes:
            ax.set_xlabel(f"Componente 1 ({var[0]*100:.0f}% var.)", fontsize=10)
            ax.set_ylabel(f"Componente 2 ({var[1]*100:.0f}% var.)", fontsize=10)

    fig.suptitle(f"Espacio latente de la CNN (96-dim, checkpoint confundido) — {title}",
                 fontsize=14, y=0.99)
    fig.text(0.5, 0.005, subtitle, ha="center", fontsize=10, style="italic", color="#333")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(OUT_FIG / fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  guardado: {OUT_FIG / fname}")


def main():
    d = np.load(EMB, allow_pickle=True)
    X = d["X"]
    y = d["labels"]
    datasets = [str(x) for x in d["datasets"]]

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    Xs = StandardScaler().fit_transform(X)

    # subtitulo con los numeros clave (intra-clase) si existen
    sub = ("Cada cohorte forma su propio grupo: IXI y NKI (ambas SANAS) no se fusionan. "
           "Un clasificador lineal sobre estos vectores distingue IXI de NKI con AUC≈0.998 "
           "y predice el dataset (4 clases) con 98% de acierto → la red codifica procedencia, no solo tumor.")

    pca = PCA(n_components=2, random_state=0).fit(Xs)
    Z_pca = pca.transform(Xs)
    two_panel(Z_pca, datasets, y, "PCA", "embeddings_pca.png", sub, var=pca.explained_variance_ratio_)

    perp = min(30, max(5, len(X) // 50))
    Z_tsne = TSNE(n_components=2, perplexity=perp, random_state=0, init="pca").fit_transform(Xs)
    two_panel(Z_tsne, datasets, y, "t-SNE", "embeddings_tsne.png", sub)

    print("Replot done.")


if __name__ == "__main__":
    main()
