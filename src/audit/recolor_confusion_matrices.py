"""
recolor_confusion_matrices.py
-----------------------------
Re-renderiza la figura E3 (matrices de confusion por regimen) con color de
texto legible: blanco sobre las celdas oscuras de la paleta "Blues" y negro
sobre las claras. La version anterior usaba texto negro fijo, ilegible en las
celdas con conteos altos (diagonal).

Los conteos son los del run oficial recuperado (identicos a la figura previa);
se fijan aqui porque las predicciones crudas de LODO A/B y Ghent intra-dominio
no estan en este equipo (la red se entreno en la UAX). Solo cambia el color del
texto, no los datos ni la paleta.

Salida (la misma figura en las dos ubicaciones que consume el PDF / la auditoria):
  memoria_tfg/figuras/confusion_matrices.png
  docs/audit/figures/confusion_matrices.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATHS = [
    REPO_ROOT / "memoria_tfg" / "figuras" / "confusion_matrices.png",
    REPO_ROOT / "docs" / "audit" / "figures" / "confusion_matrices.png",
]

# cm[i][j] con i = clase real (sano, tumor), j = prediccion (sano, tumor).
PANELS = [
    ("Confounded (mix)\nAUC≈1.0", np.array([[165, 0], [1, 174]])),
    ("LODO A\nAUC=0.62", np.array([[242, 281], [190, 397]])),
    ("LODO B\nAUC=0.20", np.array([[557, 20], [574, 6]])),
    ("Ghent intra-dominio\nAUC=0.40", np.array([[4, 7], [10, 15]])),
]


def main() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, (title, cm) in zip(axes, PANELS):
        ax.imshow(cm, cmap="Blues")
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                txt_color = "white" if cm[i, j] > thresh else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=14, color=txt_color, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred sano", "pred tumor"], fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["real sano", "real tumor"], fontsize=8)
        ax.set_title(title, fontsize=10)
    fig.suptitle("E3 — Matrices de confusión por régimen (umbral 0.5)", fontsize=12)
    fig.tight_layout()
    for out in OUT_PATHS:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f"  guardado: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
