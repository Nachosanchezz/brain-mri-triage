# Resumen consolidado de la auditoria

Tabla unica con todos los resultados ordenados de mas confundido a mas honesto.

## Tabla maestra

| Experimento | Modelo | AUC | IC95% | Notas |
|---|---|---|---|---|
| Mixed (4 datasets, split aleatorio) | Tiny baseline LogReg | **1.0000** | — | atajo trivial de intensidad |
| Mixed (4 datasets, split aleatorio) | Tiny baseline RF     | **0.9989** | — | idem |
| Mixed (4 datasets, split aleatorio) | CNN 3D 2-canales     | **1.0000** | — | confounded baseline |
| LODO A: BraTS+IXI -> UPENN+NKI | Tiny baseline LogReg | 0.9952 | — | |
| LODO A: BraTS+IXI -> UPENN+NKI | **CNN 3D 2-canales** | **0.6236** | — | sen=0.6763  spe=0.4627 |
| LODO B: UPENN+NKI -> BraTS+IXI | Tiny baseline LogReg | 0.3184 | — | |
| LODO B: UPENN+NKI -> BraTS+IXI | **CNN 3D 2-canales** | **0.2012** | — | sen=0.0103  spe=0.9653 |
| **Ghent intra-dominio (n=36, k-fold)** | Tiny baseline LogReg | **0.5491** | [0.3192, 0.7882] | T1-only |
| **Ghent intra-dominio (n=36, k-fold)** | Tiny baseline RF     | **0.4055** | [0.2154, 0.6161] | T1-only |
| **Ghent intra-dominio (n=36, k-fold)** | **CNN 3D 1-canal**   | **0.4036** | [0.2129, 0.6231] | sen=0.6000  spe=0.3636 |

## Embeddings (E1) — espacio latente de la CNN

- silhouette global: por etiqueta = 0.754  |  por dataset = 0.366 (diluido)
- **intra-clase (huella de procedencia aislada):** IXI vs NKI (sanos) LogReg CV-AUC = 0.998; BraTS vs UPENN (tumor) = 0.991
- clasificador de dataset (4 clases) desde embeddings: acc = 0.982 (azar 0.25)
- Conclusion: el latente identifica el centro de origen incluso entre sujetos de la MISMA clase -> codifica procedencia, no solo tumor. Ver `figures/embeddings_tsne.png`.

## Lectura sugerida

- **Mixed AUC ~ 1.0** (CNN y baseline lineal): confound trivial.
- **LODO**: caos asimetrico, AUC < 0.5 en algunos casos = predicciones invertidas. Firma de dataset, no tumor.
- **Ghent intra-dominio**: con dominio controlado, AUC honesto con IC95%. ESE es el numero entregable.
- **Embeddings**: el espacio latente codifica origen del dato mas alla de la clase.

## Figuras (docs/audit/figures/)

auc_summary, roc_curves, score_hist_confound, score_hist_lodo, confusion_matrices, btc_kfold_bars, embeddings_tsne, embeddings_pca, intensity_by_dataset, gradcam/{confound,lodo_A,lodo_B}/*

_Generado por `src/audit/consolidate_results.py`_
