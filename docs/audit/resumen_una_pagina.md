# TFG 007 — Resumen de una página

**Ignacio Sánchez Calabrese · Dir. Jorge Calvo Martín · GMAT (UAX)**

**Título propuesto:** *Priorización automática de RM cerebral mediante aprendizaje
profundo: análisis crítico del sesgo de dominio en datasets públicos multi-fuente.*

**Motivación (objetivo clínico, sin cambios).** Triaje automático de RM cerebral
para priorizar estudios con sospecha de masa tumoral. Clasificación binaria
tumor/no-tumor a nivel de estudio, con T1/T2 y métrica principal la sensibilidad.

**Amenaza metodológica central.** Al construir el problema con datasets públicos
multi-fuente, la clase queda **completamente correlacionada con el dataset de
origen**: positivos = BraTS + UPENN-GBM; negativos = IXI + NKI Rockland
(`P(tumor | BraTS,UPENN)=1`, `P(tumor | IXI,NKI)=0`). Esto convierte cualquier
métrica global en ambigua: "detectar tumor" y "reconocer el escáner" son
indistinguibles sobre estos datos.

## Tabla principal de resultados

| Régimen de evaluación | Modelo | AUC | IC95% / nota |
|---|---|---|---|
| Mezcla aleatoria (4 datasets) | CNN 3D (modelo del TFG) | **1.000** | punto de partida *sospechoso* |
| Mezcla aleatoria (4 datasets) | Baseline trivial (LogReg, 16 stats intensidad) | **1.000** | sin red ni anatomía |
| Cross-dataset LODO A (BraTS+IXI → UPENN+NKI) | CNN 3D | 0.624 | cae fuera de dominio |
| Cross-dataset LODO B (UPENN+NKI → BraTS+IXI) | CNN 3D | 0.201 | **regla invertida** (sen 0.01) |
| **Intra-dominio BTC_preop (Ghent, mismo escáner)** | CNN 3D (T1, 5-fold) | **0.404** | **IC95% [0.21, 0.62] → azar** |
| Intra-dominio BTC_preop | Baseline trivial | 0.549 | IC95% [0.32, 0.79] → azar |
| Embeddings: IXI vs NKI (ambos sanos) | LogReg sobre latente | **0.998** | la red codifica procedencia |
| Embeddings: clasificar dataset (4 clases) | LogReg sobre latente | acc 0.982 | azar = 0.25 |

## Los cinco resultados (estructura de discusión)

1. **La CNN 3D obtiene métricas aparentemente perfectas** (AUC 1.0, sen 0.99, spe 1.0).
2. **Un baseline trivial también alcanza AUC = 1.0** → la señal discriminativa no
   requiere anatomía; basta la firma de intensidad/preprocesado de cada fuente.
3. **La evaluación leave-one-dataset-out muestra caída e inversión** del rendimiento
   (0.62 y 0.20) → el modelo no generaliza fuera de los dominios de entrenamiento.
4. **La validación intra-dominio no confirma capacidad real de detección** con el
   tamaño disponible (AUC 0.40, IC95% cruza 0.5; n=36).
5. **Interpretación:** el modelo aprende dominio/procedencia, no necesariamente tumor.
   Confirmado a nivel de representación: el espacio latente distingue dos cohortes
   *sanas* (IXI/NKI) con AUC 0.998.

**Hipótesis alternativas descartadas con evidencia:** fuga por partición (0 solape de
sujeto), duplicados (0 hashes cruzados), errores de métrica (cálculo verificado).

## Figuras principales (en `figures/principales/`)

1. `auc_summary.png` — desplome del AUC al eliminar el confound (con IC95%).
2. `embeddings_tsne.png` — el latente agrupa por procedencia (IXI≠NKI pese a ser ambos sanos).
3. `confusion_matrices.png` — diagonal perfecta → colapso/inversión → azar.
4. `intensity_by_dataset.png` — el confound ya es visible en los píxeles crudos.

*Figuras de apoyo en `figures/anexo/`:* `roc_curves.png`, `score_hist_*.png`,
`btc_kfold_bars.png`, `embeddings_pca.png`, y `gradcam/` (análisis exploratorio;
los mapas resaltan bordes/periferia de forma no concluyente — se reporta con cautela).

## Conclusión / contribución

El valor del trabajo **no es desplegar un clasificador final**, sino **demostrar de
forma reproducible cómo se infla el rendimiento cuando clase y procedencia están
confundidas**, y proponer un **protocolo de auditoría anti-confounding** (baseline
trivial → LODO → validación intra-dominio → análisis de embeddings → descarte de
leakage) aplicable a futuros sistemas de IA en RM cerebral. Se subraya la necesidad
de **controles intra-dominio**, **validación externa real** y **auditorías de sesgo**
antes de reportar métricas en imagen médica.

**Trabajo futuro:** replicar la validación intra-dominio en T1+T2 con la cohorte de
Edinburgh (acceso ya aprobado) y en cohortes de mayor tamaño.

---
*Cifras trazables a `docs/audit/resumen_consolidado.md` y a los JSON en
`docs/audit/` y `outputs/evaluation/`. Reproducible con seed=42.*
