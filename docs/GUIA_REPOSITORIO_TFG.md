# Guía maestra del repositorio `brain-mri-triage`

> Documento de auditoría documental generado el **2026-06-06**. Sirve como
> *mapa maestro* del repositorio para redactar la memoria del TFG con orden.
> **No describe trabajo nuevo ni mueve/borra nada**: solo clasifica lo que existe.
> Todas las cifras son trazables a los JSON de `docs/audit/`; donde una cifra no
> se ha podido localizar en un archivo versionado se escribe **"no localizado"**.
>
> Convención de estado usada en todo el documento:
> **principal** (guía la memoria) · **auxiliar** (apoyo, no se cita como evidencia) ·
> **legacy** (antiguo/sustituido) · **experimental** (exploratorio, tratar con cautela) ·
> **no tocar** (institucional / ignorado por git).

---

## 1. Resumen general del proyecto

TFG del Grado en Ingeniería Matemática (UAX) sobre **triaje automático de
resonancia magnética (RM) cerebral**: priorizar estudios con sospecha de masa
tumoral mediante **clasificación binaria tumor / no-tumor** a nivel de estudio,
con una **CNN 3D** entrenada sobre datos públicos multi-fuente (BraTS 2021,
UPENN-GBM, IXI y NKI Rockland).

El proyecto sufrió un **giro metodológico**: el modelo alcanza un rendimiento
aparentemente perfecto (AUC ≈ 1.0), pero una **auditoría sistemática** demuestra
que ese rendimiento se debe a un **confound estructural dominio↔clase**
(`label = 1 ⟺ dataset ∈ {BraTS, UPENN}`; `label = 0 ⟺ dataset ∈ {IXI, NKI}`).
Es decir, "detectar tumor" y "reconocer el escáner de origen" son
**indistinguibles** sobre estos datos (*shortcut learning*).

**Idea central de la memoria:** el rendimiento aparente está íntegramente
afectado por el confound dominio-clase; cuando se controla (validación
intra-dominio sobre BTC_preop, mismo escáner), el modelo rinde **al azar**. La
contribución no es desplegar un clasificador, sino **caracterizar de forma
reproducible el aprendizaje espurio y proponer un protocolo de auditoría
anti-confounding** (tiny baseline → LODO → intra-dominio → embeddings → descarte
de leakage).

---

## 2. Mapa general de carpetas

| Carpeta | Qué contiene | Para qué sirve | Importancia memoria | Estado |
|---|---|---|---|---|
| `configs/` | 2 YAML de entrenamiento (`train_3d.yaml`, `train_lodo.yaml`) | Hiperparámetros y rutas del pipeline CNN | Alta (cap. Metodología / Diseño experimental) | **principal** |
| `src/` | Todo el código Python (datos, modelo, entrenamiento, evaluación, preprocesado, auditoría) | Núcleo reproducible del TFG | Muy alta | **principal** |
| `scripts/` | Lanzadores `.sh` (LODO, cadena BTC) y `gen_tablas_pdf.py` | Orquestación de experimentos y tablas para el tutor | Media (reproducibilidad / anexos) | **auxiliar** |
| `docs/` | Documentación, auditoría, narrativa, figuras, entregas | Materia prima textual de la memoria | Muy alta | **principal** |
| `docs/audit/` | JSON de resultados, figuras, borradores de memoria | **Fuente de verdad de las cifras y figuras** | Máxima | **principal** |
| `notebooks/` | 2 notebooks exploratorios | Inspección puntual; no son resultados oficiales | Baja | **experimental** |
| `memoria_tfg/` | Proyecto LaTeX de la memoria (plantilla UAX) | Documento que se entrega | Máxima (es el entregable) | **no tocar** (institucional) |
| `data/` | Datos crudos y `.npz` preprocesados (`raw/`, `processed/`, `processed_btc/`, `splits.json`) | Datos de entrada | Solo se cita su composición | **no tocar** (ignorado por git: `/data/`) |
| `outputs/` | Checkpoints, evaluaciones (`evaluation/*.json`), plots | Resultados crudos de cada run | Se citan rutas, no se versiona | **no tocar** (ignorado por git: `outputs/`) |
| `env/` | Entorno virtual Python | Dependencias instaladas | Ninguna | **no tocar** (ignorado: `env/`) |

> **Nota de git** (`.gitignore`): `data/`, `outputs/`, `env/`, `.claude/`,
> `CLAUDE.md` y artefactos Python están **ignorados**. Lo que guía la memoria
> y se versiona es `src/`, `configs/`, `scripts/`, `docs/` y `memoria_tfg/`.

---

## 3. Archivos centrales del TFG

| Archivo | Función | Capítulo memoria | Evidencia que aporta | Comentario |
|---|---|---|---|---|
| `configs/train_3d.yaml` | Config del run **confundido** (CNN 2 canales T1+T2, 4 datasets) | 5 Metodología / 6 Diseño exp. | Hiperparámetros reproducibles (seed 42, crop 128×160×128, lr 5e-5) | `recreate_splits: true` (split se regenera cada run; determinista por seed) |
| `configs/train_lodo.yaml` | Config LODO; copia idéntica salvo `recreate_splits:false` y `checkpoint_dir` propio | 6 Diseño exp. | Comparación *apples-to-apples* confound vs LODO | Lee el `splits.json` que escribe `make_lodo_splits.py` |
| `src/models/cnn3d.py` | Arquitectura `BrainTumorCNN3D` (compacta, GroupNorm) | 5 Metodología | Define el modelo; embedding de 96-dim tras `AdaptiveAvgPool3d` | Mismo modelo en todos los regímenes (confound, LODO, BTC) |
| `src/training/train_3d.py` | Bucle de entrenamiento, early stopping por balanced-acc con sen≥0.80 | 5 Metodología | Selección de modelo por val (sin leak de test) | Núcleo del pipeline |
| `src/evaluation/evaluate_3d.py` | Métricas globales y por dataset (AUC, sen, spe, PR-AUC, matriz confusión) | 7 Resultados | AUC sobre probabilidades sigmoides; orden y_true/y_pred correcto | Código verificado correcto en la auditoría (§6 del doc de auditoría) |
| `src/evaluation/threshold_analysis.py` | Elige umbral en val y lo aplica en test | 6 Diseño exp. | Umbral honesto (no elegido en test) | Auxiliar de evaluación |
| `src/data/dataset_3d.py` | `Dataset` PyTorch + `create_splits` (estratificado por (dataset,label), agrupado por subject) | 4 Materiales / 5 Metodología | Split por sujeto, seed 42; augmentation | Pieza clave de "sin leakage de partición" |
| `src/data/audit_splits.py` | Audita solapamiento de sujetos y composición de splits | 6 Diseño exp. / 8 Discusión | **0 solapamiento** train/val/test | Descarta la hipótesis de leakage por partición |
| `src/audit/audit_leakage.py` | Tiny baseline (LogReg/RF sobre 16 stats de intensidad) + clf de origen + duplicados | 7 Resultados / 8 Discusión | **Tiny baseline AUC=1.0**; clf dataset acc=0.985; 0 duplicados | **Prueba definitiva del confound** |
| `src/audit/audit_lodo.py` | Tiny baseline bajo leave-one-dataset-out (4 configs) | 7 Resultados | Transfer espurio / inversión cross-dataset | Complementa el LODO de la CNN |
| `src/audit/make_lodo_splits.py` | Genera `splits.json` para LODO (configs A/B/C/D) | 6 Diseño exp. | Reproducibilidad del LODO | No toca el pipeline |
| `src/audit/btc_tiny_baseline.py` | Tiny baseline k-fold intra-dominio sobre BTC | 7 Resultados | LogReg AUC 0.5491 [0.32,0.79] → azar | Control trivial intra-dominio |
| `src/audit/btc_cnn_kfold.py` | CNN 3D 1-canal, 5-fold por sujeto, intra-dominio BTC | 7 Resultados | **CNN AUC 0.4036 [0.21,0.62] → azar** | **El número entregable honesto** |
| `src/audit/embeddings_tsne.py` | Extrae embeddings 96-dim del checkpoint confundido, PCA+t-SNE, silhouette | 7 Resultados | silhouette label 0.754 / dataset 0.366 | Genera `embeddings.npz` (reutilizable) |
| `src/audit/embeddings_intraclass.py` | Separabilidad de datasets **dentro de cada clase** + clf de dataset desde embeddings | 7 Resultados / 8 Discusión | IXI vs NKI (sanos) AUC **0.998**; clf dataset acc 0.982 | Aísla la huella de procedencia |
| `src/audit/gradcam_3d.py` | Grad-CAM 3D sobre la CNN | 11 Anexos | Mapas de atención (bordes/periferia) | **Exploratorio / no concluyente** — tratar con cautela |
| `src/audit/consolidate_results.py` | Junta todos los JSON en `resumen_consolidado.{json,md}` | 7 Resultados | Tabla maestra única | Punto de entrada para las cifras |

---

## 4. Archivos de preprocesado (`src/preprocessing/`)

Pipeline común: **HD-BET (skull-strip) → reorient RAS → resample 1 mm³ → crop/pad
192×224×192 → z-score (Otsu sobre tejido)** → guarda `.npz`. La lógica común vive
en `base_preprocessing.py`; cada dataset solo aporta una lista de `DatasetSample`.

| Archivo | Dataset | Entrada esperada | Salida | Estado | ¿En memoria? |
|---|---|---|---|---|---|
| `base_preprocessing.py` | (común) | `DatasetSample` (rutas NIfTI T1/T2 + label) | `.npz` + `preprocessing_summary.json` | **principal** | Sí (cap. 4, describir pipeline una vez) |
| `preprocess_brats.py` | BraTS 2021 | NIfTI T1/T2 BraTS | `data/processed/positives/*.npz` (label 1) | **principal** | Sí (mención) |
| `preprocess_ixi.py` | IXI | NIfTI T1/T2 IXI | `data/processed/negatives/*.npz` (label 0) | **principal** | Sí (mención) |
| `preprocess_upenn.py` | UPENN-GBM | NIfTI T1/T2 UPENN | `data/processed/positives/*.npz` (label 1) | **principal** | Sí (mención) |
| `preprocess_nki_rockland.py` | NKI Rockland | NIfTI T1/T2 NKI | `data/processed/negatives/*.npz` (label 0) | **principal** | Sí (mención) |
| `preprocess_btc.py` | BTC_preop (ds001226) | NIfTI **T1-only** | `data/processed_btc/{positives,negatives}/*.npz` | **principal** | Sí (cap. 4/6, experimento honesto) |
| `download_nki_rockland.py` | NKI Rockland | `aws_links.csv` (S3) | `data/raw/nki_rockland/...` + pairs.csv | **auxiliar** | Opcional (reproducibilidad/anexo) |
| `conversion/dicom_to_nifti.py` | BraTS / UPENN | DICOM | NIfTI | **auxiliar** | Opcional (anexo de preprocesado) |
| `dataset/summarize_processed.py` | (todos) | `.npz` existentes | `preprocessing_summary.json` | **auxiliar** | No (utilidad interna) |
| `preprocess_volumes.py` | **solo BraTS + IXI** | — | lanzador conjunto antiguo | **legacy** | **No** — sustituido por los `preprocess_<dataset>.py` individuales (solo cubre 2 de 5 datasets) |
| `src/preprocessing/README.md` | — | — | doc del preprocesado (288 líneas) | **auxiliar** | Apoyo de redacción del cap. 4 |

> **`preprocess_volumes.py` es el principal candidato a legacy** en preprocesado:
> es un lanzador que solo cubre BraTS+IXI, anterior a la incorporación de UPENN,
> NKI y BTC. No usar como referencia del pipeline actual.

---

## 5. Documentos importantes en `docs/`

| Documento | Qué contiene | ¿Actualizado? | Capítulo | Qué reutilizar |
|---|---|---|---|---|
| `docs/audit/auditoria_resultados_sospechosos.md` (349 l.) | Auditoría técnica completa: confound, splits, duplicados, métricas, entrenamiento, gravedad | **Actualizado** (2026-05-28) | 7, 8 | Tablas de gravedad, "frase metodológica para la memoria" (§11), reproducción |
| `docs/audit/borrador_memoria.md` (406 l.) | **Columna vertebral de Resultados y Discusión**: tabla maestra, narrativa en 5 pasos, frases listas, inventario de figuras, preguntas de tribunal | **Actualizado** (2026-05-31) | 7, 8, 10 | Casi todo: frases en §5, tabla §1, preguntas §6 | 
| `docs/audit/resumen_consolidado.md` + `.json` | Tabla única de todos los AUC + embeddings | **Actualizado** (generado por script) | 7 | Cifras oficiales (fuente de verdad) |
| `docs/audit/resumen_una_pagina.md` (74 l.) | Síntesis de 1 página para el tutor | **Actualizado** | 1 Introducción / 9 Conclusiones | Motivación, conclusión/contribución |
| `docs/01_memoria/indice.md` (414 l.) | Índice/estructura de la memoria | Revisar coherencia con `memoria_tfg/secciones/` | Estructura | Esqueleto de capítulos |
| `docs/01_memoria/justificacion_cambios.md` (362 l.) | Justificación de decisiones de diseño | Revisar fecha | 8 Discusión / 10 | Argumentos de diseño |
| `docs/01_memoria/evolucion_respecto_primera_entrega.md` (416 l.) | Evolución vs primera entrega (p. ej. §1.2: UPENN pasó a train) | Revisar | 8, 10 | Justificación de ausencia de validación externa |
| `docs/01_memoria/reconstruccion_evolucion_tfg.md` (1180 l.) | Reconstrucción larga de la evolución del TFG | **Posiblemente desactualizado/redundante** con los otros dos | — | Revisar antes de usar; puede solapar |
| `docs/01_memoria/material_redaccion_y_tutor.md` (751 l.) | Material de redacción y notas del tutor | Revisar | varios | Apoyo |
| `docs/03_estudio/plan_trabajo_tfg.md` / `referencias.md` | Plan y referencias bibliográficas | Revisar | 3 Estado del arte | **`referencias.md` → poblar `bibliography.bib`** |
| `docs/03_estudio/guia_fomo_amaes.{html,pdf}`, `plan_estudio.html` | Material de estudio externo | Auxiliar | — | No es contenido de la memoria |
| `docs/00_entregas/TFG.pdf` | Entrega previa (PDF) | **Histórico** (entrega anterior) | — | Referencia de qué se entregó antes; **no** es la memoria actual |
| `docs/01_memoria/tablas_resultados.pdf` | PDF de tablas (salida de `gen_tablas_pdf.py`) | Regenerable | 7 / anexo | Tablas para el tutor |

**Documentos a tratar como potencialmente desactualizados/redundantes:**
`reconstruccion_evolucion_tfg.md` (muy largo, puede solapar con
`evolucion_respecto_primera_entrega.md` y `justificacion_cambios.md`). No es
fuente de cifras; verificar antes de citar. **Las cifras oficiales SIEMPRE salen
de los JSON de `docs/audit/`, nunca de los `.md` narrativos.**

---

## 6. Resultados que debo usar en la memoria

> Fuente de verdad: `docs/audit/resumen_consolidado.json` y los JSON individuales.
> Cifras verificadas en esta auditoría.

| Resultado | Archivo fuente | Métrica | Valor | Interpretación correcta | Capítulo |
|---|---|---|---|---|---|
| CNN 3D, pool multi-fuente | `resumen_consolidado.json` (`confound_cnn_test_auc`) | AUC test | **0.9999** | Rendimiento **aparente**; NO es detección de tumor | 7 |
| CNN 3D, pool multi-fuente (sen/spe) | `outputs/evaluation/cnn3d_test_results.json` (citado en borrador) | sen / spe | **0.994 / 1.000** | Síntoma del confound (scores bimodales por dataset) | 7 |
| Baseline trivial intensidad (mezcla) | `audit_leakage.json` (`tiny_baseline_label.logreg`) | AUC / acc | **1.0000 / 0.994** | Etiqueta decodificable **sin red ni anatomía** → confound probado | 7, 8 |
| Baseline trivial RF (mezcla) | `audit_leakage.json` (`rf_test_auc`) | AUC / acc | 0.9989 / 0.988 | Ídem | 7 |
| Identificabilidad del dominio | `audit_leakage.json` (`dataset_origin_classifier`) | acc (4 clases) | **0.985** (azar 0.25) | El origen es trivialmente identificable | 7, 8 |
| LODO A (BraTS+IXI → UPENN+NKI) | `resumen_consolidado.json` | AUC / sen / spe | **0.6236 / 0.676 / 0.463** | Cae fuera de dominio | 7 |
| LODO B (UPENN+NKI → BraTS+IXI) | `resumen_consolidado.json` | AUC / sen / spe | **0.2012 / 0.010 / 0.965** | **Regla invertida** (AUC<0.5) | 7 |
| LODO tiny baseline (4 configs) | `audit_lodo.json` | AUC | 0.995 / 0.318 / 0.041 / 0.047 (logreg) | Caos asimétrico = firma de dominio | 7 |
| BTC_preop intra-dominio (CNN) | `btc_intradomain_tinybaseline.json` / `resumen_consolidado.json` (`btc_cnn_*`) | AUC / IC95% | **0.4036 / [0.213, 0.623]** | **Al azar** (IC cruza 0.5); sen 0.60 spe 0.364 | 7 (número entregable) |
| BTC_preop tiny baseline LogReg | `btc_intradomain_tinybaseline.json` | AUC / IC95% | 0.5491 / [0.319, 0.788] | Al azar | 7 |
| BTC_preop tiny baseline RF | `btc_intradomain_tinybaseline.json` | AUC / IC95% | 0.4055 / [0.215, 0.616] | Al azar | 7 |
| Embeddings silhouette | `embeddings_silhouette.json` | silhouette | label **0.754** / dataset **0.366** | Separación por clase ≡ por dataset (no concluye detección) | 7 |
| Embeddings intra-clase (sanos) | `embeddings_intraclass.json` | LogReg CV-AUC | IXI vs NKI = **0.998** | El latente identifica el centro **entre sujetos sanos** → codifica procedencia | 7, 8 |
| Embeddings intra-clase (tumor) | `embeddings_intraclass.json` | LogReg CV-AUC | BraTS vs UPENN = **0.991** | Ídem entre tumores | 7 |
| Clf dataset desde embeddings | `embeddings_intraclass.json` | acc CV (4 clases) | **0.982** (azar 0.25) | La representación codifica origen | 7, 8 |
| Grad-CAM | `figures/anexo/gradcam/**` | (cualitativo) | — | **No concluyente**: resalta bordes/periferia | 11 Anexos (con cautela) |
| Composición datos | `auditoria_resultados_sospechosos.md` §2 | n por dataset | BraTS 580 / UPENN 587 / IXI 577 / NKI 523 (1167 pos / 1100 neg) | Ningún dataset aporta ambas clases | 4 |
| Composición BTC | `data/raw/btc_preop/participants.tsv` (no leído; en `data/` ignorado) | n | 36 sujetos (25 tumor / 11 control) | Mismo escáner (Ghent) | 4 |

---

## 7. Resultados que debo tratar con cuidado

| Resultado / artefacto | Riesgo | Cómo interpretarlo en la memoria |
|---|---|---|
| **AUC ≈ 1.0 en pool multi-fuente** | Vender detección de tumor donde solo hay detección de dominio | Presentarlo **como anomalía a auditar**, nunca como logro. Acompañar SIEMPRE del tiny baseline (AUC 1.0) en la misma frase |
| **Métricas globales casi perfectas (sen 0.994, spe 1.0)** | Parecen excelentes | Son **síntoma del confound** (scores bimodales por dataset, varianza intra-dataset ≈ 0) |
| **BTC intra-dominio (n=36)** | Tentación de concluir "el modelo NO detecta tumor" | Solo permite decir: "**con n=36 no se observa señal por encima del azar**". IC95% ancho cruza 0.5 |
| **AUC 0.40 (< 0.5)** | Interpretarlo como "anti-detección" | Es **azar**: el IC95% [0.21,0.62] incluye 0.5 con holgura |
| **Resultados de datasets mono-clase** | AUC intra-dataset = NaN | Esa imposibilidad de medir AUC intra-dataset **es en sí la prueba** del confound, no un bug |
| **Grad-CAM** (`figures/anexo/gradcam/`) | Sobre-interpretar mapas de atención | **Exploratorio y no concluyente**; reportar con cautela, solo en anexo |
| **Notebooks** (`notebooks/*.ipynb`) | Cifras no versionadas/no reproducibles | NO citar como evidencia; solo inspección |
| **Checkpoints en `outputs/`** | Ignorados por git; se borran/cambian | Citar la **ruta y el timestamp** (`20260527_152619/best.pt`), no depender del binario |
| **`docs/00_entregas/TFG.pdf`** | Es la entrega **anterior** | No refleja la narrativa actual de auditoría |
| **`reconstruccion_evolucion_tfg.md`** | Posible solape/desfase | Verificar contra los JSON antes de usar |

---

## 8. Figuras principales y secundarias (`docs/audit/figures/`)

> **Aviso de reproducibilidad:** las figuras están organizadas en
> `figures/principales/` y `figures/anexo/`, pero los scripts que las generan
> (`embeddings_tsne.py`, `gradcam_3d.py`, `make_plots.py`, `make_extra_figures.py`)
> escriben en `docs/audit/figures/` (raíz, sin subcarpeta). Es decir, la
> organización principales/anexo se hizo **a mano después**. Si re-ejecutas los
> scripts, regenerarán en la raíz y habrá que volver a clasificarlas (o usar
> `replot_embeddings.py`, que sí pule para memoria).

### Figuras principales para memoria (`figures/principales/`)

| Figura | Qué muestra | Dónde meterla | Mensaje clave | Caption provisional |
|---|---|---|---|---|
| `principales/auc_summary.png` | Barras de AUC + IC95% por experimento | 7 Resultados (síntesis) | El AUC se desploma de ~1.0 a azar al eliminar el confound | *"AUC por régimen de evaluación con IC95%; el rendimiento colapsa al controlar el dominio."* |
| `principales/embeddings_tsne.png` | Proyección t-SNE del latente por dataset y por etiqueta | 7 Resultados (**figura estrella**) | IXI y NKI (ambos sanos) no se fusionan → el latente agrupa por procedencia | *"Espacio latente de la CNN: los clústeres se organizan por dataset de origen, no por clase clínica."* |
| `principales/confusion_matrices.png` | Matrices 2×2 por régimen | 7 Resultados | Diagonal perfecta → colapso/inversión (LODO) → dispersión (intra-dominio) | *"Matrices de confusión: del acierto trivial confundido al azar intra-dominio."* |
| `principales/intensity_by_dataset.png` | Boxplots de estadísticos de intensidad por dataset | 7 Resultados / 4 Materiales | El confound existe ya en los píxeles crudos | *"Distribuciones de intensidad separables por dataset: el confound precede al modelo."* |

### Figuras secundarias o anexos (`figures/anexo/`)

| Figura | Qué muestra | Dónde | Comentario |
|---|---|---|---|
| `anexo/roc_curves.png` | ROC superpuestas (confound, LODO A/B, Ghent) | Resultados o anexo | Apoyo de `auc_summary` |
| `anexo/score_hist_confound.png` | Histograma de scores por dataset (run confundido) | Anexo | Scores bimodales por dataset |
| `anexo/score_hist_lodo.png` | Histograma de scores en LODO A/B | Anexo | Descolocación cross-dataset |
| `anexo/btc_kfold_bars.png` | AUC por fold (intra-dominio) | Anexo | IC95% ancho, dispersión |
| `anexo/embeddings_pca.png` | Versión PCA del t-SNE (varianza explicada) | Anexo | Complementa la figura estrella |
| `anexo/gradcam/{confound,lodo_A,lodo_B}/*.png` | Mapas Grad-CAM por sujeto | **Solo anexo** | **No concluyente** — declarar exploratorio |

---

## 9. Flujo reproducible del proyecto

Cadena lógica (los comandos exactos solo se indican donde aparecen literalmente
en scripts/docs; el resto se describe sin inventar invocación).

1. **Preparar datos** — `data/raw/*`. Descarga NKI: `src/preprocessing/download_nki_rockland.py`. DICOM→NIfTI: `src/preprocessing/conversion/dicom_to_nifti.py {brats|upenn}`.
2. **Preprocesar** — `preprocess_{brats,ixi,upenn,nki_rockland}.py` → `data/processed/`; `preprocess_btc.py` → `data/processed_btc/`. (Resumen: `dataset/summarize_processed.py`.)
3. **Crear splits** — `create_splits` en `src/data/dataset_3d.py` (vía `train_3d.py` con `recreate_splits:true`) → `data/splits.json`. Auditar: `python -m src.data.audit_splits`.
4. **Entrenar CNN** — `python -m src.training.train_3d` (config `configs/train_3d.yaml`) → checkpoint en `outputs/checkpoints/<timestamp>/best.pt`.
5. **Evaluar** — `python -m src.evaluation.evaluate_3d --checkpoint <run>/best.pt` → `outputs/evaluation/cnn3d_test_results.json`.
6. **Auditar leakage** — `python -m src.audit.audit_leakage` → `docs/audit/audit_leakage.json`, `audit_features.csv`.
7. **Ejecutar LODO** — `src/audit/make_lodo_splits.py` (configs A/B/C/D) → `scripts/run_lodo.sh` (entrena+evalúa con `train_lodo.yaml`); tiny baseline LODO: `python -m src.audit.audit_lodo` → `audit_lodo.json`.
8. **BTC intra-dominio** — `scripts/run_btc_chain.sh`: `preprocess_btc.py` → `btc_tiny_baseline.py` → `btc_cnn_kfold.py` → `outputs/evaluation/btc_intradomain/cnn_kfold_results.json` + `docs/audit/btc_intradomain_tinybaseline.json`.
9. **Analizar embeddings** — `python -m src.audit.embeddings_tsne` → `embeddings.npz` + `embeddings_silhouette.json`; luego `embeddings_intraclass.py` → `embeddings_intraclass.json`. Pulido: `replot_embeddings.py`.
10. **Generar figuras** — `make_plots.py`, `make_extra_figures.py`, `gradcam_3d.py` → `docs/audit/figures/`.
11. **Consolidar resultados** — `python -m src.audit.consolidate_results` → `resumen_consolidado.{json,md}`. Tablas PDF: `scripts/gen_tablas_pdf.py`.
12. **Redactar memoria** — `memoria_tfg/` (LaTeX UAX), usando este documento y `borrador_memoria.md` como guía.

> `seed = 42` (split y entrenamiento), `seed = 0` (bootstrap IC95% y proyecciones).
> Entorno: ver `requirements.txt` (PyTorch 2.5.1 **ROCm** → GPU AMD; MONAI 1.5; HD-BET 2.0.1).

---

## 10. Relación entre repo y capítulos de la memoria

| Capítulo memoria (`memoria_tfg/secciones/`) | Archivos del repo a consultar | Figuras/tablas | Resultados | Comentarios |
|---|---|---|---|---|
| 1 Introducción (`01_introduccion.tex`) | `resumen_una_pagina.md`, `borrador_memoria.md §5` | — | Motivación clínica | Plantear el giro a auditoría |
| 2 Objetivos (`02_objetivos.tex`) | `resumen_una_pagina.md`, `justificacion_cambios.md` | — | — | Objetivo clínico + objetivo metodológico |
| 3 Estado del arte (`03_estado_arte.tex`) | `docs/03_estudio/referencias.md`, `plan_trabajo_tfg.md` | — | — | Poblar `bibliography.bib` |
| 4 Materiales y datos (`04_materiales_datos.tex`) | `src/preprocessing/*`, `auditoria_...md §2` | `intensity_by_dataset.png` | Composición (1167/1100), BTC 25/11 | Describir pipeline una vez |
| 5 Metodología (`05_metodologia.tex`) | `cnn3d.py`, `train_3d.py`, `dataset_3d.py`, `configs/train_3d.yaml` | — | — | Modelo + entrenamiento + split |
| 6 Diseño experimental (`06_diseno_experimental.tex`) | `make_lodo_splits.py`, `configs/train_lodo.yaml`, `audit_splits.py`, `threshold_analysis.py` | — | Esquema confound/LODO/intra-dominio | Justificar comparación apples-to-apples |
| 7 Resultados (`07_resultados.tex`) | `resumen_consolidado.md`, todos los JSON de `docs/audit/` | **4 figuras principales** | Tabla maestra (§6) | Núcleo cuantitativo |
| 8 Discusión (`08_discusion.tex`) | `auditoria_...md §§4,11`, `borrador_memoria.md §§2-3` | `embeddings_tsne.png` | Confound, embeddings, hipótesis descartadas | Frases listas en `borrador §5` |
| 9 Conclusiones (`09_conclusiones.tex`) | `resumen_una_pagina.md` (conclusión/contribución) | — | — | Protocolo de auditoría como contribución |
| 10 Limitaciones y futuro (`10_limitaciones_futuro.tex`) | `borrador_memoria.md §§4,6,7` | — | n=36, T1-only, sin validación externa | Edinburgh SN-851861 como trabajo futuro |
| 11 Anexos (`anexos.tex`) | `gradcam_3d.py`, `gen_tablas_pdf.py`, `reproducir` (auditoría §final) | figuras `anexo/`, `tablas_resultados.pdf` | per-fold, Grad-CAM | Material de apoyo |

---

## 11. Archivos legacy, auxiliares o de prueba

> **Nada se borra ni se mueve.** Solo se clasifica.

| Archivo / carpeta | Por qué parece auxiliar/legacy | Riesgo de usarlo mal | Recomendación |
|---|---|---|---|
| `src/preprocessing/preprocess_volumes.py` | Lanzador antiguo que solo cubre BraTS+IXI (pre-UPENN/NKI/BTC) | Creer que es el pipeline actual | **Revisar** / ignorar como referencia |
| `notebooks/explorar_processed_dinamico.ipynb` | Notebook exploratorio | Citar cifras no reproducibles | **Ignorar** como evidencia |
| `notebooks/test_modelo_3d.ipynb` | Notebook de prueba del modelo | Ídem | **Ignorar** como evidencia |
| `docs/00_entregas/TFG.pdf` | Entrega anterior | Confundir con la memoria actual | **Conservar** como histórico |
| `docs/01_memoria/reconstruccion_evolucion_tfg.md` | Muy largo, posible solape con otros 2 docs de evolución | Citar narrativa desfasada | **Revisar** antes de usar |
| `docs/03_estudio/*.html`, `guia_fomo_amaes.*`, `plan_estudio.html` | Material de estudio externo | Tomarlo por contenido de la memoria | **Ignorar** (no es del TFG) |
| `docs/audit/figures/*` generadas en raíz por scripts vs `principales/anexo/` | Doble ubicación posible tras re-ejecutar scripts | Enlazar la figura equivocada en LaTeX | **Revisar** rutas al insertar |
| `README.md` | Solo contiene el título (1 línea) | — | **Conservar** / completar opcionalmente |
| `outputs/` (checkpoints, plots) | Binarios pesados, ignorados por git | Depender de un binario que cambia | **Conservar** local; citar ruta+timestamp |
| `src/**/__pycache__/`, `memoria_tfg/_minted/` | Artefactos de compilación | — | **Ignorar** (regenerables) |
| `memoria_tfg/_plantilla_original/` | Restos de la plantilla UAX (movidos en sesión previa) | Editar el archivo equivocado | **Conservar** como referencia, fuera de `secciones/` |

---

## 12. Recomendaciones de orden para redactar

Orden sugerido (de lo más sólido/cerrado a lo más interpretativo):

1. **Materiales y datos (4)** — el preprocesado y la composición están cerrados y documentados. Arranque fácil.
2. **Metodología (5)** — modelo y entrenamiento ya fijos en código/config; describir sin ambigüedad.
3. **Diseño experimental (6)** — encadena confound → LODO → intra-dominio → embeddings; define el armazón de Resultados.
4. **Resultados (7)** — con las cifras de `resumen_consolidado` y las 4 figuras principales; es mecánico una vez fijado el diseño.
5. **Discusión (8)** — interpretar el confound; reutilizar frases de `borrador_memoria.md §5`.
6. **Introducción (1)** — se redacta mejor **después** de saber qué cuentan Resultados/Discusión.
7. **Estado del arte (3)** — requiere poblar `bibliography.bib`; hacerlo en paralelo a Introducción.
8. **Conclusiones (9)** y **Limitaciones/futuro (10)** — destilan lo anterior.
9. **Resumen/abstract** — lo último (ya hay versión en `main.tex` `\resumen{}` y en `resumen_una_pagina.md`).
10. **Anexos (11)** — al final: Grad-CAM, per-fold, reproducibilidad.

**Razón:** los capítulos descriptivos (4-6) están "congelados" por el código y se
escriben sin decisiones abiertas; Resultados se apoya en JSON ya consolidados; y
los capítulos interpretativos/marco (1, 3, 9) se benefician de tener el cuerpo ya
escrito para no contradecirse.

---

## 13. Glosario interno del proyecto

- **Pool multi-fuente** — conjunto combinado BraTS + UPENN-GBM (tumor) + IXI + NKI Rockland (sano); base del run confundido.
- **Confound dominio-clase** — la etiqueta coincide al 100 % con el dataset de origen; "tumor" y "escáner de origen" son indistinguibles.
- **Shortcut learning** — el modelo aprende un atajo (firma de adquisición/preprocesado) en lugar de la señal clínica.
- **LODO** (*leave-one-dataset-out*) — entrenar con unos datasets y testear en otros nunca vistos; mide generalización cross-dataset (configs A/B/C/D).
- **BTC_preop** — dataset OpenNeuro **ds001226** (Aerts et al., CC0), 36 sujetos del **mismo escáner** (Ghent University Hospital), 25 tumor / 11 control; T1-only. *Sinónimos en el repo:* **"Ghent"**, **"ds001226"**, **"intra-dominio"** → **es la misma cohorte** (atención a la triple nomenclatura).
- **Tiny baseline** — modelo lineal (LogReg) o RF sobre 16 estadísticos triviales de intensidad (sin red, sin anatomía); cota trivial.
- **Embeddings** — vector de 96 dimensiones de la última capa convolucional (tras `AdaptiveAvgPool3d`), antes del clasificador.
- **Grad-CAM** — mapa de atención 3D; aquí **exploratorio y no concluyente**.
- **Split por sujeto** — partición que garantiza que ningún `subject_id` aparece en dos splits (sin leakage de partición).
- **Leakage** — fuga de información que infla métricas; aquí es **leakage de etiqueta vía confound de dominio** (no por partición ni duplicados, ambos descartados).
- **AUC** — área bajo la curva ROC; calculada sobre probabilidades sigmoides.
- **Sensibilidad** — proporción de tumores correctamente detectados (métrica clínica principal del triaje).
- **Especificidad** — proporción de sanos correctamente clasificados.
- **IC95%** — intervalo de confianza al 95 % por bootstrap (2000 resamples); si cruza 0.5 en AUC → indistinguible del azar.
- **Silhouette** — compacidad de clústeres; aquí, por etiqueta (0.754) vs por dataset (0.366).

---

## 14. Checklist para no perderme

**Antes de redactar cada capítulo, mirar:**
- [ ] Cap. 4 → `src/preprocessing/*` + `auditoria_resultados_sospechosos.md §2` (composición).
- [ ] Cap. 5 → `cnn3d.py`, `train_3d.py`, `dataset_3d.py`, `configs/train_3d.yaml`.
- [ ] Cap. 6 → `make_lodo_splits.py`, `configs/train_lodo.yaml`, `audit_splits.py`.
- [ ] Cap. 7 → `docs/audit/resumen_consolidado.{md,json}` + JSON individuales + `figures/principales/`.
- [ ] Cap. 8 → `borrador_memoria.md §§2-5`, `auditoria_...md §§4,11`.
- [ ] Cap. 10 → `borrador_memoria.md §§4,6,7`.

**Cifras oficiales (no usar otras):** confound CNN AUC **0.9999** (sen 0.994 / spe 1.000) · tiny baseline mezcla **1.0000** · clf dominio **0.985** · LODO A **0.6236** · LODO B **0.2012** · BTC CNN **0.4036 [0.21,0.62]** · BTC tiny LogReg **0.5491 [0.32,0.79]** · embeddings IXI-vs-NKI **0.998** · clf dataset desde embeddings **0.982**. Fuente: `docs/audit/*.json`.

**Figuras principales (las 4):** `auc_summary.png`, `embeddings_tsne.png`, `confusion_matrices.png`, `intensity_by_dataset.png` (en `figures/principales/`).

**Lo que NO debo afirmar:**
- [ ] Que la CNN "detecta tumores" (solo detecta dominio).
- [ ] Que AUC ≈ 1.0 es un logro.
- [ ] Que el modelo "NO detecta tumor" (solo: "con n=36 no hay señal sobre el azar").
- [ ] Que AUC 0.40 es "anti-detección" (es azar; IC cruza 0.5).
- [ ] Que Grad-CAM demuestra algo (es exploratorio).

**Lo que debo explicar con cuidado:**
- [ ] El confound como **eje central**, no como nota al pie.
- [ ] Que el split es correcto pero **no salva** el experimento (el problema está en la composición).
- [ ] Silhouette global (0.366) **y** análisis intra-clase (0.998) juntos, por honestidad.

**Pendiente / experimental:**
- [x] `bibliography.bib` con **48 entradas** (sin duplicados). Añadidas y **verificadas por web (2026-06-06)** las 11 🔴 que faltaban (datasets BraTS/TCIA y herramientas: nnU-Net, dcm2niix, PyTorch, AdamW, GroupNorm, Grad-CAM, t-SNE, scikit-learn). Claves reales estilo `apellidoAÑOpalabra` (p. ej. `menze2015brats`), NO las de `referencias.md`. **Pendiente menor:** campos `% TODO` en 5 entradas (`aerts2022btcpreop`, `contreras2017epidemiologia`, `who2007cns`, `martucci2023mri`, `lambert2023acceptance`).
- [ ] **Cuarentena:** `outputs/evaluation/cnn3d_test_notebook_summary.json` es un run de **notebook** (n y AUC distintos) que NO corresponde al run confundido oficial (AUC 0.9999) — **no citar**.

**Cifras del pool RECUPERADAS Y VERIFICADAS (2026-06-06)** — se localizaron los JSON originales en el equipo de la UAX y se versionaron en `docs/audit/`:
- `docs/audit/cnn3d_test_results.json` → pool confundido: **AUC 0.99996, sen 0.9943, spe 1.000, PR-AUC 0.99997, n=340 (175 pos / 165 neg)**. Checkpoint `20260527_152619`.
- `docs/audit/cnn3d_test_predictions.csv` → 340 predicciones por muestra (verificado: 175/165; brats 87 / ixi 86 / nki 79 / upenn 88).
- `docs/audit/lodo_A_cnn3d_test_results.json` (AUC 0.6236, sen 0.676, spe 0.463) y `docs/audit/lodo_B_cnn3d_test_results.json` (AUC 0.2012, sen 0.010, spe 0.965).
- `docs/audit/btc_intradomain_cnn_kfold_results.json` → CNN intra-dominio AUC 0.4036 IC95% [0.213, 0.623] + detalle por fold (scores ≈0.50 constantes).
- `docs/audit/preprocessing_summary.json` → composición (BraTS 580 / UPENN 587 / IXI 577 / NKI 523; 1167 pos / 1100 neg; total 2267).
- Splits originales (pool n=340, LODO A/B) confirmados en el equipo UAX; no versionados por tamaño, pero `cnn3d_test_results.json` ya confirma n y composición del test.
- [ ] Redactar todas las secciones `memoria_tfg/secciones/*.tex` (hoy en `% TODO`).
- [ ] Validación externa real (Edinburgh SN-851861) → trabajo futuro.
- [ ] Revisar si `reconstruccion_evolucion_tfg.md` aporta algo no cubierto.
- [ ] Reorganizar/verificar rutas de figuras (`principales/anexo/` vs raíz) al insertarlas en LaTeX.

---

*Documento de orientación. No sustituye a los JSON de `docs/audit/` como fuente de
cifras ni al código como fuente de la lógica. Generado en auditoría de solo lectura.*
