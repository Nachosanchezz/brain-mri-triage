# CONTEXTO COMPACTO — TFG `brain-mri-triage` (UAX, Ing. Matemática)

> **Para qué sirve.** Único archivo de contexto a cargar en los chats de
> redacción. Cifras **verificadas contra JSON versionados de `docs/audit/`**.
> Generado en auditoría de solo lectura el 2026-06-06. No redacta memoria.
>
> **Regla de oro:** una cifra solo es *verificada* si aparece en un JSON de
> `docs/audit/`. Lo que solo está en `.md`, en `data/` o en `outputs/`
> (ignorados por git) va como **NO VERIFICADA**. No inventar; lo que falta es
> **PENDIENTE**.

---

## 0. Tesis en una frase

CNN 3D tumor/no-tumor sobre pool público multi-fuente da **AUC ≈ 1.0**, pero ese
rendimiento es **íntegramente un confound dominio↔clase** (`label=1 ⟺ {BraTS,
UPENN}`; `label=0 ⟺ {IXI, NKI}`): "detectar tumor" ≡ "reconocer el escáner".
Al controlar el dominio (intra-dominio BTC_preop, mismo escáner) el modelo rinde
**al azar**. Contribución = caracterizar el aprendizaje espurio y proponer un
**protocolo de auditoría anti-confounding** (tiny baseline → LODO → intra-dominio
→ embeddings → descarte de leakage). NO es desplegar un clasificador.

---

## 1. Cifras oficiales

### 1A. VERIFICADAS (fuente: JSON de `docs/audit/`)

| Métrica | Valor | Archivo · clave | Estado |
|---|---|---|---|
| CNN pool multi-fuente, AUC test | **0.9999** (0.999965) | `cnn3d_test_results.json` · `auc` (y `resumen_consolidado.json`) | verificada |
| CNN pool, **sen / spe** | **0.9943** / **1.000** | `cnn3d_test_results.json` · `metrics_at_threshold.sensitivity/specificity` | verificada |
| CNN pool, PR-AUC / acc | 0.99997 / 0.99706 | `cnn3d_test_results.json` · `pr_auc` / `metrics_at_threshold.accuracy` | verificada |
| CNN pool, matriz @0.5 (TP/FP/TN/FN) | 174 / 0 / 165 / 1 (FN único `BraTS2021_00736`) | `cnn3d_test_results.json` · `metrics_at_threshold` + `cnn3d_test_predictions.csv` | verificada |
| CNN pool, test n / composición | **340** (175 pos / 165 neg; brats 87 / upenn 88 / ixi 86 / nki 79) | `cnn3d_test_results.json` · `n_samples` + `per_dataset.*.n` | verificada |
| Scores por dataset (media±std) | upenn 0.9995±0.0005 · brats 0.986±0.098 · ixi 0.0070±0.018 · nki 0.0008±0.0005 | `cnn3d_test_results.json` · `per_dataset.*.score_mean/score_std` | verificada |
| Composición pool por dataset | BraTS **580** / UPENN **587** / IXI **577** / NKI **523** | `preprocessing_summary.json` · `datasets.*.total` | verificada |
| Tiny baseline mezcla LogReg, AUC / acc | **1.0000** / 0.9941 | `audit_leakage.json` · `tiny_baseline_label.logreg_test_auc` / `.logreg_test_acc` | verificada |
| Tiny baseline mezcla RF, AUC / acc | 0.9989 / 0.9882 | `audit_leakage.json` · `tiny_baseline_label.rf_test_auc` / `.rf_test_acc` | verificada |
| Clf de origen (intensidad, 4 clases), acc | **0.9853** (azar 0.25) | `audit_leakage.json` · `dataset_origin_classifier.rf_dataset_test_acc` | verificada |
| Duplicados exactos cross-split | **0 grupos** | `audit_leakage.json` · `exact_duplicates_cross_split` | verificada |
| label ≡ dataset | brats:[1] upenn:[1] ixi:[0] nki:[0] | `audit_leakage.json` · `label_equals_dataset` | verificada |
| LODO A (BraTS+IXI→UPENN+NKI) CNN, AUC/sen/spe | **0.6236** / 0.6763 / 0.4627 | `lodo_A_cnn3d_test_results.json` (y `resumen_consolidado.json`); n=1110 (587+523) | verificada |
| LODO B (UPENN+NKI→BraTS+IXI) CNN, AUC/sen/spe | **0.2012** / 0.0103 / 0.9653 | `lodo_B_cnn3d_test_results.json` (y `resumen_consolidado.json`); n=1157 (580+577) | verificada |
| LODO tiny LogReg (A/B/C/D) | 0.9952 / 0.3184 / 0.0412 / 0.0469 | `audit_lodo.json` (4 claves) | verificada |
| LODO tiny RF (A/B/C/D) | 0.9889 / 0.6957 / 0.0338 / 0.2752 | `audit_lodo.json` (4 claves) | verificada |
| BTC intra-dominio CNN 1-canal, AUC / IC95% | **0.4036** / [0.213, 0.623] | `btc_intradomain_cnn_kfold_results.json` · `overall_auc` / `overall_auc_ci95` (y `resumen_consolidado.json`) | verificada |
| BTC CNN, sen / spe | 0.60 / 0.3636 | `btc_intradomain_cnn_kfold_results.json` · `metrics_at_thr_0.5` (y `resumen_consolidado.json`) | verificada |
| BTC CNN, AUC por fold | 0.4 / 0.3 / 0.2 / 0.6 / 0.2 (scores ≈0.50 constantes) | `btc_intradomain_cnn_kfold_results.json` · `per_fold[].auc/scores` | verificada |
| BTC tiny LogReg, AUC / IC95% | 0.5491 / [0.319, 0.788] | `btc_intradomain_tinybaseline.json` · `logreg_auc_overall` / `logreg_auc_ci95` | verificada |
| BTC tiny RF, AUC / IC95% | 0.4055 / [0.215, 0.616] | `btc_intradomain_tinybaseline.json` · `rf_auc_overall` / `rf_auc_ci95` | verificada |
| BTC composición / k-fold | **36** (25 tumor / 11 control), k=5 | `btc_intradomain_tinybaseline.json` · `n_samples/n_positives/n_negatives/k_fold` | verificada |
| Embeddings silhouette label / dataset | **0.754** / 0.366 | `embeddings_silhouette.json` · `silhouette_by_label` / `silhouette_by_dataset` | verificada |
| Embeddings intra-clase IXI vs NKI (sanos) CV-AUC (sil) | **0.998** (0.561) | `embeddings_intraclass.json` · `intraclass_sanos_IXI_vs_NKI.logreg_cv_auc` | verificada |
| Embeddings intra-clase BraTS vs UPENN (tumor) CV-AUC (sil) | 0.991 (0.186) | `embeddings_intraclass.json` · `intraclass_tumor_BraTS_vs_UPENN.logreg_cv_auc` | verificada |
| Clf dataset **desde embeddings** (4 clases), acc | **0.982** (azar 0.25) | `embeddings_intraclass.json` · `dataset_classifier_from_embeddings.cv_accuracy` | verificada |
| Composición pool (totales) | **1167 tumor / 1100 sanos** (=2267) | `embeddings_intraclass.json` · `intraclass_tumor_*.n=1167`, `intraclass_sanos_*.n=1100` | verificada (derivada) |

> Nota técnica: `audit_leakage.json` reporta `n_train=600 / n_test=340` — es la
> **submuestra balanceada del tiny baseline**, NO el split del pool. No confundir
> con el split de la CNN (ver §1B). El tiny baseline del pool usa **16** stats
> (T1+T2); el de BTC usa **8** (T1-only, `feature_names` del JSON).

> **Actualización 2026-06-06:** las cifras del pool (sen/spe/PR-AUC/acc, matriz,
> composición por dataset, scores por dataset) **se recuperaron del equipo de la
> UAX y se versionaron** en `docs/audit/` → ahora están en §1A como **verificadas**.
> Archivos nuevos: `cnn3d_test_results.json`, `cnn3d_test_predictions.csv`,
> `lodo_A_cnn3d_test_results.json`, `lodo_B_cnn3d_test_results.json`,
> `btc_intradomain_cnn_kfold_results.json`, `preprocessing_summary.json`.

### 1B. NO VERIFICADAS (lo que aún NO tiene JSON versionado)

| Métrica | Valor citado | Fuente declarada | Por qué NO verificada |
|---|---|---|---|
| Tamaños de split train/val | train 1587 (817/770) / val 340 (175/165) | `splits_audit_pool.json` (derivado) | **derivado** y consistente (1587+340+340=2267); test SÍ verificado (340: 175/165). Train/val pos-neg confirmables con `src.data.audit_splits` |
| 0 solapamiento de `subject_id` train/val/test | 0 overlap | `splits_audit_pool.json` · `subject_overlap` | test confirmado (340 subject_id únicos en CSV); train/val por construcción. Prueba formal: `python -m src.data.audit_splits` |
| Parámetros del modelo | 504 553 (2 ch) / 504 229 (1 ch) | `src/models/cnn3d.py` | dato de código, no JSON (verificable leyendo el código) |
| BTC = mismo escáner Ghent, T1-only, CC0 | metadato cualitativo | `data/raw/btc_preop/participants.tsv` | `data/` ignorado; es metadato, no cifra medida |

---

## 2. Afirmaciones permitidas vs prohibidas

**Puedo afirmar:**
- La CNN obtiene AUC ≈ 1.0 en el pool, **presentado como anomalía a auditar**.
- Un tiny baseline lineal sobre intensidad alcanza **AUC 1.0** sin red ni anatomía → confound probado.
- El origen del dataset es trivialmente identificable (intensidad 0.985; embeddings 0.982).
- En LODO el rendimiento cae (0.62) e **invierte** (0.20, AUC<0.5).
- Intra-dominio (n=36): "**con n=36 no se observa señal por encima del azar**".
- El latente codifica **procedencia**: IXI vs NKI (ambos sanos) AUC 0.998.
- Confound, leakage por partición (0 overlap) y duplicados (0) son hipótesis tratadas; el confound es la confirmada.

**NO puedo afirmar:**
- Que la CNN "detecta tumores" (solo detecta dominio).
- Que AUC ≈ 1.0 sea un logro.
- Que el modelo "NO detecta tumor" (solo: "con n=36 no hay señal sobre el azar").
- Que AUC 0.40 sea "anti-detección" (es azar; IC95% [0.21,0.62] cruza 0.5).
- Que Grad-CAM demuestre algo (exploratorio / no concluyente, solo anexo).
- Citar como verificada cualquier cifra de §1B sin antes versionar su JSON.

---

## 3. Terminología canónica (una forma por concepto)

| Forma canónica | Sinónimos en el repo (no mezclar en el texto) |
|---|---|
| **BTC_preop** (OpenNeuro **ds001226**, Aerts et al., CC0) | Ghent · ds001226 · intra-dominio · "experimento honesto" · BTC — **misma cohorte** |
| **Pool multi-fuente** | mezcla / mixed / pool de 4 datasets (BraTS+UPENN tumor; IXI+NKI sano) |
| **Confound dominio↔clase** | sesgo de dominio / domain bias (usar "confound" como término eje) |
| **Baseline trivial de intensidad** | tiny baseline · baseline trivial · línea base de intensidad (en prosa usar **"baseline trivial de intensidad"**; "tiny baseline" es jerga del repo, evitar en el texto de la memoria) |
| **LODO** | leave-one-dataset-out / validación cross-dataset |
| **Régimen de evaluación** | confundido · cross-dataset (LODO) · intra-dominio |
| **Embeddings** | vector latente de **96-dim** (tras `AdaptiveAvgPool3d`) |
| **Aprendizaje espurio** | shortcut learning / atajo |
| **NKI Rockland** | NKI / nki_rockland |

---

## 4. Referencias por capítulo (clave de `bibliography.bib` · prioridad · estado)

> Prioridad de `referencias.md`: 🔴 obligatoria · 🟡 marco · 🟢 contexto.
> **Estado bibliográfico: TODAS sin verificar año/DOI** (aviso de `referencias.md`).
> Marcadas **[.bib]** = ya en `bibliography.bib`; **[falta]** = en `referencias.md`
> pero aún NO en el `.bib`. **** = entrada en `.bib` con `% TODO: verificar`.

**Cap. 3.5 / 8 — Confound y shortcut learning (🟡 núcleo discusión)**
- `geirhos2020shortcut` 🟡 [.bib] — la más importante.
- `zech2018pneumonia` 🟡 [.bib] · `degrave2021shortcuts` 🟡 [.bib]
- `glocker2019scanner` 🟡 [.bib] · `castro2020causality` 🟡 [.bib] (no estaban en `referencias.md`)
- revisión generalización ("wynants") 🟡 [falta]

**Cap. 4.1 — Datasets (🔴 obligatorias)**
- `baid2021brats` 🔴 [.bib] · `bakas2022upenngbm` 🔴 [.bib] · `nooner2012nki` 🔴 [.bib]
- `ixi_dataset` 🔴 [.bib] · `aerts2022btcpreop` 🔴 [.bib ✅] (autores corregidos: Aerts, Colenbier, Almgren, Marinazzo; DOI dataset + descriptor Sci Data 9:676)
- `menze2015brats` 🔴 [.bib ✅verif-web] · `bakas2017advancing` 🔴 [.bib ✅] · `clark2013tcia` 🔴 [.bib ✅]

**Cap. 5.1–5.6 — Herramientas/métodos (🔴 obligatorias) — [.bib ✅ verificadas por web 2026-06-06]**
- `isensee2019hdbet` 🔴 [.bib] · `isensee2021nnunet` 🔴 [.bib ✅]
- `li2016dcm2niix` · `paszke2019pytorch` · `loshchilov2019adamw` · `wu2018groupnorm` · `pedregosa2011sklearn` — todas 🔴 [.bib ✅]
- `selvaraju2017gradcam` 🔴 [.bib ✅] · `maaten2008tsne` 🔴 [.bib ✅] (cap. 6.7/7.6)

> **Nota de claves:** las claves reales del `.bib` son `apellidoAÑOpalabra` (p. ej. `menze2015brats`), NO las de `referencias.md` (`brats_menze2015`). DOI/año/páginas verificados por web. AdamW y GroupNorm sin DOI limpio → citadas por arXiv (eprint) + venue (ICLR/ECCV).

**Cap. 10.2 — Armonización / adaptación de dominio (🟡 futuro)**
- `johnson2007combat` (ComBat orig., Biostatistics 8(1):118-127) · `fortin2017harmonization` (ComBat neuroimagen, NeuroImage 161:149-170) · `dann_ganin2016`→clave real `ganin2016dann` (DANN, JMLR 17(59):1-35) — 🟡 [.bib ✅ verif-web 2026-06-07]

**Cap. 3.3 / 3.6 / 8 — IA en radiología y triaje (🟢 contexto, en `.bib`)**
- `litjens2017survey` 🟢 [.bib] · `titano2018cranial` 🟢 [.bib] (triaje neuro)
- `avanzo2024evolution`, `driver2019radiology`, `langs2023ai`, `wagner2021radiomics`, `yeasmin2024advances`, `zech2022fracture`, `lee2022implementation`, `sharma2020diagnostic`, `hickman2021breast`, `lambert2023acceptance` — 🟢 [.bib]
- `esteva` · `triage_oref/worklist` 🟢 [falta]

**Cap. 3.1–3.4 — Clínica / WHO / RM / segmentación (🟢, en `.bib`)**
- `louis2021who` · `kleihues2000who` · `who2007cns` · `ostrom2023cbtrus` · `mcneill2016epidemiology` · `wen2008gliomas` · `contreras2017epidemiologia` · `perezsegura2025tumores` · `cha2009neuroimaging` · `villanuevameyer2017imaging` · `martucci2023mri`
- `kamnitsas2017deepmedic` · `labella2023brats` · `laukamp2021meningioma`

---

## 5. Mapa figura → mensaje (solo `figures/principales|anexo/`)

**Principales (`figures/principales/`) — al cuerpo:**
| Figura | Cap. | Mensaje |
|---|---|---|
| `auc_summary.png` | 7 | AUC se desploma de ~1.0 a azar al quitar el confound (con IC95%). |
| `embeddings_tsne.png` | 7 | **Figura estrella**: el latente agrupa por procedencia; IXI≠NKI pese a ser ambos sanos. |
| `confusion_matrices.png` | 7 | Diagonal perfecta (confundido) → colapso/inversión (LODO) → dispersión (intra-dominio). |
| `intensity_by_dataset.png` | 4/7 | El confound ya existe en los píxeles crudos. |

**Anexo (`figures/anexo/`) — apoyo:**
| Figura | Mensaje |
|---|---|
| `roc_curves.png` | ROC por régimen (confound pega al techo, intra-dominio en diagonal). |
| `score_hist_confound.png` | Scores bimodales por dataset, no por contenido. |
| `score_hist_lodo.png` | Descolocación de scores en LODO A/B. |
| `btc_kfold_bars.png` | AUC por fold disperso en torno al azar, IC95% ancho. |
| `embeddings_pca.png` | Versión PCA del t-SNE (varianza explicada). |
| `gradcam/{confound,lodo_A,lodo_B}/*.png` | **Exploratorio / no concluyente** (bordes/periferia). Solo anexo. |

---

## 6. Discrepancias detectadas

1. **[RESUELTO 2026-06-06] Cifras del pool sin JSON versionado.** `indice.md` y
   `material_redaccion_y_tutor.md` marcaban `[V]` la composición por dataset
   (580/587/577/523), sen/spe/PR-AUC del pool y el split, con fuentes en `data/`
   y `outputs/` (ignorados por git). → Se recuperaron los JSON del equipo UAX y
   se versionaron en `docs/audit/` (`cnn3d_test_results.json`,
   `preprocessing_summary.json`, predictions.csv, LODO A/B, BTC k-fold). **Ya
   están en §1A como verificadas.** Solo quedan sin versionar los tamaños de
   split train/val y el 0-overlap (ver §1B).

2. **Dos clasificadores de dataset distintos, fácil de confundir.** acc **0.985**
   = RF sobre **intensidad** (`audit_leakage.json`); acc **0.982** = sobre
   **embeddings** (`embeddings_intraclass.json`). → Mantener separados; no citar
   una cifra con la fuente de la otra.

3. **Estado del `.bib` desactualizado en la guía.** `GUIA_REPOSITORIO_TFG.md`
   (§14) dice "4 entradas placeholder"; el `.bib` real tiene ~31 entradas.
   → **Elegido: el `.bib` es el estado actual**; la guía está desfasada en eso.

4. **Claves de cita divergentes.** `referencias.md` propone `[brats_baid2021]`,
   `[shortcut_geirhos2020]`…; el `.bib` usa `baid2021brats`, `geirhos2020shortcut`…
   → **Elegido: las claves del `.bib`** (son las reales de `\cite`).

5. **Rutas de figura dobles.** `resumen_consolidado.md` apunta a
   `figures/embeddings_tsne.png` (raíz); el archivo vive en
   `figures/principales/embeddings_tsne.png`. (Guía §8: los scripts escriben en
   raíz; la organización principales/anexo se hizo a mano.)
   → **Elegido: rutas `principales/anexo/`** al insertar en LaTeX.

6. **`n` del tiny baseline ≠ split del pool.** `audit_leakage.json` da
   `n_train=600/n_test=340` (submuestra balanceada), no el split real
   (1587/340/340, no verificado). El `340` de test coincide por casualidad.

---

## 7. Huecos PENDIENTE (antes de redactar)

- **[HECHO]** Versionados los JSON del pool/LODO/BTC en `docs/audit/` → sen/spe/
  PR-AUC/composición del pool ya son citables. Auditoría de partición en
  `splits_audit_pool.json` (test verificado; train/val derivados). **Queda solo**
  (opcional): re-ejecutar `python -m src.data.audit_splits` para confirmar el
  pos/neg exacto de train/val y los tres solapamientos = 0 de forma formal.
- **[HECHO 2026-06-06]** Pobladas las 11 🔴 que faltaban (verificadas por web):
  `menze2015brats`, `bakas2017advancing`, `clark2013tcia`, `isensee2021nnunet`,
  `li2016dcm2niix`, `paszke2019pytorch`, `loshchilov2019adamw`, `wu2018groupnorm`,
  `selvaraju2017gradcam`, `maaten2008tsne`, `pedregosa2011sklearn`. El `.bib`
  tiene ahora 48 entradas, sin claves duplicadas, llaves balanceadas.
- **[HECHO 2026-06-06]** Verificados por web los 5 campos `% TODO`:
  `aerts2022btcpreop` (autores corregidos + descriptor Sci Data 9:676),
  `contreras2017epidemiologia` (28(3):332--338), `who2007cns` (4ª ed., editores
  Louis/Ohgaki/Wiestler/Cavenee), `martucci2023mri` (art. 628, DOI-consistente),
  `lambert2023acceptance` (npj Digit Med 6:111). **El `.bib` ya no tiene `% TODO`.**
- **Redactar las secciones `memoria_tfg/secciones/*.tex`** (hoy en TODO).
- **Validación externa real** (Edinburgh, UK Data Service SN-851861): trabajo
  futuro; UPENN-GBM se pasó a train, por lo que no hay validación externa actual.
- **Revisar `reconstruccion_evolucion_tfg.md`** (largo, posible solape) antes de
  citar nada de él.
- **Confirmar título y tutor**: solo en `resumen_una_pagina.md` (Dir. Jorge Calvo
  Martín; título propuesto sobre "sesgo de dominio").

---

*Fuente de verdad de cifras: JSON de `docs/audit/`. Fuente de lógica: `src/`.
seed=42 (split/entrenamiento), seed=0 (bootstrap/proyecciones de los tiny
baselines y embeddings). **Excepción:** el bootstrap del IC95% de la CNN
intra-dominio BTC (`btc_cnn_kfold.py`) reutiliza la semilla de partición (42),
no 0. El bootstrap usa 2000 remuestreos.*
</content>
</invoke>
