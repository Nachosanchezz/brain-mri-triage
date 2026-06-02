# Esqueleto anotado de la memoria del TFG — `brain-mri-triage`

> **Propósito.** Esqueleto de redacción directo: para cada apartado se indica
> *qué contar*, *qué figura/tabla usar* y *qué cifras exactas citar* (con su
> archivo de origen en el repositorio). Objetivo de extensión: **40–50 páginas**
> de cuerpo.
>
> **Convenciones:**
> - **[cifra verificada]** = dato comprobado contra los JSON del repo; entre
>   paréntesis su archivo fuente.
> - **[cita externa — verificar]** = referencia bibliográfica sugerida que el
>   autor debe localizar y comprobar antes de citar (no inventar DOIs).
> - **[fig]** = figura disponible en `docs/audit/figures/`.
> - Fuentes de verdad internas: `docs/reconstruccion_evolucion_tfg.md`,
>   `docs/auditoria_resultados_sospechosos.md`, `docs/justificacion_cambios.md`,
>   `docs/audit/borrador_memoria.md`, `docs/audit/resumen_consolidado.{md,json}`.

---

## Mapa de cifras maestras (para tenerlas a mano al redactar)

| Resultado | Valor | Archivo fuente |
|---|---|---|
| Composición pool | 1167 pos (BraTS 580 + UPENN 587) / 1100 neg (IXI 577 + NKI 523) = 2267 | `data/processed/preprocessing_summary.json` |
| Split | train 1587 (817/770), val 340 (175/165), test 340 (175/165), seed 42 | `docs/justificacion_cambios.md` |
| Parámetros modelo | 504 553 (2 canales) / 504 229 (1 canal) | `src/models/cnn3d.py` (verificado) |
| CNN confundida (test) | AUC 0.999965, PR-AUC 0.999968, acc 0.99706, sen 0.99429, spe 1.0, TP174/FP0/TN165/FN1 | `outputs/evaluation/cnn3d_test_results.json` |
| Falso negativo único | `BraTS2021_00736`, score 0.092 | `docs/justificacion_cambios.md` |
| Scores por dataset | upenn 0.9995±0.0005, brats 0.986±0.098, ixi 0.0070±0.018, nki 0.0008±0.0005; AUC intra-dataset = NaN | `outputs/evaluation/cnn3d_test_results.json` |
| Tiny baseline (mezcla) | LogReg AUC 1.0 (acc 0.9941); RF AUC 0.9989 (acc 0.9882) | `docs/audit/audit_leakage.json` |
| Clf de dataset (intensidad) | acc 0.9853 (4 clases, azar 0.25) | `docs/audit/audit_leakage.json` |
| Duplicados / overlap | 0 duplicados, 0 cross-split, 0 overlap de sujeto | `docs/audit/audit_leakage.json` |
| `label_equals_dataset` | brats:[1], upenn:[1], ixi:[0], nki_rockland:[0] | `docs/audit/audit_leakage.json` |
| Features top (RF) | nz_frac_t1 0.17, p75_t1 0.17, p25_t2 0.13, nz_frac_t2 0.10 | `docs/audit/audit_leakage.json` |
| LODO A (CNN) | AUC 0.6236, sen 0.676, spe 0.463 | `outputs/evaluation/lodo_A/cnn3d_test_results.json` |
| LODO B (CNN) | AUC 0.2012, sen 0.0103, spe 0.9653 | `outputs/evaluation/lodo_B/cnn3d_test_results.json` |
| LODO tiny (A/B/C/D LogReg) | 0.9952 / 0.3184 / 0.0412 / 0.0469; ref random-mix 1.0 | `docs/audit/audit_lodo.json` |
| BTC tiny LogReg | AUC 0.5491, IC95 [0.319, 0.788] | `docs/audit/btc_intradomain_tinybaseline.json` |
| BTC tiny RF | AUC 0.4055, IC95 [0.215, 0.616] | `docs/audit/btc_intradomain_tinybaseline.json` |
| BTC CNN 1-canal | AUC 0.4036, IC95 [0.213, 0.623], sen 0.60, spe 0.364 | `outputs/evaluation/btc_intradomain/cnn_kfold_results.json` |
| Embeddings silhouette | label 0.754 / dataset 0.366 | `docs/audit/embeddings_silhouette.json` |
| Embeddings intra-clase | IXI-vs-NKI CV-AUC 0.998 (sil 0.561); BraTS-vs-UPENN CV-AUC 0.991 (sil 0.186) | `docs/audit/embeddings_intraclass.json` |
| Clf dataset desde embeddings | acc 0.982 (azar 0.25) | `docs/audit/embeddings_intraclass.json` |
| UPENN shell (Otsu) | máscara vieja pctl-5 = 62.99 % → Otsu = 22.43 %; nz_frac UPENN 0.66 vs ~0.16–0.21 resto | `docs/justificacion_cambios.md` |

---

# 1. Introducción  *(≈ 4 pp.)*

### 1.1. Contexto clínico y motivación
- **Qué va:** carga asistencial en radiología; el cuello de botella de la lectura
  de RM; idea de triaje (priorizar estudios con probable hallazgo). Distinguir
  triaje (priorización) de diagnóstico (juicio del radiólogo).
- **Cifras/fuente:** —
- **Citas:** [cita externa — verificar] estadísticas de volumen radiológico /
  tiempos de espera; rol de IA como apoyo.

### 1.2. Planteamiento del problema
- **Qué va:** formulación técnica: clasificación binaria **a nivel de estudio**
  (tumor/no tumor) sobre volúmenes T1+T2. Salida `P(tumor)` vía sigmoide.
  Sensibilidad como métrica prioritaria (coste asimétrico del falso negativo).
- **Fuente interna:** `src/models/cnn3d.py`, `src/evaluation/evaluate_3d.py`.

### 1.3. De la hipótesis inicial a la pregunta metodológica
- **Qué va (apartado clave, honesto):** anuncia el giro. La hipótesis inicial era
  «¿puede una CNN 3D detectar tumor para triaje?». El sistema dio métricas casi
  perfectas, lo que reorientó el trabajo hacia «¿esas métricas reflejan detección
  de tumor o un artefacto?». Plantéalo como reformulación legítima guiada por la
  evidencia, no como fracaso.
- **Cifra gancho:** AUC ≈ 0.99997 en test **[cifra verificada]** (`cnn3d_test_results.json`).

### 1.4. Objetivos y contribuciones
- **Qué va:** adelanto de los objetivos (cap. 2) y de las 2 contribuciones:
  (a) pipeline 3D reproducible para RM multi-fuente; (b) **protocolo de auditoría
  de validez** que detecta y cuantifica un confound dominio↔clase.

### 1.5. Estructura de la memoria
- **Qué va:** una frase por capítulo.

---

# 2. Objetivos  *(≈ 1,5 pp.)*

### 2.1. Objetivo general
- «Desarrollar **y validar críticamente** un sistema de triaje de RM cerebral
  basado en DL, determinando si su rendimiento constituye evidencia de detección
  de tumor.»

### 2.2. Objetivos específicos
- O1. Pipeline de preprocesado homogéneo multi-dataset.
- O2. Implementar y entrenar una CNN 3D de clasificación binaria.
- O3. Evaluar con métricas clínicas (sensibilidad, especificidad, AUC, PR-AUC).
- O4. **Auditar la validez**: descartar fugas triviales y cuantificar el sesgo de
  dominio.
- O5. Evaluar la **generalización fuera de dominio** (LODO) e **intra-dominio**
  (BTC_preop).
- O6. Analizar la **representación interna** (embeddings) e **interpretabilidad**
  (Grad-CAM).

---

# 3. Estado del arte  *(≈ 7–8 pp.)*

### 3.1. Diagnóstico por imagen de masas tumorales intracraneales
- **Qué va:** gliomas/meningiomas; papel de la imagen; por qué importa la detección
  precoz. **Citas:** [cita externa — verificar].

### 3.2. RM cerebral y limitaciones del diagnóstico radiológico
- **Qué va:** modalidades (T1, T2, FLAIR…); variabilidad inter-observador y de
  protocolo/escáner. Sienta la base del problema de dominio.

### 3.3. Deep Learning en imagen médica como apoyo al diagnóstico
- **Qué va:** CNN en imagen médica; clasificación vs segmentación; CADx/CADe.

### 3.4. DL aplicado a RM cerebral: enfoques, datasets y retos de generalización
- **Qué va:** BraTS y la línea de segmentación/clasificación; transfer learning;
  el reto de generalizar entre centros. **Citas:** [cita externa — verificar]
  BraTS (Menze 2015; Baid 2021); UPENN-GBM (TCIA); IXI; NKI-Rockland (FCP-INDI).

### 3.5. Sesgo de dominio, *shortcut learning* y fugas de información en DL médico
- **Qué va (apartado NUEVO y central):** marco teórico de tu hallazgo. Atajos,
  confound, dataset bias; por qué métricas altas multi-fuente pueden ser espurias.
- **Citas candidatas [verificar]:** Geirhos et al. 2020 «Shortcut learning in deep
  neural networks»; Zech et al. 2018 (confounding en radiografía de tórax entre
  hospitales); DeGrave et al. 2021 (atajos en detección de COVID por rayos X);
  literatura de *domain shift*/*harmonization* en neuroimagen (p. ej. ComBat).

### 3.6. Sistemas de priorización/triaje en radiología
- **Qué va:** worklist prioritization, productos comerciales/regulatorios;
  requisitos de sensibilidad y de validación fuera de dominio.

---

# 4. Materiales y datos  *(≈ 5 pp.)*

### 4.1. Datasets
- **Qué va:** ficha de cada uno. **Tabla** (de `reconstruccion_evolucion_tfg.md §7`):

  | Dataset | Clase | n | Modalidades | Script | Papel |
  |---|---|---|---|---|---|
  | BraTS 2021 | Pos | 580 | T1+T2 | `preprocess_brats.py` | Entrenamiento |
  | UPENN-GBM | Pos | 587 | T1+T2 | `preprocess_upenn.py` | Entrenamiento (era test externo) |
  | IXI | Neg | 577 | T1+T2 | `preprocess_ixi.py` | Entrenamiento |
  | NKI Rockland | Neg | 523 | T1+T2 | `preprocess_nki_rockland.py` | Entrenamiento (reequilibrio) |
  | BTC_preop (ds001226) | Mixto | 25/11 | T1-only | `preprocess_btc.py` | Validación intra-dominio |

- **Cifras:** [cifra verificada] (`preprocessing_summary.json`). BTC: ds001226,
  Ghent University Hospital, CC0; 25 pacientes (gliomas/meningiomas) / 11 controles
  (`participants.tsv`).
- **Decisión a declarar:** UPENN-GBM pasó de test externo a entrenamiento →
  reformulación a «robustez multi-fuente» y pérdida de validación externa
  (`docs/evolucion_respecto_primera_entrega.md §1.2`).

### 4.2. Definición de clases y etiquetado
- **Qué va:** label 1 = tumor (BraTS, UPENN); label 0 = sano (IXI, NKI). En BTC,
  `sub-PAT*`→1, `sub-CON*`→0.

### 4.3. Composición del problema: relación entre clase y dataset de origen
- **Qué va (apartado NUEVO, prepara el confound como hecho de diseño):** explicitar
  que **ningún dataset del pool aporta ambas clases** → correlación clase↔dataset
  del 100 %. Presentarlo aquí como característica de los datos, no como sorpresa.
- **Cifra:** `label_equals_dataset` = {brats:[1], upenn:[1], ixi:[0], nki:[0]}
  **[cifra verificada]** (`audit_leakage.json`).

### 4.4. Selección de modalidades
- **Qué va:** T1+T2 como modalidades comunes; consecuencia: BTC es T1-only y obliga
  a un experimento de 1 canal.

### 4.5. Análisis exploratorio de los datos
- **Qué va:** recuentos, balance, distribución de intensidades por dataset.
- **[fig] `intensity_by_dataset.png`** — los 4 datasets son separables ya en
  estadísticos triviales de intensidad (anticipa §7.2).
- **Notebook:** `notebooks/explorar_processed_dinamico.ipynb`.

---

# 5. Metodología  *(≈ 6 pp.)*

### 5.1. Preprocesado homogéneo
- **Qué va:** pipeline DICOM→NIfTI (dcm2niix) → HD-BET (IXI, NKI, BTC) → RAS →
  remuestreo 1 mm³ → crop/pad (192,224,192) → z-score sobre máscara de tejido con
  **umbral Otsu**. T2 remuestreado a la rejilla de T1.
- **Cifra/decisión:** corrección del «shell» de UPENN: máscara vieja (pctl-5)
  62.99 % → Otsu 22.43 %; nz_frac UPENN 0.66 vs ~0.16–0.21 resto **[cifra
  verificada]** (`justificacion_cambios.md` paso 8). **Matiz honesto:** Otsu
  corrige el shift *dentro* de la clase positiva, **no** el confound estructural.
- **Fuente:** `src/preprocessing/base_preprocessing.py`, `src/preprocessing/README.md`.

### 5.2. Arquitectura del modelo
- **Qué va:** `BrainTumorCNN3D`: 4 bloques Conv3D (12→24→48→96) con **GroupNorm**,
  MaxPool, AdaptiveAvgPool3d, cabeza FC 96→96→1 (1 logit). Entrada (2,128,160,128);
  variante 1 canal para BTC.
- **Nota:** el baseline 2D (ResNet18) **se excluye** de la memoria: su código no se
  conserva en el repositorio y su evaluación no es reproducible (solo sobreviven el
  checkpoint y `baseline_2d_history.json`). Además, comparar AUC entre modelos sobre
  el pool confundido reproduciría el sesgo que el trabajo desmonta.
- **Cifra:** **504 553 parámetros** (504 229 en 1 canal) **[cifra verificada]**.
- **Decisión técnica:** GroupNorm en lugar de BatchNorm por `batch_size=1`
  (`docs/justificacion_cambios.md` paso 4). **[fig]** (opcional) diagrama de la red.

### 5.3. Función de pérdida, optimización e hiperparámetros
- **Qué va / tabla** (de `configs/train_3d.yaml`): `BCEWithLogitsLoss`
  (`pos_weight=1.0`), AdamW, lr 5e-5, weight_decay 1e-3, scheduler cosine, AMP off,
  n_epochs 40, patience 10, `min_sensitivity_for_save 0.80`.

### 5.4. Data augmentation
- **Qué va:** flips en 3 ejes + gamma (0.8–1.25) + ruido gaussiano (σ=0.03)
  **aplicados solo sobre vóxeles ≠0** (máscara de cerebro), para no inyectar señal
  en el fondo y forzar contraste relativo. Fuente: `dataset_3d.py::_augment`.

### 5.5. Estrategia de partición a nivel de sujeto
- **Qué va:** `create_splits`: estratificación por `(dataset, label)` + agrupación
  por `subject_id`; 70/15/15; seed 42.
- **Cifra:** train 1587 / val 340 / test 340 **[cifra verificada]**.

### 5.6. Entorno computacional y herramientas
- **Qué va:** GPU AMD Radeon RX 6700 XT, ROCm, PyTorch 2.5.1+rocm6.2, env conda
  `igsan`; HD-BET, nibabel, scikit-learn/image. Fuente: `requirements.txt`.

---

# 6. Diseño experimental y protocolo de validación  *(≈ 5 pp.)*

> Este capítulo describe **el diseño**; los resultados van en el cap. 7 en el mismo
> orden.

### 6.1. Métricas de evaluación y análisis operativo
- **Qué va:** AUC (Mann-Whitney), PR-AUC, sensibilidad, especificidad, precision,
  NPV, F1, balanced accuracy, matriz de confusión; selección de umbral en
  **validación** y aplicación a test (Youden / sensibilidad mínima). IC95 % por
  bootstrap (2000 resamples) en los experimentos intra-dominio.
- **Fuente:** `evaluate_3d.py`, `threshold_analysis.py`, `btc_*` (bootstrap).

### 6.2. Validación interna sobre el pool multi-fuente
- **Qué va:** protocolo del run principal; el test se evalúa una sola vez al final;
  checkpoint por balanced_accuracy con sen ≥ 0.80.

### 6.3. Protocolo de auditoría de validez (visión general)
- **Qué va:** justifica *por qué* auditar (métricas implausibles) y enumera la
  batería: control trivial → identificabilidad de dominio → LODO → intra-dominio →
  latente/Grad-CAM. Esquema de embudo: descartar lo trivial, aislar el confound.
- **Fuente:** `docs/auditoria_resultados_sospechosos.md`.

### 6.4. Pruebas de control: baseline trivial sobre estadísticos de intensidad
- **Qué va:** 16 features no clínicas (nz_frac, media, std, percentiles de T1/T2);
  LogReg/RF para predecir etiqueta y para predecir dataset; duplicados por sha1.
- **Fuente:** `src/audit/audit_leakage.py`.

### 6.5. Validación cruzada por dominio (LODO)
- **Qué va:** definición de *leave-one-domain-out*; configuraciones A/B/C/D; el
  test toma todos los sujetos de los datasets held-out; config idéntica a la
  principal salvo `recreate_splits:false`.
- **Fuente:** `make_lodo_splits.py`, `configs/train_lodo.yaml`, `run_lodo.sh`,
  `audit_lodo.py`.

### 6.6. Validación intra-dominio (BTC_preop)
- **Qué va:** por qué BTC es la única medición honesta (ambas clases, mismo
  escáner); k-fold estratificado por sujeto, 20 épocas fijas/fold sin early stopping
  (no contaminar la selección); T1-only; IC95 % bootstrap. Limitaciones: n=36,
  T1-only.
- **Fuente:** `preprocess_btc.py`, `btc_tiny_baseline.py`, `btc_cnn_kfold.py`,
  `run_btc_chain.sh`.

### 6.7. Análisis del espacio latente e interpretabilidad
- **Qué va:** extracción del vector latente de 96-d (antes del clasificador);
  PCA/t-SNE; silhouette por label y por dataset; silhouette y LogReg-CV intra-clase;
  clasificador de dataset desde embeddings; Grad-CAM 3D. Limitaciones de Grad-CAM
  (cualitativo, baja resolución tras los poolings).
- **Fuente:** `embeddings_tsne.py`, `embeddings_intraclass.py`, `gradcam_3d.py`.

---

# 7. Resultados  *(≈ 8–9 pp.)*

> Narrativa en cascada: el AUC≈1.0 se desmonta paso a paso.

### 7.1. Rendimiento aparente en el pool multi-fuente
- **Cifras [verificadas]** (`cnn3d_test_results.json`): AUC 0.999965, PR-AUC
  0.999968, acc 0.99706, sen 0.99429, spe 1.0; TP174/FP0/TN165/FN1; único FN
  `BraTS2021_00736` (0.092).
- **Tabla** de scores por dataset: upenn 0.9995±0.0005, brats 0.986±0.098, ixi
  0.0070±0.018, nki 0.0008±0.0005; **AUC intra-dataset = NaN** (mono-clase).
- **[fig] `confusion_matrices.png`** (panel confounded). Texto: la varianza
  intra-dataset casi nula y la imposibilidad de AUC intra-dataset son ya señales de
  alarma.

### 7.2. Decodificabilidad trivial de la etiqueta (tiny baseline)
- **Cifras [verificadas]** (`audit_leakage.json`): LogReg AUC **1.0** (acc 0.9941),
  RF AUC 0.9989. Features top: nz_frac_t1 0.17, p75_t1 0.17, p25_t2 0.13.
- **[fig] `intensity_by_dataset.png`** (reutilizable aquí). Texto: la etiqueta es
  decodificable sin red ni información espacial → la señal es de dominio/preprocesado.

### 7.3. Identificabilidad del dominio (clasificador de dataset)
- **Cifras [verificadas]:** clf de dataset (4 clases) acc **0.9853** vs azar 0.25
  (`audit_leakage.json`). Duplicados 0; overlap de sujeto 0 (descarta fugas
  triviales) (`audit_leakage.json`, `audit_splits`).

### 7.4. Generalización fuera de dominio (LODO)
- **Cifras [verificadas]:** CNN LODO A AUC **0.6236** (sen 0.676, spe 0.463); LODO B
  AUC **0.2012** (sen 0.0103, spe 0.9653) (`lodo_{A,B}/cnn3d_test_results.json`).
  Tiny LODO A/B/C/D LogReg 0.9952/0.3184/0.0412/0.0469; ref random-mix 1.0
  (`audit_lodo.json`).
- **[fig] `roc_curves.png`**, **`score_hist_lodo.png`**, panel LODO de
  `confusion_matrices.png`. Texto: AUC<0.5 en B = regla **invertida**; colapso de
  sensibilidad (0.01). Incompatible con detección real.

### 7.5. Validación intra-dominio honesta (BTC_preop)
- **Cifras [verificadas]** (`btc_intradomain_tinybaseline.json`,
  `btc_intradomain/cnn_kfold_results.json`): tiny LogReg AUC 0.5491 [0.319, 0.788];
  tiny RF 0.4055 [0.215, 0.616]; **CNN 1-canal AUC 0.4036 [0.213, 0.623]**, sen
  0.60, spe 0.364.
- **[fig] `btc_kfold_bars.png`**. Texto: con el confound eliminado, todo cae al
  azar (IC cruzan 0.5). Matiz: n=36, T1-only → «no detectamos señal», no «no existe».

### 7.6. Evidencia en el espacio latente y mapas de atención
- **Cifras [verificadas]** (`embeddings_silhouette.json`, `embeddings_intraclass.json`):
  silhouette label 0.754 / dataset 0.366; **IXI-vs-NKI (ambos sanos) CV-AUC 0.998**;
  BraTS-vs-UPENN 0.991; clf de dataset desde embeddings acc 0.982.
- **[fig] `embeddings_tsne.png`** (figura estrella), `embeddings_pca.png`,
  `gradcam/confound/*`. Texto: el latente identifica el centro incluso dentro de una
  misma clase → codifica procedencia, no tumor.

### 7.7. Síntesis consolidada de resultados
- **Tabla maestra** (de `docs/audit/resumen_consolidado.md`): de más confundido a
  más honesto. **[fig] `auc_summary.png`** (figura principal de la memoria).

---

# 8. Discusión  *(≈ 5 pp.)*

### 8.1. Interpretación: el confound estructural dominio↔clase
- **Qué va:** integrar las 5 evidencias; explicar el mecanismo (positivos y
  negativos de fuentes disjuntas).

### 8.2. Por qué las métricas casi perfectas no son evidencia clínica
- **Qué va:** un modelo lineal trivial iguala a la CNN; varianza intra-dataset ~0;
  AUC intra-dataset no medible. No suavizar.

### 8.3. Viabilidad y condiciones para un triaje radiológico válido
- **Qué va:** qué haría falta para una afirmación clínica (ambas clases por dominio,
  n mayor, multicéntrico, validación prospectiva). El sistema **no** es desplegable
  como detector hoy.

### 8.4. Comparación con la literatura
- **Qué va:** encuadrar como caso de *shortcut learning* / dataset bias. **Citas
  [verificar]:** Geirhos 2020; Zech 2018; DeGrave 2021.

### 8.5. Validez metodológica y reproducibilidad
- **Qué va:** seed fijo, IC95 %, split por sujeto, protocolo reutilizable; ser
  honesto con `recreate_splits:true` (pendiente de congelar).

---

# 9. Conclusiones  *(≈ 1,5 pp.)*
- **Qué va:** (1) el modelo no es defendible como detector de tumor; (2) el AUC≈1.0
  se explica por confound; (3) la aportación es la detección, demostración y
  documentación del fallo + el protocolo de auditoría; (4) lección metodológica
  para DL médico multi-fuente. Apoyarse en `reconstruccion_evolucion_tfg.md §24`.

---

# 10. Limitaciones y líneas futuras  *(≈ 2,5 pp.)*

### 10.1. Limitaciones del estudio
- Confound estructural; ausencia de validación externa real (UPENN en train);
  BTC n=36 y T1-only; sin calibración; LODO C/D solo con tiny baseline;
  `recreate_splits:true`.

### 10.2. Propuestas de mejora y líneas futuras
- Cohorte intra-dominio mayor (n≥100) y multicéntrica con ambas clases; T1+T2;
  calibración (Platt, reliability diagrams); domain-adversarial / armonización
  (discutir su límite: si dominio≡clase al 100 %, eliminar dominio elimina clase);
  LODO completo; validación prospectiva.
- **Fuente:** `borrador_memoria.md §6.4, §7`.

---

# Referencias bibliográficas
- Mezcla de: datasets (BraTS, UPENN-GBM/TCIA, IXI, NKI-Rockland/FCP-INDI, ds001226/
  Aerts et al.), herramientas (HD-BET/Isensee, nnU-Net), y marco teórico
  (shortcut learning, dataset bias, harmonization). **Todas [cita externa —
  verificar].**

# Anexos
- **A.** Detalle del preprocesado y `configs/` completas.
- **B.** Tabla maestra consolidada (`resumen_consolidado.json`) y per-fold de BTC.
- **C.** Figuras adicionales (Grad-CAM LODO A/B, histogramas por dataset).
- **D.** Reproducibilidad: comandos (de `reconstruccion_evolucion_tfg.md §22`),
  seeds, entorno.

---

## Inventario de figuras → dónde colocarlas

| Figura (`docs/audit/figures/`) | Apartado | Función |
|---|---|---|
| `intensity_by_dataset.png` | 4.5 / 7.2 | Confound visible en píxeles crudos |
| `confusion_matrices.png` | 7.1 / 7.4 | Diagonal perfecta → colapso/inversión → dispersión |
| `roc_curves.png` | 7.4 | ROC superpuestas (confounded/LODO/intra) |
| `score_hist_confound.png` | 7.1 | Scores bimodales por dataset |
| `score_hist_lodo.png` | 7.4 | Scores descolocados cross-dataset |
| `btc_kfold_bars.png` | 7.5 | AUC por fold disperso, IC ancho |
| `embeddings_tsne.png` | 7.6 | **Figura estrella**: latente agrupa por procedencia |
| `embeddings_pca.png` | 7.6 | Versión lineal |
| `gradcam/confound/*` | 7.6 | Atención fuera de la lesión |
| `auc_summary.png` | 7.7 | **Figura principal**: AUC↓ de ~1.0 a azar con IC95 % |

**Selección mínima si hay que recortar:** `auc_summary`, `embeddings_tsne`,
`confusion_matrices`, un `gradcam/confound`.

---

*Esqueleto generado como andamiaje de redacción. Cifras verificadas contra los JSON
del repositorio; referencias bibliográficas marcadas como pendientes de verificar.*
