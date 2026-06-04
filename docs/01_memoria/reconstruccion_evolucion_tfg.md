# Reconstrucción técnica y metodológica del proyecto `brain-mri-triage`

> **Documento base para la memoria del TFG.**
> Reconstrucción completa, honesta y trazable de la evolución del repositorio desde
> el commit inicial hasta el estado actual.
>
> **Convención de honestidad.** A lo largo del documento se distingue:
> - **Hecho observado**: verificado leyendo directamente el código, los JSON de
>   resultados o el historial de Git. Se acompaña de la ruta del archivo.
> - **Interpretación**: lectura metodológica del autor/revisor; se introduce con
>   fórmulas del tipo «se interpreta que», «esto sugiere», «es coherente con».
> - **No confirmable / pendiente**: cuando un dato no puede verificarse con la
>   información del repositorio se indica explícitamente.
>
> Las cifras proceden de los JSON reales del repositorio (`docs/audit/*.json`,
> `outputs/evaluation/*.json`) y se citan con su archivo de origen. No se ha
> inventado ninguna métrica.

---

## 1. Resumen ejecutivo

El proyecto `brain-mri-triage` nació como un sistema de **triaje automático de
resonancia magnética (RM) cerebral** mediante *Deep Learning*: una clasificación
binaria a nivel de estudio (tumor / no tumor) sobre volúmenes T1+T2, concebida
como herramienta de **priorización** para el radiólogo, con la **sensibilidad**
como métrica clínica prioritaria.

Durante el desarrollo se entrenó una CNN 3D propia (`BrainTumorCNN3D`) sobre un
pool de cuatro datasets y se obtuvieron métricas **casi perfectas**: AUC ≈ 0.99997,
sensibilidad ≈ 0.994 y especificidad = 1.0 en el conjunto de test
(`outputs/evaluation/cnn3d_test_results.json`). En un problema clínico real, un
resultado de esta magnitud es **intrínsecamente sospechoso**.

La parte central y más madura del trabajo es la **auditoría metodológica** que
siguió a esa observación. Dicha auditoría descartó las fugas triviales (duplicados
y solapamiento de sujetos entre particiones) y, sobre todo, demostró que el
rendimiento se explica por un **confound estructural dominio↔clase**: en el pool de
entrenamiento la etiqueta coincide al 100 % con el dataset de origen
(`label = 1 ⟺ dataset ∈ {BraTS, UPENN}`; `label = 0 ⟺ dataset ∈ {IXI, NKI}`),
verificado en `docs/audit/audit_leakage.json` (campo `label_equals_dataset`).

Las pruebas de control son contundentes y se desarrollan en este documento:

1. Un *tiny baseline* lineal sobre 16 estadísticos de intensidad **no clínicos**
   alcanza **AUC = 1.0** sin red neuronal ni información espacial.
2. Un test de generalización cruzada (LODO) hace caer la CNN a **AUC ≈ 0.624** en
   una dirección y **≈ 0.201** en la opuesta (valor < 0.5 = regla invertida).
3. Una validación **intra-dominio** sobre BTC_preop (ambas clases del mismo
   escáner) sitúa la CNN en **AUC ≈ 0.404, IC95 % ≈ [0.21, 0.62]**, indistinguible
   del azar.
4. El espacio latente de la red identifica el **dataset de origen** incluso entre
   sujetos de la misma clase (IXI vs NKI, ambos sanos: CV-AUC ≈ 0.998).

**Conclusión metodológica.** El AUC ≈ 1.0 inicial **no constituye evidencia de
detección de tumor**; es atribuible al confound de dominio. El TFG no termina en
«una CNN que clasifica tumores», sino en **una demostración rigurosa y reproducible
de por qué esa CNN parecía funcionar y por qué su rendimiento no es clínicamente
válido**. Esa transición —de clasificador aparentemente exitoso a auditoría de
validez— es la aportación real del trabajo.

---

## 2. Objetivo clínico y técnico del TFG

**Problema clínico.** En la práctica radiológica, el volumen de estudios de RM
cerebral pendientes de lectura puede generar demoras. Un sistema de **triaje**
busca **priorizar** los estudios con mayor probabilidad de hallazgo patológico
(en este caso, masa tumoral) para que el radiólogo los lea antes, sin sustituir
en ningún momento su juicio diagnóstico.

**Formulación técnica.** Clasificación binaria **a nivel de estudio** (no de vóxel
ni de corte): para cada par de volúmenes T1+T2 de un sujeto, predecir
`P(tumor) ∈ [0, 1]`. El repositorio implementa exactamente esto: la CNN emite un
único logit y se aplica `sigmoid` para obtener la probabilidad de tumor
(`src/models/cnn3d.py`, `src/evaluation/evaluate_3d.py::positive_probability`).

**Sensibilidad como métrica crítica.** En triaje, el coste de un **falso negativo**
(no priorizar un estudio con tumor) es clínicamente mucho mayor que el de un falso
positivo (priorizar un estudio sano). Por ello la configuración y la selección de
modelo del repositorio penalizan explícitamente la pérdida de sensibilidad: el
checkpoint solo se guarda si `sensitivity ≥ 0.80` (`configs/train_3d.yaml`,
`min_sensitivity_for_save`) y el análisis de umbral incluye un criterio de
«sensibilidad mínima objetivo» (`src/evaluation/threshold_analysis.py`).

**Soporte, no sustitución.** El sistema se plantea como ayuda a la priorización;
no emite diagnóstico. Esto es relevante para la discusión clínica: un triaje con
falsos positivos es tolerable; uno con falsos negativos sistemáticos es peligroso.

**Importancia de la validación fuera de dominio.** Dado que un sistema de triaje
debería funcionar sobre estudios procedentes de escáneres y protocolos no vistos
durante el entrenamiento, la **generalización fuera de dominio** no es un lujo
sino un requisito. Como se verá, es precisamente este eje el que destapa la
ausencia de validez del modelo entrenado sobre el pool confundido.

---

## 3. Estado inicial del repositorio

La primera etapa del repositorio (commits `97204af` a `3c0bb2e`) establece el
esqueleto del proyecto:

- **`97204af` — Initial commit.** Crea `.gitignore` y un `README.md` que contiene
  únicamente el título `# brain-mri-triage` (hecho observado: el README no se ha
  ampliado posteriormente).
- **`3c0bb2e` — Requirements.txt.** Añade `requirements.txt` y amplía `.gitignore`.

**Dependencias declaradas** (`requirements.txt`, hecho observado):
`nibabel`, `scikit-image`, `numpy`, `scipy`, `scikit-learn`, `tqdm`,
`torch==2.5.1+rocm6.2`, `torchvision==0.20.1+rocm6.2`, `monai`, `pyyaml`,
`HD-BET==2.0.1`, `pydicom`, `matplotlib`, `pandas`, `boto3`.

De aquí se desprenden dos hechos relevantes:

1. La pila de PyTorch es **ROCm** (`+rocm6.2`), es decir, el entrenamiento se
   realizó sobre **GPU AMD** (coherente con el hardware AMD Radeon documentado en
   `docs/justificacion_cambios.md`: «GPU AMD Radeon RX 6700 XT (ROCm)»).
2. La presencia de `HD-BET` (skull-stripping basado en *deep learning*), `pydicom`
   (lectura DICOM) y `boto3` (acceso a S3) anticipa el pipeline de datos.

La primera memoria del TFG se incorpora más tarde como binario `docs/TFG.pdf`
(commit `8770f9b`, 163 KB). No se ha inspeccionado su contenido por ser PDF; se
interpreta que recoge el planteamiento de la primera entrega.

El objetivo de esta etapa, según se interpreta del historial, era **disponer de un
pipeline funcional** (preprocesado → modelo → evaluación) antes de abordar el
problema científico.

---

## 4. Evolución cronológica del repositorio

Historial completo de la rama `main` (hecho observado: `git log --oneline`,
12 commits, lineal y sin *merges*):

```
d2c1997 Add audit pipeline and LODO validation for 3D CNN
5bda6f1 Notebook testeo
dc71220 Evaluación modelo
3cf7807 codigo de modelo
4d84cd1 Prueba entrenamiento
b933ef6 Notebook visualización
2be8733 Cambi configuración gitignore
b5cb5d6 Explicación de dataset de carga y preprocesado
609fe40 Reorganización del repositorio para el preprocesado
8770f9b Preprocesado
3c0bb2e Requirements.txt
97204af Initial commit
```

(Existe además una rama `claude/eloquent-nash-e022a0` apuntando al commit inicial
y un *worktree* en `.claude/`; no aportan contenido al estado final.)

| # | Hash | Mensaje | Archivos que aparecen/cambian | Problema que resolvía | Mejora introducida | Limitación abierta | Conexión con la etapa siguiente |
|---|---|---|---|---|---|---|---|
| 1 | `97204af` | Initial commit | `.gitignore`, `README.md` | Arranque del repo | Estructura mínima | README vacío de contenido | Añadir dependencias |
| 2 | `3c0bb2e` | Requirements.txt | `requirements.txt`, `.gitignore` | Reproducibilidad de entorno | Pila ROCm + HD-BET + pydicom + boto3 | Sin versionado de lockfile | Construir preprocesado |
| 3 | `8770f9b` | Preprocesado | `src/preprocessing/*` (base, conversores DICOM, preprocess_brats/ixi/upenn/volumes, skull_strip_ixi), `docs/TFG.pdf` | Homogeneizar datasets heterogéneos | Primer pipeline RAS→resample→crop→z-score y scripts por dataset | Código monolítico/duplicado | Refactor modular |
| 4 | `609fe40` | Reorganización del preprocesado | `conversion/`, `dataset/`; borrado de conversores y `skull_strip_ixi.py` antiguos | Mantenibilidad | Modularización (`conversion/dicom_to_nifti.py`, etc.) | Persiste `preprocess_volumes.py` legacy | Documentar |
| 5 | `b5cb5d6` | Explicación de dataset y preprocesado | `src/preprocessing/README.md` | Documentar el pipeline | README detallado del preprocesado | **Quedará desactualizado** (ver §19) | Explorar datos |
| 6 | `b933ef6` | Notebook visualización | `notebooks/explorar_processed_dinamico.ipynb` | Inspección de `data/processed` | EDA dinámico de los `.npz` | — | Modelar |
| 7 | `4d84cd1` | Prueba entrenamiento | `configs/train_3d.yaml`, `src/data/dataset_3d.py`, `src/models/cnn3d.py`, `src/training/train_3d.py` | Tener un modelo entrenable | Primera versión CNN 3D + Dataset + config | Inestabilidad de entrenamiento (ver §9) | Afinar |
| 8 | `3cf7807` | codigo de modelo | `train_3d.py`, `dataset_3d.py`, `train_3d.yaml` | Ajustes del bucle | Mejoras de entrenamiento | Métricas/checkpoint aún por madurar | Evaluar |
| 9 | `dc71220` | Evaluación modelo | `src/evaluation/evaluate_3d.py` (+328 líneas) | Medir el modelo | Métricas globales y **por dataset** | Sin IC ni calibración | Inspección manual |
| 10 | `5bda6f1` | Notebook testeo | `notebooks/test_modelo_3d.ipynb` | Revisar aciertos/fallos | Inspección de predicciones de test | — | Auditoría |
| 11 | `d2c1997` | **Add audit pipeline and LODO validation** | **67 ficheros, +7359/−378** | **Validez metodológica** | Toda `src/audit/`, LODO, BTC, embeddings, Grad-CAM, GroupNorm, Otsu, `threshold_analysis.py`, `audit_splits.py`, 4 documentos y figuras | Docs antiguos quedan stale; faltan validación externa real y calibración | Redacción de la memoria |

**Observación clave (hecho observado):** el commit `d2c1997` concentra el grueso
intelectual del proyecto. Es el momento en que el trabajo deja de ser «entrenar un
clasificador» y pasa a ser «auditar por qué el clasificador parece perfecto».

**Cambios sin commitear (hecho observado: `git status` al inicio de la sesión).**
El análisis de **embeddings** (E1) es el desarrollo más reciente y aún no estaba
commiteado: aparecen como modificados `docs/audit/borrador_memoria.md`,
`docs/audit/resumen_consolidado.md` y `src/audit/consolidate_results.py`, y como
nuevos `docs/audit/embeddings.npz`, `embeddings_intraclass.json`,
`embeddings_silhouette.json`, `src/audit/embeddings_intraclass.py` y las figuras
`embeddings_{pca,tsne}.png`.

---

## 5. Estructura final del repositorio

| Ruta | Función | Etapa del pipeline | Importancia | Comentario técnico |
|---|---|---|---|---|
| `configs/train_3d.yaml` | Config CNN 3D principal | Entrenamiento | Alta | Parámetros del run confundido (`recreate_splits: true`) |
| `configs/train_lodo.yaml` | Config LODO | Auditoría (OOD) | Alta | Clon de la anterior con `recreate_splits: false` |
| `src/data/dataset_3d.py` | Dataset + splits | Datos/Entrenamiento | Crítica | `create_splits` estratifica por (dataset,label), agrupa por subject_id; `BrainMRI3DDataset` recorta a 128×160×128 y augmenta |
| `src/data/audit_splits.py` | Overlap de sujetos | Auditoría | Media | Verifica 0 solapamiento; solo stdout |
| `src/models/cnn3d.py` | `BrainTumorCNN3D` | Modelo | Crítica | GroupNorm, 4 bloques conv, 1 logit |
| `src/training/train_3d.py` | Entrenamiento end-to-end | Entrenamiento | Crítica | Checkpoint por balanced_acc + sen mínima |
| `src/evaluation/evaluate_3d.py` | Métricas globales y por dataset | Evaluación | Alta | AUC por rangos; PR-AUC; por dataset |
| `src/evaluation/threshold_analysis.py` | Umbral val→test | Evaluación | Media | Youden / sensibilidad mínima |
| `src/preprocessing/base_preprocessing.py` | Núcleo de preprocesado | Datos | Crítica | RAS, resample, crop, **z-score con Otsu** |
| `src/preprocessing/preprocess_brats.py` | BraTS → positivos | Datos | Alta | label=1 |
| `src/preprocessing/preprocess_ixi.py` | IXI → negativos | Datos | Alta | Skull-strip HD-BET |
| `src/preprocessing/preprocess_upenn.py` | UPENN → positivos | Datos | Alta | Filtra T1/T2; series CaPTk |
| `src/preprocessing/preprocess_nki_rockland.py` | NKI → negativos | Datos | Alta | Skull-strip HD-BET |
| `src/preprocessing/preprocess_btc.py` | BTC → intra-dominio | Datos (auditoría) | Alta | T1-only; CON=0, PAT=1 |
| `src/preprocessing/download_nki_rockland.py` | Descarga S3 NKI | Datos | Media | Filtra T1w+T2w misma sesión |
| `src/preprocessing/conversion/dicom_to_nifti.py` | DICOM→NIfTI | Datos | Media | `dcm2niix` |
| `src/preprocessing/dataset/summarize_processed.py` | Resumen de processed | Datos | Baja | Reconstruye summary JSON |
| `src/preprocessing/preprocess_volumes.py` | Lanzador legacy BraTS+IXI | Datos | Baja | Redundante (marcado por el propio repo) |
| `src/audit/audit_leakage.py` | Tiny baseline + clf dataset + duplicados | Auditoría | Crítica | Prueba B y C; sha1 |
| `src/audit/audit_lodo.py` | LODO con tiny baseline | Auditoría (OOD) | Alta | 4 configs + referencia |
| `src/audit/make_lodo_splits.py` | Splits LODO A/B/C/D | Auditoría (OOD) | Alta | Escribe `splits.json` |
| `src/audit/btc_tiny_baseline.py` | Tiny baseline k-fold intra-dominio | Auditoría | Alta | IC95% bootstrap |
| `src/audit/btc_cnn_kfold.py` | CNN 1-canal k-fold intra-dominio | Auditoría | Crítica | 20 épocas/fold, sin early stopping |
| `src/audit/embeddings_tsne.py` | Latente 96-d + PCA/t-SNE | Auditoría (E1) | Alta | Silhouette |
| `src/audit/embeddings_intraclass.py` | Silhouette intra-clase + clf dataset | Auditoría (E1) | Alta | IXI-vs-NKI, BraTS-vs-UPENN |
| `src/audit/gradcam_3d.py` | Grad-CAM 3D | Interpretabilidad | Media | Mapas de atención |
| `src/audit/make_plots.py` | ROC, AUC summary, histogramas, kfold | Figuras | Media | A partir de JSON/CSV |
| `src/audit/make_extra_figures.py` | Intensidad por dataset + confusión | Figuras | Media | E2 y E3 |
| `src/audit/consolidate_results.py` | Tabla maestra consolidada | Consolidación | Alta | Genera `resumen_consolidado.{md,json}` |
| `scripts/run_btc_chain.sh` | Pipeline BTC completo | Orquestación | Alta | Con timeouts |
| `scripts/run_lodo.sh` | Driver de una config LODO | Orquestación | Alta | split→train→eval |
| `notebooks/explorar_processed_dinamico.ipynb` | EDA | Exploración | Baja | — |
| `notebooks/test_modelo_3d.ipynb` | Inspección de predicciones | Evaluación | Baja | — |
| `docs/auditoria_resultados_sospechosos.md` | Informe de auditoría | Documentación | Crítica | Veredicto «no defendible» |
| `docs/justificacion_cambios.md` | Los 8 pasos + run confundido | Documentación | Alta | Vigente |
| `docs/evolucion_respecto_primera_entrega.md` | Evolución vs 1ª entrega | Documentación | Media | **Parcialmente desactualizado** |
| `docs/audit/borrador_memoria.md` | Columna vertebral de la memoria | Documentación | Crítica | Tabla maestra + narrativa |
| `docs/audit/resumen_consolidado.md/.json` | Síntesis de resultados | Documentación | Alta | Generado por script |

(Carpetas `data/` y `outputs/` ignoradas por `.gitignore`; ver §6.)

---

## 6. Pipeline de datos y preprocesado

El objetivo declarado del preprocesado (`src/preprocessing/README.md`) es que
**todos los datasets compartan el mismo formato** para que «la red no aprenda
diferencias artificiales entre datasets». Es relevante señalar, ya desde aquí, que
este objetivo **no se alcanzó plenamente** y que esa es la raíz del problema que el
proyecto acaba documentando.

**Pipeline común** (hecho observado: `src/preprocessing/base_preprocessing.py`):

1. **DICOM → NIfTI** (cuando aplica): `conversion/dicom_to_nifti.py` con `dcm2niix`
   para BraTS y UPENN (lee `SeriesDescription` y usa series `Processed_CaPTk` en
   UPENN).
2. **Skull-stripping con HD-BET**: aplicado a IXI, NKI Rockland y BTC (datasets que
   pueden venir con cráneo). BraTS y UPENN ya vienen procesados sin cráneo. La
   motivación (README): evitar que el modelo aprenda «cráneo = no tumor».
3. **Reorientación a RAS** (`reorient_to_ras`, vía `nibabel`): ejes anatómicos
   consistentes.
4. **Resampling a 1 mm³ isótropo** (`resample_volume`, `scipy.ndimage.zoom`,
   `order=1`): escala comparable entre datasets.
5. **Crop/pad a forma fija `(192, 224, 192)`** (`crop_or_pad`): tensores de tamaño
   homogéneo.
6. **Normalización z-score sobre máscara de tejido** (`normalize_intensity`): media
   y desviación calculadas **solo sobre vóxeles de tejido**. El umbral de la máscara
   se determina por **Otsu** sobre los vóxeles positivos, con *fallback* al
   percentil 30 si `scikit-image` falla, y salvaguardas si la máscara queda con
   menos de 1000 vóxeles.
7. **Emparejado T1/T2**: en `preprocess_paired_volumes`, T2 se **remuestrea a la
   rejilla de T1** (`resample_from_to`) antes de normalizar, garantizando que ambos
   canales comparten geometría (importante para una CNN multicanal).
8. **Guardado `.npz`** (`save_sample_npz`): cada muestra contiene
   `t1, t2, label, dataset, subject_id, source_t1, source_t2`.

**El paso Otsu en detalle (decisión técnica documentada).** Según
`docs/justificacion_cambios.md` (paso 8), UPENN-GBM (procesado con CaPTk) traía un
«shell» de fondo gris no-cero: ~66 % de vóxeles ≠0 frente a ~16–21 % en
BraTS/IXI/NKI. La normalización antigua (umbral = percentil 5 de positivos) metía
ese shell en la máscara y dejaba a UPENN con una escala distinta. La corrección con
Otsu redujo UPENN a ~22–28 % de vóxeles no-cero, comparable al resto. **Hecho
observado**: tras la corrección, los scores medios de UPENN (0.999) y BraTS (0.986)
se igualan (`docs/justificacion_cambios.md`). **Interpretación crítica**: esto
elimina el *domain shift* **dentro** de la clase positiva, pero **no** el confound
estructural entre clases (positivos y negativos siguen procediendo de fuentes
disjuntas).

**Estructura de salida** (hecho observado: `data/processed/preprocessing_summary.json`):

- `data/processed/positives/` y `data/processed/negatives/` — pool principal, T1+T2,
  2267 muestras (1167 pos / 1100 neg).
- `data/processed_btc/positives/` y `.../negatives/` — dataset intra-dominio BTC,
  **T1-only**, 36 muestras (25 pos / 11 neg).

La separación en dos árboles (`processed` vs `processed_btc`) responde a que BTC es
T1-only y se procesa con un *script* propio (`preprocess_btc.py`) y se usa solo en
los *scripts* de auditoría (`src/audit/btc_*.py`), sin tocar el pipeline 2-canales.

**Papel de cada script de preprocesado:**

- `base_preprocessing.py`: funciones comunes y el formato `.npz`; define
  `PreprocessingConfig(target_shape=(192,224,192), target_spacing=(1,1,1))`.
- `preprocess_brats.py`: BraTS → `positives/`, label 1.
- `preprocess_ixi.py`: IXI → `negatives/`, label 0; incluye skull-strip HD-BET
  (`--skull-strip`).
- `preprocess_upenn.py`: UPENN → `positives/`, label 1; filtra pacientes con T1/T2.
- `preprocess_nki_rockland.py`: NKI → `negatives/`, label 0; skull-strip HD-BET;
  espera BIDS.
- `preprocess_btc.py`: BTC (ds001226) → `processed_btc/`, T1-only; `sub-CON*` → 0,
  `sub-PAT*` → 1.
- `download_nki_rockland.py`: descarga desde el bucket S3 público de FCP-INDI
  filtrando sujetos con **T1w y T2w en la misma sesión** (usa `boto3`).
- `dataset/summarize_processed.py`: reconstruye `preprocessing_summary.json`.

**Qué se versiona y qué no (hecho observado: `.gitignore`).** Están ignorados
`/data/` (todos los datos raw y procesados, `splits.json`, summary) y `outputs/`
(checkpoints, evaluaciones, plots, logs), además de `.claude/` y `CLAUDE.md`. **Sí**
se versiona `docs/audit/` (JSON, CSV, figuras), de modo que los resultados de la
auditoría son trazables sin necesidad de los datos pesados. La razón de ignorar
`data/` y `outputs/` es práctica y ética: son archivos pesados (volúmenes médicos,
checkpoints) que no deben subirse a un repositorio de código y que, en el caso de
los datos clínicos, pueden estar sujetos a licencias de uso.

---

## 7. Datasets y composición del problema

### BraTS 2021
- **Papel**: clase positiva (tumor), entrenamiento. **n ≈ 580** procesados
  (`preprocessing_summary.json`: 580).
- **Tipo**: glioma; volúmenes ya skull-stripped de origen (challenge).
- **Riesgo de dominio**: firma de adquisición/preprocesado de tipo *challenge*,
  compartida con UPENN.

### UPENN-GBM
- **Papel**: clase positiva (tumor), **n ≈ 587** (`summary`: 587).
- **Historia metodológica (hecho observado: `docs/evolucion_respecto_primera_entrega.md` §1.2)**:
  originalmente previsto como **conjunto de evaluación externa** cross-dataset; se
  decidió **incorporarlo al entrenamiento** con label=1.
- **Consecuencia**: el repositorio **pierde su único conjunto de validación externa**.
  Se reformula el objetivo hacia «robustez multi-fuente». Esta es una de las
  decisiones más relevantes del proyecto y debe declararse como desviación
  consciente respecto a la primera entrega.

### IXI
- **Papel**: clase negativa (sano), **n ≈ 577** (`summary`: 577).
- **Necesidad de skull-stripping**: IXI venía con cráneo; sin homogeneizar, el
  modelo podría haber aprendido «cráneo = sano». Se aplicó HD-BET
  (`preprocess_ixi.py`).
- **Riesgo de dominio**: cohorte de investigación con protocolos heterogéneos.

### NKI Rockland
- **Papel**: clase negativa (sano) **adicional**, **n ≈ 523** (`summary`: 523).
- **Motivación (hecho observado: `evolucion...` §1.3)**: reequilibrar las clases
  tras incorporar UPENN (que había dejado ~2:1 a favor de positivos). Con NKI, el
  balance final es 1167:1100 (≈ 1.06:1).
- **Descarga**: `download_nki_rockland.py` filtra sujetos con T1w+T2w en la misma
  sesión BIDS.

### BTC_preop / OpenNeuro ds001226
- **Papel**: dataset **intra-dominio honesto**, único con **ambas clases dentro del
  mismo dominio** (mismo escáner, Ghent University Hospital).
- **Composición (hecho observado: `participants.tsv` y `processed_btc/`)**: 36
  sujetos, **25 pacientes con tumor** (`sub-PAT*`: meningiomas, gliomas, etc.) y
  **11 controles sanos** (`sub-CON*`).
- **Modalidad**: **T1-only** (ds001226 no incluye T2 estructural).
- **Valor metodológico**: permite comprobar qué ocurre **al eliminar el confound de
  dominio por construcción**. Si el rendimiento cae al azar, queda demostrado que el
  AUC ≈ 1.0 previo era confound.
- **Limitaciones**: n pequeño (IC anchos) y una sola modalidad.

**Tabla resumen:**

| Dataset | Clase | n aprox. | Modalidades | Script asociado | Papel | Riesgo metodológico |
|---|---|---|---|---|---|---|
| BraTS 2021 | Positivo (1) | 580 | T1+T2 | `preprocess_brats.py` | Entrenamiento | Firma de challenge compartida con UPENN |
| UPENN-GBM | Positivo (1) | 587 | T1+T2 | `preprocess_upenn.py` | Entrenamiento (era test externo) | Pérdida de validación externa; shell CaPTk (mitigado con Otsu) |
| IXI | Negativo (0) | 577 | T1+T2 | `preprocess_ixi.py` | Entrenamiento | Requería skull-strip; protocolo heterogéneo |
| NKI Rockland | Negativo (0) | 523 | T1+T2 | `preprocess_nki_rockland.py` | Entrenamiento (reequilibrio) | Otra firma de dominio en el lado negativo |
| BTC_preop (ds001226) | Mixto (25/11) | 36 | T1-only | `preprocess_btc.py` | Validación intra-dominio | n pequeño; T1-only |

**El hecho metodológico central (hecho observado: `audit_leakage.json`):**

```
label = 1  ⟺  dataset ∈ {brats, upenn}
label = 0  ⟺  dataset ∈ {ixi, nki_rockland}
```

es decir, **correlación clase↔dataset del 100 %**. Ningún dataset del pool
principal aporta ambas clases. Por construcción, cualquier modelo que distinga
«BraTS/UPENN vs IXI/NKI» obtiene etiqueta perfecta sin necesidad de detectar tumor.

---

## 8. Modelo CNN 3D

Definición en `src/models/cnn3d.py` (hecho observado).

**`BrainTumorCNN3D`** — CNN 3D compacta para clasificación binaria:

- **Entrada**: `(B, in_channels, 128, 160, 128)`; `in_channels = 2` para el pool
  principal (canales T1 y T2), `in_channels = 1` para BTC (reutilizado en
  `btc_cnn_kfold.py` y `embeddings_tsne.py`).
- **Bloques convolucionales**: 4 × `ConvBlock3D`, cada uno con dos `Conv3d` (k=3,
  padding=1, sin bias) + `GroupNorm` + `ReLU`, y `Dropout3d` opcional. Canales
  `12 → 24 → 48 → 96` (`base_channels = 12`). Entre los tres primeros bloques hay
  `MaxPool3d(2)`; tras el cuarto, `AdaptiveAvgPool3d(1)`.
- **Cabeza clasificadora**: `Flatten → Linear(96→96) → ReLU → Dropout(0.25) →
  Linear(96→1)`. El `forward` aplica `.squeeze(1)` → **un único logit por muestra**.
- **Salida**: logit; la probabilidad de tumor es `sigmoid(logit)` (aplicado en
  evaluación e inferencia).

**GroupNorm en lugar de BatchNorm (decisión técnica documentada).** La función
`_norm()` usa `GroupNorm` (8 grupos). El motivo, explícito en el docstring y en
`docs/justificacion_cambios.md` (paso 4): con `batch_size = 1`, BatchNorm calcula
estadísticas sobre una sola muestra (ruido enorme) y sus *running stats* no
convergen, generando un *gap* masivo entre train y val. GroupNorm no depende del
tamaño de batch. Este fue, según el documento, el cambio de mayor impacto para
estabilizar el entrenamiento.

**Número de parámetros (hecho observado, verificado instanciando el modelo):**
**504 553 parámetros entrenables** en la configuración de 2 canales
(`in_channels=2`) y **504 229** en la variante de 1 canal (`in_channels=1`, usada
en BTC). Nota: el documento `docs/evolucion_respecto_primera_entrega.md` estimaba
«~200K», cifra **incorrecta** y corregida en aquel documento.

**Limitaciones del modelo:**

- `batch_size = 1` por restricciones de memoria GPU (forzó GroupNorm).
- Entrenado **desde cero** (sin transfer learning): justificable por la diferencia
  de dominio respecto a ImageNet, pero exigente en datos.
- **Sensibilidad a dominio**: como demuestra la auditoría, el modelo aprende firmas
  de dominio. No es un defecto del modelo en sí, sino de la composición de datos,
  pero el modelo carece de cualquier mecanismo de invariancia de dominio.
- **Crop volumétrico** a 128×160×128: puede excluir lesiones periféricas si el
  tumor no está centrado.
- **Sin validación externa real** disponible (consecuencia de mover UPENN a train).

---

## 9. Entrenamiento

Implementado en `src/training/train_3d.py`, configurado por `configs/train_3d.yaml`
(hechos observados).

**Datos y splits:**
- `make_loaders` crea/carga los splits. Con `recreate_splits: true` (config
  principal), `create_splits` se ejecuta en cada *run*.
- **Estratificación por `(dataset, label)`** y **agrupación por `subject_id`**
  (`dataset_3d.py::create_splits`): evita que un re-split deje un dataset
  sub/sobre-representado y previene leakage de sujeto en caso de sesiones múltiples.
- Ratios 70/15/15. **Composición real (hecho observado: `docs/justificacion_cambios.md`)**:
  train 1587 (817 pos / 770 neg), val 340 (175/165), test 340 (175/165).
- Train usa `augment=True`; val/test no. Con `random_crop_train: false`, el train
  usa **center crop** igual que val/test (para eliminar el *shift* espacial
  train↔val).

**Augmentation (`BrainMRI3DDataset._augment`):** flips en los 3 ejes espaciales +
**gamma (0.8–1.25) y ruido gaussiano (σ=0.03) aplicados solo sobre vóxeles ≠0**
(máscara de cerebro), preservando el signo del z-score. Objetivo declarado:
forzar a usar contraste relativo y no el brillo absoluto característico de cada
dataset.

**Optimización:**
- Pérdida: `BCEWithLogitsLoss` con `pos_weight` configurable (`"auto"` =
  n_neg/n_pos; en la config principal fijado a `1.0` porque las clases están
  equilibradas).
- Optimizador: **AdamW** (`lr=5e-5`, `weight_decay=1e-3`).
- Scheduler: **cosine** (`CosineAnnealingLR`).
- AMP: **desactivado** (`amp: false`) — float16 con batch=1 + GroupNorm amplificaba
  inestabilidades numéricas.

**Selección de checkpoint y early stopping (hecho observado, l. 376–407):** se
guarda `best.pt` por **`balanced_accuracy` (= (sen+spe)/2) exigiendo
`sensitivity ≥ min_sensitivity_for_save` (0.80)**. Early stopping tras `patience`
(10) épocas sin mejora. **El test se evalúa una sola vez al final** con el mejor
checkpoint (no se usa para decidir).

**Métricas durante entrenamiento:** loss, accuracy, AUC (implementación propia por
rangos, equivalente a Mann-Whitney; **no usa sklearn**), sensibilidad,
especificidad y balanced_accuracy, tanto en train como en val.

**Salida:** `outputs/checkpoints/<timestamp>/{best.pt, history.json, curves.png}`.
`best.pt` almacena `model_state_dict`, `config`, `epoch` y métricas de val.

**Cambios introducidos para corregir el estado inicial (hecho observado:
`docs/justificacion_cambios.md`, «los 8 pasos»).** El estado previo tenía
`specificity = 0` en test. Las correcciones:

| Paso | Cambio | Motivo |
|---|---|---|
| 1 | Estratificar por (dataset, label, subject_id) | Robustez del split a futuro |
| 2 | Evaluación por dataset | Diagnóstico de dónde fallan los negativos |
| 3 | Umbral elegido en validación | `specificity=0` era por umbral 0.5 mal calibrado |
| 4 | **BatchNorm → GroupNorm** | Estabilidad con batch=1 (cambio de mayor impacto) |
| 5 | Checkpoint por balanced_acc + sen mínima | Evitar guardar modelos triviales con spe=0 |
| 6 | `pos_weight=1`, `lr↓` (5e-5), `wd↑` (1e-3), cosine, AMP off | Regularizar y estabilizar |
| 7 | Center crop en train + augmentation de intensidad | Reducir shift train/val y dependencia del brillo |
| 8 | Re-normalizar UPENN con **Otsu** | Eliminar el shell gris CaPTk (domain shift intra-clase) |

**Resultado tras los 8 pasos (hecho observado: `docs/justificacion_cambios.md`).**
El run `20260527_152619` superó los tres objetivos numéricos del plan
(sen ≥ 0.85, spe ≥ 0.60, score_mean UPENN ≈ BraTS). **Pero** el propio documento
añade la sección «Limitación conocida — confound estructural dataset↔clase»: las
métricas casi perfectas no pueden interpretarse como detección de tumor. Es decir,
**el éxito del plan de estabilización fue, paradójicamente, lo que destapó el
problema de fondo**: un modelo bien entrenado sobre datos confundidos aprende el
confound a la perfección.

---

## 10. Evaluación

Implementada en `src/evaluation/evaluate_3d.py` y `threshold_analysis.py` (hechos
observados).

**Métricas (`evaluate_3d.py`):** AUC (por rangos), PR-AUC (`average_precision`), y
a un umbral dado: tp/fp/tn/fn, accuracy, sensibilidad, especificidad, precision,
NPV, F1, balanced_accuracy, matriz de confusión. Además, **métricas por dataset**
(con `auc = NaN` cuando el dataset es mono-clase). Salida:
`outputs/evaluation/cnn3d_<split>_results.json` + `_predictions.{json,csv}`.

**Análisis de umbral (`threshold_analysis.py`):** calcula el umbral **en validación**
(Youden / balanced_accuracy y «sensibilidad mínima maximizando especificidad») y lo
**aplica a test**, con histogramas globales y por dataset. Regla metodológica
correcta: el umbral se elige en val, nunca en test.

**Resultado del run confundido (hecho observado: `outputs/evaluation/cnn3d_test_results.json`):**

| Métrica | Valor |
|---|---|
| n (test) | 340 (175 pos / 165 neg) |
| AUC | **0.999965** |
| PR-AUC | 0.999968 |
| Accuracy | 0.99706 |
| Sensibilidad @0.5 | 0.99429 |
| Especificidad @0.5 | 1.0 |
| Confusión | TP=174, FP=0, TN=165, FN=1 |

El único falso negativo es `BraTS2021_00736` (score 0.092), un *outlier* dentro de
BraTS (`docs/justificacion_cambios.md`).

**Scores por dataset (hecho observado, mismo JSON):**

| Dataset | Clase | score medio | std | AUC intra-dataset |
|---|---|---|---|---|
| upenn | tumor | 0.9995 | ±0.0005 | NaN (mono-clase) |
| brats | tumor | 0.9860 | ±0.098 | NaN |
| ixi | sano | 0.0070 | ±0.018 | NaN |
| nki_rockland | sano | 0.0008 | ±0.0005 | NaN |

**Por qué las métricas globales eran engañosas (interpretación sustentada en los
datos):**

1. **AUC ≈ 1.0** en un problema de detección de tumor real es implausible.
2. **Varianza intra-dataset casi nula** (UPENN ±0.0005, NKI ±0.0005): el modelo
   asigna prácticamente el mismo score a *todas* las imágenes de un dataset,
   independientemente del contenido. Un detector de lesión real mostraría
   variabilidad (tumores sutiles vs evidentes).
3. **AUC intra-dataset no calculable**: como cada dataset es mono-clase, no hay
   forma de medir AUC dentro de un dataset. Esa imposibilidad es, en sí misma, un
   síntoma del confound.
4. **Separación de scores por dataset, no por patología**: los scores se agrupan en
   dos modos extremos (≈1 para BraTS/UPENN, ≈0 para IXI/NKI), coincidiendo con el
   dataset de origen.

---

## 11. Punto crítico: resultados sospechosos

Esta sección es el eje del proyecto.

**Qué llamó la atención.** Un modelo de detección de tumor en RM cerebral que
alcanza AUC ≈ 0.99997, sensibilidad ≈ 0.994 y especificidad = 1.0, con un solo
falso negativo y ningún falso positivo sobre 340 estudios de test. En la literatura
clínica, la detección de tumor cerebral —incluso para sistemas maduros— no produce
separaciones perfectas, porque hay tumores sutiles, artefactos, variabilidad
anatómica y casos límite.

**Por qué era sospechoso, en términos técnicos.** Tres señales convergentes:

- La **magnitud** del rendimiento (AUC al techo).
- La **varianza intra-dataset casi nula** de los scores, impropia de un detector de
  lesión.
- La **imposibilidad de medir AUC intra-dataset** (cada dataset mono-clase), que
  delata que la única partición sobre la que el modelo se mide es, de hecho, la
  partición por dataset.

**Hipótesis formulada.** El rendimiento podía deberse a:

- *Leakage* trivial: duplicados de un mismo estudio en train y test, o el mismo
  sujeto en ambas particiones.
- **Confound de dominio**: el modelo aprende a reconocer **de qué dataset/escáner**
  proviene la imagen, no si hay tumor. Posibles señales discriminantes no clínicas:
  intensidad media, percentiles, fracción de vóxeles no-cero (firma del
  skull-stripping), resolución, formato, patrón de fondo, protocolo de adquisición,
  centro.

**Honestidad técnica.** El proyecto no oculta este punto ni lo presenta como nota
al pie. Lo convierte en el objeto de estudio: se diseñó una batería de pruebas
(secciones §12 a §17) cuyo objetivo era **falsar** la hipótesis de detección real
de tumor. El resultado de esas pruebas, como se verá, confirma el confound y
descarta las fugas triviales.

---

## 12. Auditoría de leakage y tiny baseline

Implementada en `src/audit/audit_leakage.py` (hecho observado).

**Objetivo.** Comprobar si la etiqueta y/o el dataset de origen son triviales de
predecir a partir de **features baratas y no clínicas**, y detectar duplicados.

**Features (16 por volumen):** fracción de vóxeles no-cero, media, desviación
típica y percentiles 1/25/50/75/99 de T1 y de T2 (calculados sobre tejido). Ninguna
es clínica: son estadísticos globales de intensidad.

**Pruebas implementadas:**
- **B — Tiny baseline**: Logistic Regression y Random Forest sobre las 16 features
  → predecir la etiqueta.
- **C — Clasificador de dataset**: Random Forest sobre las 16 features → predecir el
  dataset de origen (4 clases).
- **Duplicados**: sha1 de los arrays `(t1, t2)` por fichero, comprobando
  solapamiento entre splits.
- **`label_equals_dataset`**: comprueba qué etiquetas aparecen en cada dataset.

(Nota técnica, hecho observado: por defecto el train se submuestrea a 150
muestras/dataset, `--per-dataset-train 150`, de ahí que `n_train = 600` en el JSON;
el test se usa completo, `n_test = 340`.)

**Resultados (hecho observado: `docs/audit/audit_leakage.json`):**

| Prueba | Resultado |
|---|---|
| Tiny baseline LogReg (test AUC) | **1.0000** (acc 0.9941) |
| Tiny baseline Random Forest (test AUC) | 0.9989 (acc 0.9882) |
| Clasificador de dataset (RF, 4 clases) | acc **0.9853** (azar 0.25) |
| Grupos de duplicados exactos | **0** |
| Duplicados cruzando splits | **0** |
| `label_equals_dataset` | brats:[1], upenn:[1], ixi:[0], nki_rockland:[0] |

Features más informativas (importancia RF): `nz_frac_t1` (0.17), `p75_t1` (0.17),
`p25_t2` (0.13), `nz_frac_t2` (0.10) — es decir, la **fracción de vóxeles no-cero**
(firma directa del skull-stripping/preprocesado) y percentiles de intensidad.

**Conclusión (interpretación sustentada):**

- El problema **no** es un duplicado ni un *bug* de partición (ambos descartados con
  evidencia: 0 duplicados).
- El problema es **estructural**: dataset y etiqueta están acoplados, y la etiqueta
  es **decodificable sin red neuronal, sin información espacial y sin ninguna feature
  clínica**. Si un modelo lineal sobre estadísticos de brillo separa las clases
  perfectamente, el AUC ≈ 1.0 de la CNN no requiere detección de tumor.

---

## 13. Auditoría de splits

Implementada en `src/data/audit_splits.py`, con la lógica de partición en
`src/data/dataset_3d.py::create_splits` (hechos observados).

**Qué verifica.** Composición por `(split, dataset, label)`, **solapamiento de
`subject_id`** entre train/val/test, y totales por split.

**Lógica de la partición:**
- Se indexa por `subject_id` y los ficheros de un mismo sujeto caen siempre en el
  mismo split (evita leakage de sesiones múltiples, p. ej. NKI `-BAS2/-BAS3`).
- Estratificación conjunta por `(dataset, label)`.
- Determinista con `seed = 42`.

**Resultado (hecho observado: documentado en `docs/auditoria_resultados_sospechosos.md`
y `docs/justificacion_cambios.md`):** `train ∩ val = 0`, `train ∩ test = 0`,
`val ∩ test = 0`. Además, `n_samples == n_subjects` en cada (split, dataset, label):
en los datos actuales cada sujeto tiene un único fichero.

**Interpretación.** Esto **descarta una fuga trivial por solapamiento de sujeto**,
pero **no descarta el confound dominio↔clase**: la partición está bien hecha; el
problema no es *cómo* se reparten los ficheros, sino *qué representa la etiqueta*.
Un split impecable sobre datos confundidos sigue produciendo métricas confundidas.

---

## 14. Validación LODO (Leave-One-Domain-Out)

**Concepto.** *Leave-One-Domain-Out* entrena con un subconjunto de dominios y
evalúa en dominios **no vistos**, midiendo la transferencia fuera de dominio. Si el
modelo detecta tumor genuino, debería transferir; si detecta firma de dominio, la
transferencia fallará.

**Implementación (hechos observados):**
- `src/audit/make_lodo_splits.py`: genera `data/splits.json` para las
  configuraciones A/B/C/D (escribe también un registro permanente
  `data/splits_lodo_<C>.json`). El split de test toma **todos** los sujetos de los
  datasets held-out.
- `configs/train_lodo.yaml`: clon de la config principal con `recreate_splits:
  false` (usa el split que escribe `make_lodo_splits.py`) y `checkpoint_dir`
  propio. Resto idéntico, para comparación *apples-to-apples*.
- `scripts/run_lodo.sh <A|B|C|D>`: genera el split, entrena con `train_lodo.yaml`,
  localiza el run y evalúa en test → `outputs/evaluation/lodo_<C>/`.
- `src/audit/audit_lodo.py`: replica el LODO con el *tiny baseline* (lineal/árbol)
  sobre las features de intensidad, incluyendo una referencia *random-mix*.

**Configuraciones (hecho observado: `make_lodo_splits.py`):**
- A: train {brats, ixi} → test {upenn, nki_rockland}
- B: train {upenn, nki_rockland} → test {brats, ixi}
- C: train {brats, nki_rockland} → test {upenn, ixi}
- D: train {upenn, ixi} → test {brats, nki_rockland}

**Resultados CNN (hecho observado: `outputs/evaluation/lodo_{A,B}/cnn3d_test_results.json`
y `docs/audit/resumen_consolidado.json`):**

| Config | AUC | Sensibilidad | Especificidad |
|---|---|---|---|
| LODO A | **0.6236** | 0.676 | 0.463 |
| LODO B | **0.2012** | 0.010 | 0.965 |

(C y D **no se ejecutaron con la CNN**; sí con el tiny baseline.)

**Resultados tiny baseline LODO (hecho observado: `docs/audit/audit_lodo.json`):**

| Config | LogReg AUC | RF AUC |
|---|---|---|
| A: BraTS+IXI → UPENN+NKI | 0.9952 | 0.9889 |
| B: UPENN+NKI → BraTS+IXI | 0.3184 | 0.6957 |
| C: BraTS+NKI → UPENN+IXI | 0.0412 | 0.0338 |
| D: UPENN+IXI → BraTS+NKI | 0.0469 | 0.2752 |
| Referencia random-mix 70/30 | 1.0000 | 0.9991 |

**Interpretación.** La transferencia es **asimétrica y caótica**, con varios AUC muy
por debajo de 0.5 (regla de decisión **invertida**: lo que el modelo aprendió como
«tumor» en un par de dominios apunta a «sano» en otro). En LODO B, la CNN colapsa
(sensibilidad 0.010: predice casi todo como sano). Este comportamiento es
**incompatible con un detector de lesión genuino** y **consistente con un detector
de firma de adquisición/dominio**. La referencia random-mix recupera AUC ≈ 1.0,
confirmando que el atajo trivial existe cuando train y test comparten la mezcla de
dominios.

---

## 15. Validación intra-dominio BTC

Implementada en `preprocess_btc.py`, `src/audit/btc_tiny_baseline.py`,
`src/audit/btc_cnn_kfold.py` y orquestada por `scripts/run_btc_chain.sh` (hechos
observados).

**Por qué BTC es importante.** Es el único dataset con **ambas clases dentro del
mismo dominio** (mismo escáner). Por construcción, **elimina el confound
dominio↔clase**: aquí tumor y sano no se distinguen por su procedencia. Es, por
tanto, la **única medición metodológicamente honesta** de capacidad de detección
disponible en el repositorio.

**Diseño (hechos observados):**
- `preprocess_btc.py`: T1-only; `sub-CON*` → label 0, `sub-PAT*` → label 1.
- `btc_tiny_baseline.py`: tiny baseline (8 features de T1) con **k-fold
  estratificado** + IC95 % por bootstrap (2000 resamples).
- `btc_cnn_kfold.py`: **CNN 3D 1-canal** (misma arquitectura, `in_channels=1`),
  k-fold estratificado por sujeto, **20 épocas fijas por fold sin early stopping**
  (para no contaminar la selección con el fold de test), predicciones agregadas +
  IC95 % bootstrap.
- `run_btc_chain.sh`: preprocesa, verifica, ejecuta tiny + CNN k-fold y consolida,
  con *timeouts*.

**Limitaciones declaradas:** n = 36 (IC anchos) y **T1-only** (ds001226 no trae T2).

**Resultados (hecho observado: `docs/audit/btc_intradomain_tinybaseline.json`,
`outputs/evaluation/btc_intradomain/cnn_kfold_results.json`,
`docs/audit/resumen_consolidado.json`):**

| Modelo | AUC | IC95 % | Notas |
|---|---|---|---|
| Tiny baseline LogReg | 0.5491 | [0.319, 0.788] | T1-only |
| Tiny baseline RF | 0.4055 | [0.215, 0.616] | T1-only |
| **CNN 3D 1-canal** | **0.4036** | **[0.213, 0.623]** | sen 0.60 / spe 0.364 @0.5 |

**Conclusión.** Cuando desaparece la separación dominio↔clase, el rendimiento de
**todos** los modelos (lineal y profundo) cae a niveles **indistinguibles del azar**
(los IC95 % cruzan 0.5 con holgura). Esto apoya directamente la hipótesis de que el
AUC ≈ 1.0 del pool multi-fuente **no era detección clínica de tumor**.

**Matiz honesto (interpretación).** Con n = 36 y T1-only **no puede afirmarse «el
modelo no detecta tumor»** en sentido absoluto; lo que puede afirmarse es que «con
estos datos no se observa señal por encima del azar», lo cual es suficiente para
concluir que el rendimiento previo era confound.

---

## 16. Análisis de embeddings

Implementado en `src/audit/embeddings_tsne.py` (E1) y `embeddings_intraclass.py`
(hechos observados; este análisis es el más reciente y aún no estaba commiteado).

**Procedimiento:**
- Se extrae, para los 2267 volúmenes, el **vector latente de 96 dimensiones**
  (salida de `model.features` tras `AdaptiveAvgPool3d`, antes del clasificador) del
  checkpoint confundido (`embeddings_tsne.py`).
- Se proyecta a 2D con **PCA** y **t-SNE**, coloreando por dataset y por etiqueta.
- Se cuantifica la compacidad de los clusters con el coeficiente **silhouette** por
  etiqueta y por dataset (`embeddings_silhouette.json`).
- `embeddings_intraclass.py` mide la separabilidad de datasets **dentro de cada
  clase** (silhouette + LogReg con validación cruzada 5-fold) y entrena un
  clasificador de dataset (4 clases) sobre los embeddings.

**Resultados (hecho observado: `embeddings_silhouette.json`,
`embeddings_intraclass.json`):**

| Medida | Valor |
|---|---|
| Silhouette por etiqueta | 0.754 |
| Silhouette por dataset (global) | 0.366 |
| IXI vs NKI (ambos **sanos**): silhouette / LogReg CV-AUC | 0.561 / **0.998** |
| BraTS vs UPENN (ambos **tumor**): silhouette / LogReg CV-AUC | 0.186 / **0.991** |
| Clasificador de dataset (4 clases) desde embeddings | acc **0.982** (azar 0.25) |

**Interpretación.** La separación por clase (silhouette 0.754) es alta, pero como
`clase ≡ dataset` **no permite distinguir** si la red codifica «tumor» o
«procedencia». El análisis intra-clase resuelve esa ambigüedad: **dados dos sujetos
igualmente sanos, uno de IXI y otro de NKI, el espacio latente los separa con
AUC = 0.998** — una capacidad que nada tiene que ver con la presencia de tumor. La
representación interna funciona, en la práctica, como un **identificador del centro
de adquisición**. Esto constituye evidencia directa, a nivel de representación
interna, del confound de dominio.

**Nota de honestidad metodológica (hecho observado: presente en el propio borrador).**
Se reportan tanto el silhouette global (0.366, diluido porque mezcla separación
inter-clase e intra-clase) como el análisis intra-clase (limpio). El segundo no
sustituye al primero: lo explica. La medición intra-clase es la pregunta
correctamente planteada (¿se separan cohortes de la misma clase?), no una métrica
seleccionada *a posteriori*.

---

## 17. Grad-CAM e interpretabilidad

Implementado en `src/audit/gradcam_3d.py`; figuras en
`docs/audit/figures/gradcam/{confound,lodo_A,lodo_B}/` (hechos observados).

**Qué busca Grad-CAM.** Producir mapas de activación que indiquen **qué regiones del
volumen** influyen más en la predicción. La hipótesis pre-registrada en el docstring
del script: un modelo confundido debería concentrar la atención en
bordes/cráneo/fondo (firma de dominio), no en la lesión; un modelo con señal real
la concentraría en el tumor.

**Cómo ayuda.** Permite una comprobación **visual** y cualitativa: si el mapa cae
sistemáticamente fuera de regiones anatómicas relevantes (en bordes, fondo o
patrones de intensidad globales), refuerza la conclusión de que el modelo no atiende
a la patología.

**Limitaciones de Grad-CAM (interpretación, importante para el tribunal):**
- Es una técnica **cualitativa** y sensible a la capa elegida y a la resolución del
  mapa (que aquí, tras varios *poolings* y `AdaptiveAvgPool3d`, es muy baja).
- No constituye prueba cuantitativa por sí sola.
- Por ello **debe interpretarse junto** con las evidencias cuantitativas (tiny
  baseline, LODO, intra-dominio, embeddings), no de forma aislada.

Se dispone de figuras para el run confundido y para LODO A/B. **No verificado** en
esta reconstrucción el contenido visual concreto de cada PNG (son imágenes); se
documenta su existencia y propósito.

---

## 18. Figuras y resultados consolidados

**Scripts de figuras (hechos observados):**
- `src/audit/make_plots.py`: `roc_curves.png`, `auc_summary.png`,
  `score_hist_confound.png`, `score_hist_lodo.png`, `btc_kfold_bars.png` (a partir
  de los JSON y los `predictions.csv`).
- `src/audit/make_extra_figures.py`: `intensity_by_dataset.png` (E2, boxplots de
  intensidad por dataset desde `audit_features.csv`) y `confusion_matrices.png`
  (E3, matrices por experimento).
- `src/audit/embeddings_tsne.py`: `embeddings_pca.png`, `embeddings_tsne.png`.
- `src/audit/consolidate_results.py`: une todos los JSON en la tabla maestra y
  produce `docs/audit/resumen_consolidado.{md,json}`.

**Inventario de figuras y mensaje que transmite cada una (hecho observado:
`docs/audit/borrador_memoria.md`):**

| Figura | Mensaje |
|---|---|
| `auc_summary.png` | El AUC se desploma de ~1.0 a azar al eliminar el confound (barras con IC95 %) |
| `roc_curves.png` | Confounded pega al techo; LODO se aleja; intra-dominio cae a la diagonal |
| `score_hist_confound.png` | Scores bimodales extremos por dataset, no por contenido |
| `score_hist_lodo.png` | En cross-dataset los scores se descolocan |
| `confusion_matrices.png` | Diagonal perfecta → colapso/inversión (LODO) → dispersión (intra-dominio) |
| `btc_kfold_bars.png` | AUC por fold disperso en torno al azar, IC ancho |
| `intensity_by_dataset.png` | El confound existe ya en los píxeles crudos |
| `embeddings_pca.png` / `embeddings_tsne.png` | El latente agrupa por procedencia |
| `gradcam/*` | A qué atiende el modelo (bordes/fondo) |

**Figura principal recomendada (interpretación).** Como **figura central de la
memoria** se recomienda `auc_summary.png`, porque condensa en una sola imagen la
tesis completa: el AUC cae de ≈1.0 (confounded) a azar (intra-dominio), con los
intervalos de confianza que muestran la incertidumbre. Como **figura de apoyo más
elocuente**, `embeddings_tsne.png`, porque visualiza que el espacio latente agrupa
por procedencia y que IXI y NKI (ambos sanos) no se fusionan. Una selección mínima
de cuatro figuras —`auc_summary`, `embeddings_tsne`, `confusion_matrices` y un
`gradcam/confound`— cuenta la historia completa.

---

## 19. Documentación añadida

| Documento | Qué aporta | Estado |
|---|---|---|
| `docs/auditoria_resultados_sospechosos.md` | Informe de auditoría completo (2026-05-28): veredicto «no defendible como detección de tumor», pruebas del confound, clasificación de problemas por gravedad, propuestas (a/b/c) | **Vigente** |
| `docs/justificacion_cambios.md` | El porqué de los 8 pasos de estabilización + resultados del run `20260527_152619` + limitación del confound | **Vigente** |
| `docs/evolucion_respecto_primera_entrega.md` | Comparación con la 1ª entrega; decisión de mover UPENN a train; reformulación a robustez multi-fuente | **Parcialmente desactualizado** |
| `docs/audit/borrador_memoria.md` | Columna vertebral de Resultados+Discusión: tabla maestra de 10 filas, narrativa en pasos, frases listas, inventario de figuras, preguntas del tribunal, trazabilidad | **Borrador en evolución** (bloque embeddings recién añadido) |
| `docs/audit/resumen_consolidado.{md,json}` | Síntesis automática de todos los resultados (generada por script) | **Vigente** |

**Advertencias sobre documentación desactualizada (hecho observado, importante para
no usar fuentes obsoletas como verdad final):**

- `docs/evolucion_respecto_primera_entrega.md` afirma que «la CNN 3D **no ha sido
  entrenada todavía**» y que «los scripts de evaluación del modelo 3D
  (`evaluate_3d.py`) **no existen todavía**». **Ambas afirmaciones eran ciertas
  cuando se redactó el documento, pero ya no lo son**: la CNN se entrenó (run
  `20260527_152619`), `evaluate_3d.py` existe y se ha ejecutado.
- El mismo documento describe la composición de negativos como «IXI únicamente
  (577)»; el estado final incluye **IXI + NKI Rockland (1100 negativos)**.
- `src/preprocessing/README.md` también refleja un estado anterior («negatives →
  IXI») y describe el formato `.npz` con `dataset ∈ {brats, upenn, ixi}` sin NKI ni
  BTC, y menciona una orientación «2.5D» que no corresponde al modelo 3D final.
- `docs/evolucion...` referencia un `docs/revision_tecnica.md` que **no existe** en
  el repositorio (el archivo equivalente es
  `docs/auditoria_resultados_sospechosos.md`).

Recomendación: al redactar la memoria, tomar como fuentes de verdad
`docs/auditoria_resultados_sospechosos.md`, `docs/justificacion_cambios.md`,
`docs/audit/borrador_memoria.md` y los JSON de resultados.

**Estado de corrección (aplicado).** Los documentos obsoletos ya se han marcado y
corregido: `docs/evolucion_respecto_primera_entrega.md` incluye ahora un **banner
de actualización** al inicio que lista cada afirmación desfasada y remite a las
fuentes vigentes (y se corrigió el dato erróneo de parámetros «~200K» → 504 553);
`src/preprocessing/README.md` se ha corregido en línea (modelo 3D en vez de
«2.5D», negativos = IXI + NKI Rockland, enumeración de `dataset` con
`nki_rockland`, nota sobre BTC T1-only y descripción correcta de la
estratificación por `(dataset, label)` + `subject_id`).

---

## 20. Problemas encontrados y mitigaciones

| Problema | Evidencia | Riesgo | Archivos relacionados | Mitigación | Estado |
|---|---|---|---|---|---|
| Resultados demasiado buenos (AUC≈1.0) | `outputs/evaluation/cnn3d_test_results.json` | Inducir a error sobre la utilidad clínica | `evaluate_3d.py`, `justificacion_cambios.md` | Auditoría sistemática | Explicado (confound) |
| **Confound dominio↔clase** | `audit_leakage.json` (`label_equals_dataset`); tiny AUC 1.0 | **Crítico**: invalida la interpretación clínica | composición de `data/processed/`, toda `src/audit/` | Tiny baseline + LODO + BTC + embeddings | **Confirmado, estructural, no resuelto** |
| Posible leakage trivial | `audit_leakage.json` (0 duplicados) | Alto si existiera | `audit_leakage.py` | Hash sha1 cruzando splits | Descartado |
| Split por sujeto | `audit_splits` (0 overlap) | Alto si existiera | `audit_splits.py`, `dataset_3d.py` | Agrupación por subject_id | Descartado |
| Duplicados exactos | `audit_leakage.json` | Medio | `audit_leakage.py` | sha1 | Descartado (0) |
| Varianza intra-dataset casi nula | `cnn3d_test_results.json` (UPENN ±0.0005) | Síntoma de detector de dominio | `evaluate_3d.py` | Diagnóstico documentado | Documentado |
| AUC intra-dataset no calculable | `cnn3d_test_results.json` (AUC=NaN ×4) | Imposibilidad de medir detección | `evaluate_3d.py` | Reconocido como síntoma | Documentado |
| Transferencia LODO débil/invertida | `lodo_{A,B}/...json`, `audit_lodo.json` | Falta de robustez OOD | `make_lodo_splits.py`, `run_lodo.sh` | LODO A/B (CNN) + A–D (tiny) | Demostrado |
| BTC con n pequeño | `btc_intradomain_tinybaseline.json` (n=36) | IC95 % anchos | `btc_*.py` | Bootstrap IC95 % | Limitación declarada |
| Ausencia de validación externa real | `evolucion...` §1.2 (UPENN movido a train) | Validez externa no establecida | — | LODO + BTC como sustitutos | Pendiente |
| Calibración pendiente | ausente en el repo | Probabilidades no calibradas | — | Declarada trabajo futuro | Pendiente |
| Documentos desactualizados | `evolucion...`, `preprocessing/README.md` | Confusión sobre el estado real | esos documentos | Banner de actualización en `evolucion...` + correcciones inline en `README.md` | **Corregido** |
| Datos pesados no versionados | `.gitignore` (`/data/`, `outputs/`) | — (es la práctica correcta) | `.gitignore` | Versionar solo `docs/audit/` | Resuelto |
| `recreate_splits: true` (split no congelado) | `configs/train_3d.yaml` | Reproducibilidad entre runs | `configs/`, `dataset_3d.py` | Recomendación de congelar | Pendiente |

---

## 21. Estado final del proyecto

**Implementado (hecho observado):**
- Preprocesado multi-dataset homogéneo (RAS, resample 1 mm³, crop/pad, z-score con
  Otsu, skull-strip HD-BET) para BraTS, UPENN, IXI, NKI y BTC.
- CNN 3D propia (`BrainTumorCNN3D`) con GroupNorm.
- Entrenamiento end-to-end con selección de checkpoint por balanced_accuracy + sen
  mínima.
- Evaluación con métricas globales y por dataset + análisis de umbral.
- Suite de auditoría completa: leakage, splits, LODO (CNN A/B y tiny A–D),
  intra-dominio BTC (tiny + CNN k-fold), embeddings (E1), Grad-CAM.
- Figuras y consolidación de resultados.

**Documentado:** cuatro documentos Markdown principales + borrador de memoria +
resumen consolidado + figuras (con la salvedad de los documentos parcialmente
desactualizados de §19).

**Ejecutable:** `scripts/run_btc_chain.sh`, `scripts/run_lodo.sh` y todos los
`python -m src.*` bajo el entorno conda `igsan` (GPU AMD ROCm).

**Resultados existentes:** run confundido (`20260527_152619`), LODO A/B (CNN) y A–D
(tiny), BTC intra-dominio (CNN + tiny), embeddings, todas las figuras y los JSON
consolidados.

**Maduro:** el pipeline de auditoría y su documentación; es la parte defendible y
de mayor valor.

**Experimental / parcial:** el análisis de embeddings (recién añadido, sin
commitear); los checkpoints 2D legacy (`baseline_2d_*`); `preprocess_volumes.py`
(lanzador redundante).

**Pendiente:**
- Validación externa real con **ambas clases** del mismo dominio no visto.
- Cohorte intra-dominio de mayor tamaño (objetivo n ≥ 100) y, deseablemente, T1+T2.
- Calibración del clasificador (Platt scaling, reliability diagrams).
- LODO C/D con la CNN (solo ejecutados con tiny baseline).
- Congelar `splits.json` (`recreate_splits: false`) para reproducibilidad.
- ~~Actualizar o marcar la documentación desactualizada~~ → **hecho** (banner en
  `evolucion...` y correcciones en `preprocessing/README.md`; ver §19).
- **Redacción final de la memoria** (este documento es su base).

---

## 22. Flujo reproducible completo

> Entorno: conda `igsan`, GPU AMD ROCm. En el repositorio, los scripts usan la ruta
> absoluta del intérprete del entorno; aquí se muestran las invocaciones `python -m`
> equivalentes. Donde un flag no puede confirmarse, se indica.

**1. Instalar dependencias**
```bash
pip install -r requirements.txt
```

**2. Preprocesar datos** (asume datos raw ya disponibles en `data/raw/`)
```bash
# DICOM -> NIfTI (hecho observado: invocación por ruta en el README;
# como módulo -m es deducción)
python -m src.preprocessing.conversion.dicom_to_nifti brats
python -m src.preprocessing.conversion.dicom_to_nifti upenn

python -m src.preprocessing.preprocess_brats
python -m src.preprocessing.preprocess_ixi --skull-strip
python -m src.preprocessing.preprocess_upenn
# La descarga de NKI requiere el CSV de aws_links (hecho observado: README):
#   python -m src.preprocessing.download_nki_rockland --aws-links <aws_links.csv> \
#       --out-dir data/raw/nki_rockland --include-json
python -m src.preprocessing.download_nki_rockland     # (requiere --aws-links: ver README)
python -m src.preprocessing.preprocess_nki_rockland --skull-strip
python -m src.preprocessing.dataset.summarize_processed   # añadir --write para escribir el summary
# Dataset intra-dominio (T1-only):
python -m src.preprocessing.preprocess_btc
```

**3. Entrenar (run principal)**
```bash
python -m src.training.train_3d --config configs/train_3d.yaml
# Genera outputs/checkpoints/<RUN>/best.pt
```

**4. Evaluar**
```bash
python -m src.evaluation.evaluate_3d --config configs/train_3d.yaml \
    --checkpoint outputs/checkpoints/<RUN>/best.pt --split test
```

**5. Análisis de umbral**
```bash
python -m src.evaluation.threshold_analysis \
    --checkpoint outputs/checkpoints/<RUN>/best.pt --target-sensitivity 0.95
```

**6. Auditoría**
```bash
python -m src.data.audit_splits
python -m src.audit.audit_leakage           # tiny baseline + clf dataset + duplicados
python -m src.audit.audit_lodo --per-dataset 150
bash scripts/run_lodo.sh A
bash scripts/run_lodo.sh B
# (C y D análogos; aún no ejecutados con la CNN)
bash scripts/run_btc_chain.sh               # tiny + CNN k-fold intra-dominio + consolidar
```

**7. Embeddings, figuras y consolidación**
```bash
python -m src.audit.embeddings_tsne --checkpoint outputs/checkpoints/<RUN>/best.pt
python -m src.audit.embeddings_intraclass
python -m src.audit.gradcam_3d --checkpoint outputs/checkpoints/<RUN>/best.pt \
    --tag confound --samples upenn:2 ixi:2 brats:1
python -m src.audit.make_plots
python -m src.audit.make_extra_figures
python -m src.audit.consolidate_results
```

> Nota: `<RUN>` es el *timestamp* del directorio de checkpoint (en disco, el run
> confundido oficial es `20260527_152619`). Los flags avanzados de
> `embeddings_tsne.py` y `gradcam_3d.py` más allá de los mostrados en sus docstrings
> **no se han confirmado exhaustivamente**.

---

## 23. Narrativa final para la memoria

> Texto formal, académico y reutilizable en la memoria del TFG.

El presente trabajo se planteó como el desarrollo de un sistema de triaje
automático de resonancia magnética cerebral, formulado como una clasificación
binaria a nivel de estudio (presencia o ausencia de masa tumoral) sobre volúmenes
T1 y T2, con la sensibilidad como métrica clínica prioritaria por tratarse de una
herramienta de priorización y no de diagnóstico.

En una primera fase se construyó un pipeline de preprocesado homogéneo —conversión
a NIfTI, reorientación a espacio RAS, remuestreo a 1 mm isótropo, recorte a una
forma fija y normalización z-score sobre máscara de tejido— y una red neuronal
convolucional tridimensional propia (`BrainTumorCNN3D`), entrenada sobre un conjunto
multi-fuente compuesto por BraTS 2021 y UPENN-GBM (positivos) e IXI y NKI Rockland
(negativos), con un total de 2267 estudios. El sistema se entrenó con técnicas de
estabilización adecuadas a la restricción de memoria (normalización por grupos,
*batch* unitario, selección de modelo por *balanced accuracy* con sensibilidad
mínima) y alcanzó, sobre el conjunto de test, métricas próximas al óptimo: un área
bajo la curva ROC de 0.99997, una sensibilidad de 0.994 y una especificidad de 1.0,
con un único falso negativo y ningún falso positivo.

La magnitud de estas métricas, impropia de la dificultad clínica del problema,
motivó una auditoría metodológica sistemática orientada a determinar si reflejaban
una capacidad real de detección de lesión o alguna forma de contaminación. La
auditoría descartó en primer lugar las fugas triviales: no se hallaron duplicados
exactos cruzando particiones ni solapamiento de sujetos entre los conjuntos de
entrenamiento, validación y prueba, gracias a una partición estratificada por
dataset y etiqueta y agrupada por identificador de sujeto.

Sin embargo, la auditoría reveló un *confound* estructural: en el conjunto
construido, la clase positiva procede exclusivamente de BraTS y UPENN-GBM y la
negativa de IXI y NKI Rockland, de modo que la etiqueta está perfectamente
correlacionada con el dataset de origen. Una prueba de control consistente en
entrenar una regresión logística sobre dieciséis estadísticos de intensidad no
clínicos alcanzó un área bajo la curva de 1.0 sobre el mismo conjunto de prueba, lo
que demuestra que la etiqueta es decodificable sin red neuronal ni información
anatómica. Un clasificador del dataset de origen alcanzó una exactitud del 98.5 %,
confirmando que el dominio es trivialmente identificable.

La dependencia del modelo respecto al dominio se confirmó mediante un experimento de
generalización cruzada (*leave-one-domain-out*): al entrenar y evaluar sobre pares
de dominios disjuntos, el rendimiento se desplomó y, en una de las direcciones,
descendió por debajo de 0.5, lo que indica una inversión de la regla de decisión
aprendida. Para obtener una medición no contaminada se incorporó el dataset
BTC_preop (OpenNeuro ds001226), que contiene pacientes con tumor y controles sanos
adquiridos en el mismo equipo de resonancia. Bajo validación cruzada por sujeto, el
área bajo la curva de la red descendió a 0.404, con un intervalo de confianza al
95 % de [0.21, 0.62] que incluye el valor de azar. Finalmente, el análisis del
espacio latente de la red mostró que la representación interna codifica la
procedencia del estudio incluso entre sujetos de la misma clase: dos cohortes de
sujetos sanos (IXI y NKI Rockland) resultaron separables con un área bajo la curva
de 0.998.

En conjunto, estas evidencias demuestran que el rendimiento casi perfecto obtenido
en el conjunto multi-fuente es atribuible al *confound* de dominio y no constituye
evidencia de detección de tumor. En consecuencia, la aportación del trabajo se
reformuló: de un clasificador aparentemente exitoso a una caracterización
metodológica reproducible de un caso de aprendizaje espurio en clasificación
tridimensional de RM cerebral, acompañada de un protocolo de auditoría —*tiny
baseline* de intensidad, *leave-one-domain-out*, validación intra-dominio, análisis
del espacio latente e interpretabilidad— reutilizable para sistemas análogos. La
validación clínica del sistema como herramienta de triaje queda condicionada a la
disponibilidad de una cohorte intra-dominio de mayor tamaño y, deseablemente,
multicéntrica, con ambas clases representadas en cada centro.

---

## 24. Conclusión técnica

1. **El modelo inicial no puede defenderse como detector clínico de tumor.** La
   evidencia (tiny baseline AUC 1.0, clasificador de dataset acc 0.985, colapso e
   inversión en LODO, rendimiento al azar intra-dominio, separabilidad de cohortes
   de la misma clase en el espacio latente) indica que el sistema discrimina
   **dominio/procedencia**, no patología.

2. **El AUC ≈ 1.0 se explica por el confound de dominio**, no por capacidad de
   detección. La correlación clase↔dataset del 100 % hace que cualquier métrica
   sobre el pool multi-fuente sea, en grado indeterminado, una medida de
   reconocimiento de dataset.

3. **El proyecto gana valor al detectar, demostrar y documentar ese fallo.** La
   transición desde un resultado aparentemente excelente hacia una auditoría
   rigurosa es precisamente lo que confiere madurez metodológica al TFG y lo hace
   defendible ante un tribunal: no se oculta el problema, se cuantifica.

4. **La auditoría es reproducible y multi-ángulo.** Combina pruebas cuantitativas
   triviales (tiny baseline), de generalización (LODO), de dominio controlado (BTC),
   de representación interna (embeddings) e interpretabilidad (Grad-CAM), de modo que
   ninguna conclusión descansa en una sola evidencia.

5. **El siguiente paso real** es obtener una cohorte —externa o intra-dominio— con
   **ambas clases dentro del mismo dominio**, de **mayor tamaño muestral** y,
   preferiblemente, **multicéntrica y equilibrada**, sobre la que medir la capacidad
   de detección sin confound; complementariamente, calibrar el clasificador y, a
   medio plazo, plantear una validación prospectiva.

**Cierre.** El TFG no entrega un detector de tumores validado, y sería deshonesto
presentarlo como tal. Entrega algo metodológicamente más valioso para la formación
en Ingeniería Matemática aplicada a la medicina: una demostración rigurosa,
trazable y reproducible de **por qué un modelo de Deep Learning médico puede
parecer perfecto y no servir**, junto con el instrumental de auditoría necesario
para detectarlo. Esa es la lección central —y la aportación defendible— del
proyecto.

---

*Documento generado como reconstrucción técnica del repositorio `brain-mri-triage`.
Cifras verificadas contra los JSON del repositorio. Las interpretaciones se han
marcado como tales; los documentos internos desactualizados se han señalado
explícitamente en la §19.*
