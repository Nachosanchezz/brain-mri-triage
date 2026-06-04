# Material para redacción de la memoria y respuesta al tutor

> **Título propuesto:** *Priorización automática de RM cerebral mediante aprendizaje
> profundo: análisis crítico del sesgo de dominio en datasets públicos multi-fuente.*
>
> **Propósito de este documento.** Reunir, ya redactado y trazable, todo el material que
> pidió el tutor: matriz dataset/clase, tabla resumen de resultados, índice final ajustado,
> objetivos definitivos, explicación de las 4 figuras principales, decisión sobre Grad-CAM,
> narrativa de resultados en cascada, discusión, conclusiones, correo de respuesta y bloques
> listos para copiar a la memoria.
>
> **Trazabilidad.** Toda cifra marcada **[V]** se ha verificado contra un archivo del
> repositorio (se cita la ruta). Lo no confirmable se indica explícitamente como tal.
>
> **Nota sobre las figuras.** En el correo se escribió `docs/figiures/principales/`; la ruta
> real es **`docs/audit/figures/principales/`** (4 figuras) y el material de anexo está en
> **`docs/audit/figures/anexo/`**.

---

## 0. Mapa de cifras verificadas (fuente de verdad)

| Magnitud | Valor | Archivo fuente | Estado |
|---|---|---|---|
| Pool total | 2267 (1167 pos / 1100 neg) | `data/processed/preprocessing_summary.json` | [V] |
| BraTS / UPENN / IXI / NKI | 580 / 587 / 577 / 523 | `data/processed/preprocessing_summary.json` | [V] |
| Split confounded | train 1587 / val 340 / test 340, seed 42 | `data/splits_confound_original.json` | [V] |
| Test (pos/neg) | 175 / 165 (brats 87, upenn 88, ixi 86, nki 79) | `outputs/evaluation/cnn3d_test_results.json` | [V] |
| Parámetros CNN | 504 553 (2 canales) / 504 229 (1 canal) | instanciado desde `src/models/cnn3d.py` | [V] |
| CNN pool — AUC / PR-AUC | 0.999965 / 0.999968 | `outputs/evaluation/cnn3d_test_results.json` | [V] |
| CNN pool — acc / sen / spe | 0.99706 / 0.99429 / 1.0 | `outputs/evaluation/cnn3d_test_results.json` | [V] |
| CNN pool — matriz | TP 174 / FP 0 / TN 165 / FN 1 | `outputs/evaluation/cnn3d_test_results.json` | [V] |
| Scores por dataset | upenn 0.9995±0.0005, brats 0.986±0.098, ixi 0.0070±0.018, nki 0.0008±0.0005 | `cnn3d_test_results.json` | [V] |
| AUC intra-dataset | NaN (cada dataset es mono-clase) | `cnn3d_test_results.json` | [V] |
| Tiny mezcla — LogReg / RF | AUC 1.0 (acc 0.9941) / 0.9989 (acc 0.9882) | `docs/audit/audit_leakage.json` | [V] |
| Tiny mezcla — entrenamiento | n=600 balanceado (300/300), test n=340 | `docs/audit/audit_leakage.json` | [V] |
| Clf dataset (intensidad, 4 clases) | acc 0.9853 (azar 0.25) | `docs/audit/audit_leakage.json` | [V] |
| Duplicados / cross-split / overlap sujeto | 0 / 0 / 0 | `docs/audit/audit_leakage.json` | [V] |
| `label_equals_dataset` | brats[1] upenn[1] ixi[0] nki[0] | `docs/audit/audit_leakage.json` | [V] |
| Features top (RF) | nz_frac_t1 0.174, p75_t1 0.171, p25_t2 0.125, nz_frac_t2 0.101 | `docs/audit/audit_leakage.json` | [V] |
| LODO A (CNN) | AUC 0.6236, sen 0.676, spe 0.463 (test n=1110) | `outputs/evaluation/lodo_A/cnn3d_test_results.json` | [V] |
| LODO B (CNN) | AUC 0.2012, sen 0.0103, spe 0.9653 (test n=1157) | `outputs/evaluation/lodo_B/cnn3d_test_results.json` | [V] |
| Tiny LODO A/B/C/D (LogReg) | 0.9952 / 0.3184 / 0.0412 / 0.0469 (ref random-mix 1.0) | `docs/audit/audit_lodo.json` | [V] |
| BTC tiny LogReg | AUC 0.5491, IC95 [0.319, 0.788] | `docs/audit/btc_intradomain_tinybaseline.json` | [V] |
| BTC tiny RF | AUC 0.4055, IC95 [0.215, 0.616] | `docs/audit/btc_intradomain_tinybaseline.json` | [V] |
| BTC CNN 1-canal | AUC 0.4036, IC95 [0.213, 0.623], sen 0.60, spe 0.364 (n=36, k=5) | `outputs/evaluation/btc_intradomain/cnn_kfold_results.json` | [V] |
| Embeddings silhouette | label 0.754 / dataset 0.366 | `docs/audit/embeddings_silhouette.json` | [V] |
| Embeddings intra-clase | IXI-vs-NKI CV-AUC 0.998 (sil 0.561); BraTS-vs-UPENN 0.991 (sil 0.186) | `docs/audit/embeddings_intraclass.json` | [V] |
| Clf dataset desde embeddings | acc 0.982 (azar 0.25) | `docs/audit/embeddings_intraclass.json` | [V] |

> **Dos matices de honestidad que conviene declarar en la memoria:**
> 1. El *tiny baseline* de la mezcla se entrenó sobre un subconjunto **balanceado de
>    n=600** (300/300) y se evaluó sobre el mismo test de 340; no es exactamente el split
>    de la CNN (1587/340/340). La conclusión (AUC=1.0 sin red ni anatomía) es válida, pero
>    el protocolo debe describirse como es. Fuente: `docs/audit/audit_leakage.json`.
> 2. El archivo `data/splits.json` está actualmente sobrescrito por el último run de LODO
>    (`config:B`). El split de la mezcla original se conserva aparte en
>    `data/splits_confound_original.json`. Esto es exactamente la consecuencia de tener
>    `recreate_splits:true` en `configs/train_3d.yaml`, y debe documentarse como limitación
>    de reproducibilidad (ver §10).

---

## 1. Matriz / tabla dataset ↔ clase (Tarea 2)

**Tabla principal (la que pidió el tutor: cada dataset aporta una sola clase).**

| Dataset (origen) | Procedencia / dominio | Clase aportada | n procesado | Modalidades usadas | Uso en el proyecto | Riesgo metodológico |
|---|---|---|---|---|---|---|
| **BraTS 2021** | Multicéntrico (challenge BraTS) | **Tumor (1)** | 580 [V] | T1 + T2 | Entrenamiento (pool) | Aporta **solo** positivos → clase ligada al dominio |
| **UPENN-GBM** | Hospital Univ. de Pensilvania (TCIA) | **Tumor (1)** | 587 [V] | T1 + T2 | Entrenamiento (pool); fue test externo | Aporta **solo** positivos; firma de intensidad muy marcada (ver §6 fig. intensidad) |
| **IXI** | 3 hospitales de Londres (sanos) | **Sano (0)** | 577 [V] | T1 + T2 | Entrenamiento (pool) | Aporta **solo** negativos |
| **NKI Rockland** | Nathan Kline Institute (sanos) | **Sano (0)** | 523 [V] | T1 + T2 | Entrenamiento (pool); reequilibrio de negativos | Aporta **solo** negativos |
| **BTC_preop** (OpenNeuro ds001226) | Ghent University Hospital (un solo centro) | **Tumor (1) y Sano (0)** | 36 (25 pat. / 11 ctrl.) [V] | T1-only | **Validación intra-dominio** (única medición honesta) | n pequeño; T1-only |

**Lectura inmediata (frase para la memoria):** en el *pool* de entrenamiento/evaluación
principal, **ningún dataset aporta las dos clases**. Por construcción,
`P(tumor | BraTS) = P(tumor | UPENN) = 1` y `P(tumor | IXI) = P(tumor | NKI) = 0`. La
correlación entre clase y dataset de origen es del **100 %**, lo que hace que «detectar
tumor» y «reconocer el centro de procedencia» sean tareas **indistinguibles** en la
evaluación global. BTC_preop es el único conjunto que rompe esta confusión (ambas clases
en el mismo escáner) y por eso es la única validación honesta.

**Versión mínima (para diapositiva o caption):**

| | Tumor (1) | Sano (0) |
|---|---|---|
| **Pool principal** | BraTS, UPENN | IXI, NKI |
| **Validación honesta** | BTC_preop (Ghent) | BTC_preop (Ghent) |

Verificación directa del confound: `label_equals_dataset = {brats:[1], upenn:[1],
ixi:[0], nki_rockland:[0]}` en `docs/audit/audit_leakage.json` (cada dataset solo presenta
una etiqueta).

---

## 2. Tabla resumen de resultados (Tarea 3)

> Ordenada en cascada: de «parece funcionar» a la medición honesta. La columna **Validez**
> marca lo que es metodológicamente sólido (✅) frente a lo aparentemente bueno pero
> **inválido por confound** (❌).

| # | Régimen de evaluación | Modelo / prueba | Dataset o partición | AUC | Sen | Spe | IC95 % | Interpretación | Validez |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Pool multi-fuente (split aleatorio) | **CNN 3D (2 canales)** | test 340 (175+/165−) | **0.99997** | 0.994 | 1.000 | — | Rendimiento aparente casi perfecto | ❌ inflado por confound |
| 2 | Pool multi-fuente | Tiny baseline LogReg (16 stats intensidad) | test 340 (train n=600) | **1.000** | acc 0.994 | — | — | Etiqueta decodificable sin red ni anatomía | ❌ atajo trivial |
| 3 | Pool multi-fuente | Tiny baseline RF | test 340 | 0.9989 | acc 0.988 | — | — | Ídem | ❌ atajo trivial |
| 4 | Identificabilidad de dominio | RF clasif. de dataset (intensidad, 4 clases) | test 340 | acc **0.9853** | — | — | — | El centro de origen es trivialmente separable (azar 0.25) | evidencia de confound |
| 5 | Identificabilidad de dominio | Clasif. de dataset desde *embeddings* (4 clases) | latente 96-d | acc **0.982** | — | — | — | El latente de la CNN codifica procedencia (azar 0.25) | evidencia de confound |
| 6 | LODO A: BraTS+IXI → UPENN+NKI | **CNN 3D (2 canales)** | test n=1110 | **0.6236** | 0.676 | 0.463 | — | Degrada hacia el azar fuera de dominio | ❌ no generaliza |
| 7 | LODO B: UPENN+NKI → BraTS+IXI | **CNN 3D (2 canales)** | test n=1157 | **0.2012** | 0.010 | 0.965 | — | **Regla invertida / colapso** de sensibilidad | ❌ no generaliza |
| 8 | LODO A | Tiny baseline LogReg | held-out | 0.9952 | — | — | — | El atajo transfiere en una dirección | evidencia de confound |
| 9 | LODO B | Tiny baseline LogReg | held-out | 0.3184 | — | — | — | El atajo no transfiere en la otra | evidencia de confound |
| 10 | **BTC_preop intra-dominio** | **CNN 3D (1 canal)** | k-fold, n=36 | **0.4036** | 0.60 | 0.364 | **[0.213, 0.623]** | **Compatible con azar** (IC cruza 0.5) | ✅ medición honesta |
| 11 | **BTC_preop intra-dominio** | Tiny baseline LogReg | k-fold, n=36 | **0.5491** | — | — | **[0.319, 0.788]** | Compatible con azar | ✅ medición honesta |
| 12 | **BTC_preop intra-dominio** | Tiny baseline RF | k-fold, n=36 | 0.4055 | — | — | [0.215, 0.616] | Compatible con azar | ✅ medición honesta |
| 13 | *Embeddings* intra-clase | LogReg sobre latente: **IXI vs NKI (ambos sanos)** | n=1100 | **0.998** | — | — | — | Separa cohortes **dentro de una misma clase** → codifica origen, no tumor | evidencia de confound |

**Conclusión de la tabla (frase para la memoria):** las filas 1–3 muestran un rendimiento
aparentemente perfecto; las filas 4–5, 8–9 y 13 demuestran que ese rendimiento es
**identificación de dominio**; las filas 6–7 muestran que el modelo no generaliza fuera de
dominio (incluso invierte la regla); y las filas 10–12, la única medición con clase y dominio
desacoplados, sitúan el rendimiento real **al nivel del azar**.

---

## 3. Índice final ajustado con las recomendaciones del tutor (Tarea 4)

> Cambios del tutor incorporados literalmente: 1.3 retitulado, objetivo general
> reformulado (cap. 2), pilar teórico de *shortcut learning* (3.5), matriz clase-dataset
> (4.3), «Protocolo de auditoría de validez experimental» (6.3), título del cap. 7,
> nuevo 8.6, y conclusiones no defensivas.

**1. Introducción**
- 1.1. Contexto clínico y motivación (carga radiológica; concepto de triaje/priorización).
- 1.2. Planteamiento del problema (clasificación binaria a nivel de estudio sobre T1/T2; prioridad de la sensibilidad).
- 1.3. **De la hipótesis clínica inicial a la auditoría metodológica del modelo** *(giro como contribución, guiado por la evidencia)*.
- 1.4. Objetivos y contribuciones.
- 1.5. Estructura de la memoria.

**2. Objetivos**
- 2.1. Objetivo general *(reformulado: pipeline 3D + evaluación crítica de validez en escenario multi-fuente; NO promete validación clínica)*.
- 2.2. Objetivos específicos.

**3. Estado del arte**
- 3.1. Diagnóstico por imagen de masas tumorales intracraneales.
- 3.2. RM cerebral y limitaciones del diagnóstico radiológico (variabilidad de protocolo/escáner).
- 3.3. Deep Learning en imagen médica.
- 3.4. DL aplicado a RM cerebral: enfoques, datasets y retos de generalización entre centros.
- 3.5. **Sesgo de dominio, *shortcut learning* y fugas de información en DL médico** *(pilar teórico del TFG, no apartado secundario)*.
- 3.6. Sistemas de priorización/triaje en radiología (requisitos de sensibilidad y de validación externa).

**4. Materiales y datos**
- 4.1. Datasets (BraTS 2021, UPENN-GBM, IXI, NKI Rockland, BTC_preop/ds001226).
- 4.2. Definición de clases y etiquetado.
- 4.3. **Composición del problema: matriz clase ↔ dataset de origen** *(tabla clave: cada dataset aporta una sola clase)*.
- 4.4. Selección de modalidades (T1+T2; consecuencia: BTC es T1-only).
- 4.5. Análisis exploratorio de los datos (distribuciones de intensidad por dataset).

**5. Metodología**
- 5.1. Preprocesado homogéneo (DICOM→NIfTI, HD-BET, RAS, remuestreo 1 mm³, crop/pad, z-score con máscara Otsu).
- 5.2. Arquitectura del modelo (CNN 3D, 504 553 parámetros).
- 5.3. Función de pérdida, optimización e hiperparámetros.
- 5.4. Data augmentation (solo sobre vóxeles de cerebro).
- 5.5. Estrategia de partición a nivel de sujeto (estratificada por (dataset,label), agrupada por subject_id).
- 5.6. Entorno computacional y herramientas (GPU AMD/ROCm, PyTorch).

**6. Diseño experimental y protocolo de validación**
- 6.1. Métricas de evaluación y análisis operativo (AUC, PR-AUC, sen/spe, IC95 % por bootstrap).
- 6.2. Validación interna sobre el pool multi-fuente.
- 6.3. **Protocolo de auditoría de validez experimental** *(metodología propia del trabajo)*:
  - 6.3.1. Baseline trivial sobre estadísticos de intensidad.
  - 6.3.2. Identificabilidad del dominio.
  - 6.3.3. Validación cruzada por dominio (leave-one-dataset-out).
  - 6.3.4. Validación intra-dominio (BTC_preop).
  - 6.3.5. Análisis del espacio latente (*embeddings*).
  - 6.3.6. Mapas de atención (Grad-CAM) — solo como apoyo cualitativo.

**7. Resultados: análisis progresivo de validez del rendimiento aparente**
- 7.1. Rendimiento aparente en el pool multi-fuente.
- 7.2. Decodificabilidad trivial de la etiqueta (baseline sin red).
- 7.3. Identificabilidad del dominio.
- 7.4. Generalización fuera de dominio (LODO en ambas direcciones).
- 7.5. Validación intra-dominio honesta (BTC_preop).
- 7.6. Evidencia en el espacio latente y mapas de atención.
- 7.7. Síntesis consolidada de resultados.

**8. Discusión**
- 8.1. Interpretación: el confound estructural dominio↔clase.
- 8.2. Por qué las métricas casi perfectas no son evidencia clínica.
- 8.3. Viabilidad y condiciones para un triaje radiológico válido.
- 8.4. Relación con *shortcut learning*, *dataset bias* y *domain shift* (literatura).
- 8.5. Validez metodológica y reproducibilidad.
- 8.6. **Implicaciones para el diseño de estudios con datos públicos multi-fuente** *(apartado nuevo pedido por el tutor)*.

**9. Conclusiones** *(claras y no defensivas)*.

**10. Limitaciones y líneas futuras**
- 10.1. Limitaciones del estudio.
- 10.2. Propuestas de mejora y líneas futuras.

**Referencias bibliográficas**

**Anexos**
- A. Detalle del preprocesado y configuraciones (`configs/`).
- B. Tabla maestra de resultados y detalle por fold de BTC (`docs/audit/resumen_consolidado.{md,json}`).
- C. Figuras adicionales (ROC, histogramas por dataset, PCA, Grad-CAM por régimen).
- D. Reproducibilidad: comandos, seeds y entorno.

---

## 4. Objetivo general y objetivos específicos definitivos (Tarea 5)

### Objetivo general

> **Desarrollar un *pipeline* de clasificación 3D para resonancia magnética cerebral y
> evaluar críticamente su validez en un escenario multi-fuente, identificando y
> cuantificando el impacto del sesgo de dominio en la tarea de priorización tumoral.**

*(Formulación deliberadamente neutra respecto a la validez clínica: el objetivo no es
demostrar que el sistema sirve para triaje, sino determinar en qué medida su rendimiento
constituye evidencia de detección de tumor.)*

### Objetivos específicos

- **O1.** Construir un *pipeline* de preprocesado homogéneo y reproducible para RM
  multi-dataset (normalización de espacio, intensidad y geometría).
- **O2.** Implementar y entrenar una CNN 3D de clasificación binaria tumor/no-tumor a
  nivel de estudio sobre volúmenes T1+T2.
- **O3.** Evaluar el rendimiento aparente con métricas clínicas (AUC, PR-AUC,
  sensibilidad, especificidad, matriz de confusión).
- **O4.** Auditar la presencia de fugas triviales de información (duplicados, solapamiento
  de sujetos entre particiones).
- **O5.** Cuantificar la **decodificabilidad trivial** de la etiqueta mediante un *baseline*
  sin red ni información espacial (estadísticos de intensidad).
- **O6.** Evaluar la **generalización fuera de dominio** mediante validación cruzada por
  dataset (*leave-one-dataset-out*, ambas direcciones).
- **O7.** Contrastar el rendimiento con una **validación intra-dominio** honesta
  (BTC_preop: ambas clases en el mismo centro).
- **O8.** Analizar la **representación interna** (espacio latente / *embeddings*) para
  determinar si codifica tumor o procedencia.
- **O9.** Proponer y documentar un **protocolo reproducible de auditoría anti-confounding**
  como contribución metodológica transferible a otros estudios multi-fuente.

---

## 5. Explicación académica de las 4 figuras principales (Tarea 6)

> Ruta: `docs/audit/figures/principales/`. Las cuatro sostienen, en orden, la narrativa
> «parece funcionar → se audita → se desmonta».

### Figura 1 — `auc_summary.png`

- **Título académico sugerido:** *Desplome del AUC al desacoplar progresivamente clase y
  dominio: del pool confundido a la validación intra-dominio.*
- **Caption formal:** «AUC por régimen de evaluación. Barras de error: IC95 % por
  *bootstrap* cuando aplica (BTC intra-dominio). La línea discontinua marca el azar
  (0.5). El rendimiento casi perfecto del pool multi-fuente (CNN y *tiny baseline*,
  AUC≈1.0) se desploma al evaluar fuera de dominio (LODO A 0.624; LODO B 0.201) y cae al
  nivel del azar cuando clase y dominio se desacoplan (BTC intra-dominio: *tiny* LogReg
  0.549 [0.319, 0.788]; CNN 0.404 [0.213, 0.623]). Fuente: `docs/audit/resumen_consolidado.json`.»
- **Explicación para el texto principal:** la figura resume en una sola vista la tesis
  central del trabajo. A la izquierda, dos barras pegadas al techo (mezcla CNN y *tiny
  baseline*) ilustran que el rendimiento aparente es trivialmente alcanzable. A la derecha,
  el rendimiento honesto (BTC) queda con intervalos de confianza que cruzan 0.5. La
  pendiente descendente **es** el resultado del trabajo.
- **Interpretación metodológica:** el AUC no mide aquí «detección de tumor»; mide cuánto
  confound queda disponible en cada régimen. Al retirar el atajo de dominio, no queda señal.
- **Mensaje clave:** *cuanto más honesto es el régimen de evaluación, más baja el AUC.*
- **Ubicación:** §7.7 (figura de síntesis principal del capítulo de resultados).

### Figura 2 — `embeddings_tsne.png`

- **Título académico sugerido:** *Proyección t-SNE del espacio latente de la CNN: la
  representación interna se organiza por cohorte de origen.*
- **Caption formal:** «Proyección t-SNE del vector latente de 96 dimensiones (capa previa
  al clasificador) del *checkpoint* del pool confundido. Izquierda: coloreado por dataset
  de origen; derecha: por etiqueta. Las cuatro cohortes forman grupos propios. IXI y NKI,
  **ambas sanas**, no se fusionan; un clasificador lineal sobre estos vectores distingue
  IXI de NKI con CV-AUC=0.998 y predice el dataset (4 clases) con acc=0.982 (azar 0.25).
  Fuente: `docs/audit/embeddings_intraclass.json`.»
- **Explicación para el texto principal:** si la red hubiera aprendido «tumor», el panel
  derecho (por etiqueta) debería mostrar dos grupos limpios y el izquierdo (por dataset)
  debería mezclar las dos cohortes de cada clase. Ocurre lo contrario: el panel por dataset
  muestra cuatro islas y las dos cohortes sanas (IXI, NKI) permanecen separadas. La red
  representa **de dónde viene** la imagen, no si hay lesión.
- **Interpretación metodológica:** es la evidencia más directa de *shortcut learning*: la
  separabilidad **intra-clase** (CV-AUC 0.998 entre dos conjuntos de la misma clase) solo
  puede deberse a la firma del centro/escáner, nunca al tumor.
- **Mensaje clave:** *el modelo codifica procedencia, no patología.*
- **Ubicación:** §7.6 (figura estrella de la evidencia latente).

### Figura 3 — `confusion_matrices.png`

- **Título académico sugerido:** *Matrices de confusión por régimen: de la diagonal perfecta
  al colapso y la inversión de la regla.*
- **Caption formal:** «Matrices de confusión (umbral 0.5) por régimen. Confounded (mix):
  diagonal casi perfecta (TP 174 / FP 0 / TN 165 / FN 1). LODO A: dispersión (242/281;
  190/397). LODO B: **colapso de sensibilidad** (TP 6 de 580; sen 0.010) compatible con una
  regla invertida (AUC 0.201). Ghent/BTC intra-dominio: distribución compatible con azar
  (15/10; 4/7). Fuentes: `cnn3d_test_results.json`, `lodo_{A,B}/cnn3d_test_results.json`,
  `btc_intradomain/cnn_kfold_results.json`.»
- **Explicación para el texto principal:** la secuencia de cuatro paneles narra el mismo
  desmontaje que la Figura 1, pero a nivel de aciertos/errores. La diagonal impecable del
  primer panel se degrada (LODO A), se rompe e invierte (LODO B, casi nada de tumor
  detectado) y se diluye al azar (BTC).
- **Interpretación metodológica:** un AUC<0.5 (LODO B) con sen=0.01 no es «mal modelo»: es
  un modelo que aplica fuera de dominio una regla aprendida sobre la firma de UPENN/NKI que
  ya no se cumple, llegando a predecir casi todo como «sano».
- **Mensaje clave:** *fuera del dominio de entrenamiento, la regla aprendida no solo falla:
  se invierte.*
- **Ubicación:** §7.1 (panel confounded) y §7.4 (paneles LODO); se puede presentar entera en 7.4.

### Figura 4 — `intensity_by_dataset.png`

- **Título académico sugerido:** *El confound es visible en píxeles crudos: estadísticos de
  intensidad por dataset.*
- **Caption formal:** «Distribución de cuatro estadísticos de intensidad de T1 por dataset
  (fracción de vóxeles no-cero, percentil 25, mediana y percentil 99). UPENN se separa
  nítidamente del resto (mayor fracción de vóxeles no-cero y percentil 99 muy elevado), y
  cada cohorte ocupa un rango propio. Estas mismas 16 *features* permiten a una regresión
  logística predecir la etiqueta con AUC=1.0. Fuentes: `docs/audit/audit_features.csv`,
  `docs/audit/audit_leakage.json`.»
- **Explicación para el texto principal:** la figura demuestra que el confound no es sutil
  ni propio del *deep learning*: está presente en estadísticos triviales calculados sobre
  los vóxeles. Sirve de apoyo directo al *baseline* trivial (§7.2).
- **Interpretación metodológica:** si los datasets son separables ya en intensidad, y la
  clase coincide con el dataset, entonces la clase es separable en intensidad **sin
  anatomía**. La red no necesita ver el tumor para acertar.
- **Mensaje clave:** *la etiqueta está escrita en la firma de intensidad del centro.*
- **Ubicación:** §4.5 (análisis exploratorio) y §7.2 (apoyo al baseline trivial).

---

## 6. Decisión sobre los Grad-CAM (Tarea 7)

**Recomendación: ANEXO (anexo C), con una mención breve de una o dos frases en §7.6.**

**Material disponible** (`docs/audit/figures/anexo/gradcam/`): tres regímenes
(`confound/`, `lodo_A/`, `lodo_B/`), 6–8 sujetos por régimen, con cortes axial/coronal/
sagital y *overlay* de activación.

**Justificación de la decisión (revisado visualmente):**

- Los mapas son **cualitativos y de baja resolución espacial**: tras cuatro bloques de
  *pooling* el mapa de activación tiene una rejilla muy gruesa, por lo que la atención
  aparece como manchas amplias, no como contornos de lesión.
- En el régimen confounded, el positivo de ejemplo (`brats_BraTS2021_00432`, score 0.9994)
  concentra calor en un hemisferio, pero **no se puede afirmar que coincida con la lesión**
  sin la máscara de segmentación superpuesta (no disponible en la figura). En el negativo
  (`ixi_IXI282-HH-2025`, score 0.0020) la activación es **difusa y periférica** (bordes del
  cerebro), coherente con que la red atiende a características globales de
  intensidad/procedencia, pero no es una prueba limpia por sí sola.
- El tutor advirtió explícitamente: *«mapas de atención solo si son interpretables y no
  decorativos»*. Estos no son lo bastante nítidos para sostener una afirmación por sí
  mismos.

**Por tanto:** la evidencia fuerte del confound ya está cubierta por *tiny baseline*,
identificabilidad de dominio, LODO, BTC y *embeddings*. Los Grad-CAM **complementan** esa
narrativa de forma cualitativa pero no la sostienen; ubicarlos en resultados principales
arriesgaría parecer decorativo. **Anexo C**, con una mención de apoyo en §7.6 del tipo:
«de forma coherente con el análisis del espacio latente, los mapas Grad-CAM (Anexo C) no
localizan de manera consistente una lesión, sino patrones difusos y periféricos compatibles
con atención a características globales del volumen».

---

## 7. Narrativa de resultados en cascada (Tarea 8) — texto para la memoria

### 7.1. Rendimiento aparente en el pool multi-fuente

La CNN 3D, evaluada sobre la partición de test del *pool* multi-fuente (n=340; 175
positivos / 165 negativos), alcanza un rendimiento prácticamente perfecto: AUC=0.99997,
PR-AUC=0.99997, exactitud 0.997, sensibilidad 0.994 y especificidad 1.0, con una matriz de
confusión de 174 verdaderos positivos, 0 falsos positivos, 165 verdaderos negativos y un
único falso negativo (`outputs/evaluation/cnn3d_test_results.json`). Tomadas de forma
aislada, estas cifras sugerirían un detector de tumor casi ideal.

Sin embargo, el desglose por dataset enciende ya las primeras alarmas. Las puntuaciones de
salida se separan de forma casi degenerada según el origen: UPENN 0.9995±0.0005, BraTS
0.986±0.098, IXI 0.0070±0.018 y NKI 0.0008±0.0005. La varianza intra-dataset es casi nula
y, puesto que cada dataset es mono-clase, el AUC intra-dataset es **no calculable** (NaN en
los cuatro casos). Es decir: el modelo no necesita discriminar dentro de un dataset, porque
cada dataset coincide con una única etiqueta. Estas dos observaciones —separación por origen
y ausencia de AUC intra-dataset— motivan la auditoría que sigue.

### 7.2. Decodificabilidad trivial de la etiqueta

Para comprobar si la etiqueta puede predecirse **sin red ni información anatómica**, se
entrenó un *baseline* trivial sobre 16 estadísticos globales de intensidad (fracción de
vóxeles no-cero, media, desviación y percentiles de T1 y T2). Una regresión logística
alcanza **AUC=1.0** (exactitud 0.994) y un *random forest* 0.9989, sobre el mismo test
(`docs/audit/audit_leakage.json`; el *baseline* se entrena sobre un subconjunto balanceado
de 600 volúmenes). Las *features* más importantes son la fracción de vóxeles no-cero de T1
(0.174), el percentil 75 de T1 (0.171) y el percentil 25 de T2 (0.125).

La conclusión es contundente: la etiqueta es **linealmente separable a partir de la firma de
intensidad y preprocesado**, sin convoluciones ni estructura espacial. La distribución de
estos estadísticos por dataset (Figura 4, `intensity_by_dataset.png`) muestra que el
confound es visible incluso en píxeles crudos. Un modelo que iguala a la CNN sin «ver» el
cerebro indica que la señal explotada no es de tumor, sino de dominio/preprocesado.

### 7.3. Identificabilidad del dominio

Si la señal es de dominio, el dataset de origen debería ser fácilmente predecible. Lo es: un
clasificador de las cuatro cohortes a partir de las mismas *features* de intensidad alcanza
una exactitud de **0.9853** frente a un azar de 0.25 (`docs/audit/audit_leakage.json`). El
mismo resultado se reproduce, e incluso se refuerza, en el espacio latente de la red: un
clasificador de dataset a partir de los *embeddings* logra exactitud 0.982 (§7.6). En
paralelo, se descartan fugas triviales clásicas: 0 duplicados exactos, 0 grupos
solapados entre particiones y 0 solapamiento de sujeto entre *train*/*val*/*test*
(`docs/audit/audit_leakage.json`). El confound, por tanto, no es un artefacto de partición:
es estructural, propio de la composición de los datos.

### 7.4. Generalización fuera de dominio (LODO)

La validación cruzada por dominio (*leave-one-dataset-out*) entrena en dos datasets y evalúa
en los otros dos, de forma que la clase deja de coincidir con el origen visto en
entrenamiento. Los resultados desmontan la hipótesis de detección robusta:

- **LODO A** (entrena BraTS+IXI, evalúa UPENN+NKI; test n=1110): AUC=0.6236, sensibilidad
  0.676, especificidad 0.463 (`outputs/evaluation/lodo_A/cnn3d_test_results.json`). El
  rendimiento se desploma desde ≈1.0 hasta cerca del azar.
- **LODO B** (entrena UPENN+NKI, evalúa BraTS+IXI; test n=1157): AUC=0.2012, sensibilidad
  **0.010**, especificidad 0.965 (`outputs/evaluation/lodo_B/cnn3d_test_results.json`). Un
  AUC inferior a 0.5 indica una **regla invertida**: el modelo, aplicando la firma aprendida
  sobre UPENN/NKI a unas cohortes con firma distinta, clasifica casi todos los tumores de
  BraTS como sanos (6 verdaderos positivos de 580).

El *baseline* trivial exhibe el mismo comportamiento asimétrico (LODO A LogReg 0.9952; LODO
B 0.3184; configuraciones C y D 0.041 y 0.047; `docs/audit/audit_lodo.json`), lo que confirma
que la (no) transferencia se explica por la firma de intensidad de cada cohorte, no por una
representación anatómica de tumor. Ningún detector real puede colapsar a sensibilidad 0.01 al
cambiar de centro.

### 7.5. Validación intra-dominio honesta (BTC_preop)

BTC_preop (OpenNeuro ds001226, Ghent University Hospital) es el único conjunto que aporta
**las dos clases en el mismo dominio** (25 pacientes / 11 controles, T1-only), por lo que es
la única evaluación en la que clase y procedencia están desacopladas. Bajo validación
cruzada estratificada por sujeto (k=5, 20 épocas fijas por *fold* sin *early stopping*):

- **CNN 3D (1 canal):** AUC=0.4036, IC95 % [0.213, 0.623], sensibilidad 0.60, especificidad
  0.364 (`outputs/evaluation/btc_intradomain/cnn_kfold_results.json`).
- **Tiny LogReg:** AUC=0.5491, IC95 % [0.319, 0.788];
- **Tiny RF:** AUC=0.4055, IC95 % [0.215, 0.616] (`docs/audit/btc_intradomain_tinybaseline.json`).

En los tres casos el intervalo de confianza **cruza 0.5**: el rendimiento es **estadísticamente
compatible con el azar**. Este es, metodológicamente, el resultado más importante del trabajo:
cuando se elimina el confound, la señal de detección de tumor desaparece. Conviene formularlo
con precisión: con n=36 y T1-only, lo correcto es afirmar «no detectamos señal por encima del
azar», no «no existe señal»; pero el contraste con el AUC≈1.0 del *pool* es inequívoco.

### 7.6. Evidencia en el espacio latente y mapas de atención

El análisis del vector latente de 96 dimensiones (capa previa al clasificador) cierra la
demostración. El *silhouette* es mayor por etiqueta (0.754) que por dataset (0.366), pero la
prueba decisiva es **intra-clase**: un clasificador lineal distingue IXI de NKI —**ambos
sanos**— con CV-AUC=0.998, y BraTS de UPENN —ambos tumor— con CV-AUC=0.991
(`docs/audit/embeddings_intraclass.json`). Como entre dos cohortes de la misma clase no hay
diferencia de etiqueta que aprender, esa separabilidad solo puede provenir de la firma del
centro. Coherentemente, un clasificador de las cuatro cohortes desde los *embeddings* alcanza
acc=0.982 (azar 0.25), y la proyección t-SNE (Figura 2) muestra cuatro islas en lugar de dos.
De forma complementaria y cualitativa, los mapas Grad-CAM (Anexo C) no localizan de manera
consistente una lesión, sino patrones difusos y periféricos compatibles con atención a
características globales del volumen.

### 7.7. Síntesis consolidada

Reunidas las cinco líneas de evidencia (Figura 1, `auc_summary.png`), el rendimiento inicial
se explica mejor por *shortcut learning* / *dataset bias* que por detección tumoral robusta:
(i) un modelo lineal trivial iguala a la CNN; (ii) el dominio es identificable con acc≈0.98
desde intensidad y desde el latente; (iii) la generalización fuera de dominio cae al azar e
incluso invierte la regla; (iv) la única evaluación con clase y dominio desacoplados sitúa el
AUC en torno a 0.4–0.55 con IC95 % que cruzan 0.5; y (v) el espacio latente codifica la
procedencia incluso dentro de una misma clase. El AUC≈1.0 del *pool* no es, por tanto,
evidencia de detección de tumor.

---

## 8. Discusión central (Tarea 9) — texto para la memoria

### 8.1. Interpretación: el confound estructural dominio↔clase

El origen del rendimiento aparente es una propiedad de diseño de los datos, no un fallo de
entrenamiento. En el *pool*, los positivos provienen exclusivamente de BraTS y UPENN y los
negativos de IXI y NKI; ningún dataset aporta ambas clases
(`label_equals_dataset`, `docs/audit/audit_leakage.json`). En consecuencia,
`P(tumor | dataset)` es 0 o 1 de forma determinista y «clase» y «dominio» son variables
**perfectamente confundidas**. Cualquier característica que identifique el centro de origen
—firma de intensidad, sesgo de campo residual, resolución, pipeline de adquisición— es un
predictor perfecto de la etiqueta. El modelo, como cualquier optimizador, explota el atajo
disponible más fácil.

### 8.2. Por qué las métricas casi perfectas no son evidencia clínica

Un AUC de 0.99997 sería extraordinario si midiera detección de tumor; aquí mide capacidad de
reconocer el dataset de origen. Tres hechos lo prueban: un modelo lineal sin anatomía iguala
a la CNN (AUC=1.0); la varianza intra-dataset de las puntuaciones es prácticamente nula y el
AUC intra-dataset es no calculable; y la separabilidad persiste **dentro** de una misma clase
en el espacio latente (IXI vs NKI, CV-AUC 0.998). Ninguna de estas observaciones es compatible
con un detector que generalice. Por tanto, las métricas del *pool* no deben presentarse como
evidencia de validez clínica, y este trabajo no lo hace.

### 8.3. Viabilidad y condiciones para un triaje radiológico válido

El sistema **no es desplegable hoy** como detector de tumor. Una afirmación clínica exigiría,
como mínimo: cohortes en las que ambas clases coexistan dentro de cada dominio/centro; un
tamaño muestral sustancialmente mayor que el de la única validación honesta disponible
(BTC, n=36); validación externa real (no usada en entrenamiento); separación estricta por
paciente; y estudios de calibración de la probabilidad de salida. Hasta satisfacer estas
condiciones, el rendimiento medido no puede interpretarse como capacidad de priorización
clínica.

### 8.4. Relación con *shortcut learning*, *dataset bias* y *domain shift*

El fenómeno observado es un caso de manual de *shortcut learning*: la red aprende una
correlación espuria, fácil y predictiva en distribución, que no se sostiene fuera de ella.
Encaja con la literatura de *dataset bias* y de confounding por centro/hospital en imagen
médica —donde se ha documentado que modelos aparentemente excelentes aprenden marcadores del
sitio de adquisición en lugar de la patología— y con el problema de *domain shift* en
neuroimagen multicéntrica. El valor del presente trabajo es **medir** el efecto en un caso
concreto y con un protocolo replicable, no solo describirlo. *(Citas a localizar y verificar
antes de incluir: Geirhos et al. 2020; Zech et al. 2018; DeGrave et al. 2021; literatura de
harmonización tipo ComBat. No inventar DOIs.)*

### 8.5. Validez metodológica y reproducibilidad

El protocolo se diseñó para ser defendible: *seed* fija (42), partición agrupada por sujeto,
intervalos de confianza por *bootstrap* en los experimentos intra-dominio y una batería de
auditoría documentada y ejecutable (`src/audit/`). Debe declararse con honestidad una
limitación de reproducibilidad: `recreate_splits:true` en `configs/train_3d.yaml` regenera la
partición en cada ejecución, y de hecho `data/splits.json` quedó sobrescrito por el último run
de LODO; el split original del *pool* se conserva aparte en
`data/splits_confound_original.json`. Congelar el *split* (fijar `recreate_splits:false` y
versionar el JSON) es un paso pendiente para reproducibilidad exacta.

### 8.6. Implicaciones para el diseño de estudios con datos públicos multi-fuente

Este caso ofrece recomendaciones concretas para cualquier estudio que combine datasets
públicos de RM (o de imagen médica en general):

1. **Clases dentro del mismo dominio.** Garantizar que cada centro/dataset aporte positivos
   y negativos; nunca asignar una clase por dataset completo.
2. **Controles patológicos, no solo sanos.** Incluir controles con otras patologías o
   hallazgos, no únicamente sujetos sanos, para evitar que «enfermo» se confunda con «centro
   de pacientes».
3. **Validación externa real.** Reservar al menos un centro no visto en entrenamiento; no
   reconvertir el test externo en entrenamiento.
4. **Separación estricta por paciente.** Agrupar por `subject_id` en todas las particiones.
5. **Auditorías anti-*shortcut* obligatorias**, antes de reportar métricas:
   - *baseline* trivial (estadísticos de intensidad sin red);
   - análisis de identificabilidad del dominio;
   - validación cruzada por dominio (*leave-one-dataset-out*);
   - evaluación intra-dominio cuando exista;
   - análisis del espacio latente (separabilidad por origen, incluso intra-clase);
   - estudios de calibración de la salida.

Presentadas en conjunto, estas prácticas constituyen el **protocolo reproducible de auditoría
anti-confounding** que este TFG propone como contribución metodológica.

---

## 9. Conclusiones (Tarea 10) — texto para la memoria

> Claras y no defensivas, tal como pidió el tutor.

Este trabajo **no valida un sistema clínico de triaje** de RM cerebral. Lo que demuestra es
que, bajo una formulación multi-fuente aparentemente razonable, el rendimiento de un
clasificador 3D puede estar **completamente inflado por sesgo de dominio**. Las conclusiones
principales son:

1. **El modelo inicial alcanza métricas casi perfectas** en el *pool* multi-fuente
   (AUC≈0.99997, sensibilidad 0.994, especificidad 1.0), pero **estas métricas no constituyen
   evidencia de detección de tumor**.
2. **El rendimiento está explicado por un confound estructural dominio↔clase**: los positivos
   y los negativos proceden de datasets disjuntos, de modo que reconocer el origen equivale a
   acertar la etiqueta. Un *baseline* lineal sin anatomía iguala a la CNN (AUC=1.0), el
   dominio es identificable con acc≈0.98, la generalización fuera de dominio cae al azar e
   incluso se invierte (LODO B AUC=0.20, sensibilidad 0.01), y la única validación honesta
   (BTC intra-dominio) sitúa el AUC en 0.40–0.55 con IC95 % que cruzan el azar.
3. **El principal valor del TFG es demostrar lo anterior de forma reproducible.** La
   contribución no es un detector, sino un **protocolo de auditoría anti-confounding** y la
   evidencia experimental que lo sostiene, transferible a otros estudios con datos públicos
   multi-fuente.
4. **Para obtener un sistema clínicamente válido** harían falta cohortes equilibradas con
   ambas clases por dominio, controles intra-dominio (incluidos controles patológicos),
   validación externa real, separación por paciente y estudios de calibración.

En suma, el trabajo convierte un resultado inicial sospechosamente bueno en una aportación
metodológica sólida: la detección, cuantificación y documentación de un sesgo de dominio que,
de no auditarse, se habría presentado como un éxito clínico inexistente.

---

## 10. Correo de respuesta al tutor (Tarea 11)

> Tono natural, académico, agradecido y breve. Listo para copiar.

---

**Asunto:** Índice ajustado + figuras y tabla resumen de resultados

Hola Jorge,

Gracias por la revisión tan detallada. He incorporado todas tus recomendaciones al índice y
ya puedo empezar a redactar con el enfoque aprobado.

En concreto:

- **1.3** pasa a titularse «De la hipótesis clínica inicial a la auditoría metodológica del
  modelo».
- **Objetivo general** reformulado para no prometer validación clínica: «Desarrollar un
  pipeline de clasificación 3D para RM cerebral y evaluar críticamente su validez en un
  escenario multi-fuente, identificando y cuantificando el impacto del sesgo de dominio en
  la tarea de priorización tumoral».
- **Estado del arte:** el apartado de *shortcut learning* / *dataset bias* / *domain shift*
  pasa a ser un pilar teórico (3.5), no algo secundario.
- **Materiales (4.3):** incluyo la matriz clase↔dataset que adjunto, dejando claro que cada
  dataset aporta una sola clase.
- **Cap. 6:** el protocolo de auditoría se presenta como metodología propia («Protocolo de
  auditoría de validez experimental»).
- **Cap. 7:** retitulado «Resultados: análisis progresivo de validez del rendimiento
  aparente».
- **Discusión:** añado el 8.6 «Implicaciones para el diseño de estudios con datos públicos
  multi-fuente».
- **Conclusiones:** redactadas sin suavizar, en la línea que propusiste.

**Adjunto:**

1. **Matriz dataset/clase** (BraTS→tumor, UPENN→tumor, IXI→sano, NKI→sano, BTC_preop→ambas).
2. **Tabla resumen de resultados**, marcando qué es aparentemente bueno pero no válido.
3. **Las cuatro figuras principales**, que siguen la narrativa en cascada:
   (1) AUC aparente, (2) baseline trivial sobre intensidad, (3) LODO en ambas direcciones,
   (4) BTC intra-dominio; más la t-SNE de *embeddings* como evidencia del espacio latente.

Sobre los **mapas Grad-CAM**: los dejo en anexo. Tras revisarlos, son cualitativos y de baja
resolución y no localizan la lesión de forma limpia, así que prefiero no darles peso de
resultado principal; si lo ves de otro modo, los subo al cuerpo sin problema.

Salvo que veas algo más, empiezo a redactar con este enfoque.

Un abrazo,
Nacho

---

## 11. Bloques listos para memoria (Tarea 12)

> Texto directamente reutilizable. Cada bloque es autocontenido.

### 11.1. Introducción del giro metodológico (para §1.3)

> El presente trabajo partió de una hipótesis clínica razonable: determinar si una red
> neuronal convolucional 3D puede clasificar estudios de RM cerebral como tumor / no tumor a
> nivel de estudio, con vistas a la priorización de listas de trabajo en radiología. El
> sistema desarrollado alcanzó, sobre un *pool* multi-fuente, un rendimiento prácticamente
> perfecto (AUC≈0.99997). Lejos de cerrar el trabajo, ese resultado lo reorientó: una métrica
> tan próxima al máximo teórico, en un problema clínico difícil y con datos heterogéneos, es
> en sí misma motivo de sospecha. La pregunta de investigación evolucionó entonces de «¿puede
> la red detectar tumor?» a «¿ese rendimiento refleja detección de tumor o un artefacto del
> diseño de los datos?». Esta reformulación no es un cambio de rumbo improvisado, sino una
> evolución guiada por la evidencia experimental, y constituye la contribución central de la
> memoria: el diseño, la aplicación y la documentación de un protocolo reproducible de
> auditoría de validez que detecta y cuantifica un sesgo de dominio.

### 11.2. Objetivo general (para §2.1)

> Desarrollar un *pipeline* de clasificación 3D para resonancia magnética cerebral y evaluar
> críticamente su validez en un escenario multi-fuente, identificando y cuantificando el
> impacto del sesgo de dominio en la tarea de priorización tumoral.

### 11.3. Explicación de la matriz dataset↔clase (para §4.3)

> El *pool* de entrenamiento y evaluación principal se compone de cuatro datasets públicos:
> los positivos (tumor) proceden de BraTS 2021 (n=580) y UPENN-GBM (n=587); los negativos
> (sano) de IXI (n=577) y NKI Rockland (n=523), sumando 2267 estudios (1167 positivos /
> 1100 negativos). La característica determinante de esta composición es que **ningún dataset
> aporta las dos clases**: la probabilidad de tumor condicionada al dataset es 0 o 1 de forma
> determinista (`label_equals_dataset` en `docs/audit/audit_leakage.json`). En consecuencia,
> la clase y el dominio de origen quedan **perfectamente confundidos**, y las tareas «detectar
> tumor» y «reconocer el centro de procedencia» resultan indistinguibles en cualquier métrica
> calculada sobre el *pool*. Esta propiedad no es un defecto de implementación, sino un rasgo
> estructural de los datos, y es la hipótesis explicativa que el resto del trabajo somete a
> contraste. El único conjunto que rompe esta confusión es BTC_preop (Ghent University
> Hospital, OpenNeuro ds001226), que aporta pacientes y controles del mismo centro y permite,
> por tanto, la única medición honesta del rendimiento.

### 11.4. Justificación del protocolo de auditoría (para §6.3)

> Ante un rendimiento aparente incompatible con la dificultad del problema, la validación
> interna habitual es insuficiente: puede certificar que el modelo reproduce en *test* lo
> aprendido en *train* sin detectar que ambos comparten el mismo atajo. Por ello se diseñó un
> protocolo de auditoría de validez en forma de embudo, que progresa desde lo trivial hacia lo
> estructural: (i) un *baseline* trivial que comprueba si la etiqueta es decodificable sin red
> ni anatomía; (ii) un análisis de identificabilidad del dominio, que mide si el centro de
> origen es separable desde intensidad y desde el espacio latente; (iii) una validación
> cruzada por dominio (*leave-one-dataset-out*) que rompe la coincidencia clase↔origen; (iv)
> una validación intra-dominio sobre el único conjunto con ambas clases en el mismo centro;
> (v) un análisis del espacio latente que evalúa si la representación interna codifica
> patología o procedencia; y (vi) mapas de atención como apoyo cualitativo. El conjunto
> constituye una metodología propia, reproducible y transferible a otros estudios con datos
> públicos multi-fuente.

### 11.5. Interpretación del AUC ≈ 1.0 (para §7.1 / §8.2)

> El valor de AUC obtenido en el *pool* multi-fuente (0.99997) no debe interpretarse como
> evidencia de un detector de tumor casi ideal. Tres observaciones lo impiden. Primera: las
> puntuaciones de salida se separan de forma casi degenerada por dataset de origen (UPENN
> 0.9995±0.0005, BraTS 0.986±0.098, IXI 0.0070±0.018, NKI 0.0008±0.0005), con varianza
> intra-dataset prácticamente nula y AUC intra-dataset no calculable al ser cada dataset
> mono-clase. Segunda: un modelo lineal entrenado únicamente sobre 16 estadísticos de
> intensidad, sin red ni información espacial, iguala a la CNN (AUC=1.0). Tercera: el espacio
> latente de la red separa cohortes incluso dentro de una misma clase (IXI frente a NKI, ambas
> sanas, con CV-AUC=0.998). El AUC≈1.0 mide, por tanto, la capacidad de reconocer el dominio
> de procedencia, no la presencia de lesión.

### 11.6. Discusión del confound (para §8.1)

> El mecanismo subyacente es un confound estructural entre la clase y el dominio de origen.
> Como los positivos y los negativos provienen de datasets disjuntos, cualquier característica
> que identifique el centro —firma de intensidad, sesgo de campo residual, resolución efectiva
> o particularidades del *pipeline* de adquisición— actúa como predictor perfecto de la
> etiqueta. La optimización por descenso de gradiente explota el atajo más accesible, que en
> este caso no requiere modelar anatomía tumoral. La evidencia es convergente: el atajo es
> visible en píxeles crudos (estadísticos de intensidad separables por dataset), es explotable
> por un modelo trivial (AUC=1.0), no transfiere fuera de dominio (LODO A 0.624; LODO B 0.201,
> con la regla invertida y la sensibilidad colapsada a 0.01), desaparece al desacoplar clase y
> dominio (BTC intra-dominio, IC95 % que cruzan 0.5) y queda inscrito en la representación
> interna de la red (separabilidad de dominio intra-clase). En conjunto, el rendimiento inicial
> se explica mejor por *shortcut learning* que por detección tumoral robusta.

### 11.7. Conclusión final (para §9)

> Este trabajo no valida un sistema clínico de triaje, sino que demuestra que, bajo una
> formulación multi-fuente aparentemente razonable, el rendimiento de un clasificador 3D de RM
> cerebral puede estar completamente inflado por sesgo de dominio. La principal contribución es
> el protocolo de auditoría anti-confounding y la evidencia experimental que lo sostiene: un
> *baseline* trivial que iguala a la red, la identificabilidad del dominio desde intensidad y
> desde el espacio latente, el desplome y la inversión del rendimiento fuera de dominio, y una
> validación intra-dominio compatible con el azar. Convertir un resultado sospechosamente
> bueno en una demostración rigurosa de su invalidez es, precisamente, el aporte metodológico
> del trabajo, y deja trazado el camino —cohortes equilibradas por dominio, controles
> patológicos, validación externa real, separación por paciente y calibración— hacia un
> sistema que sí pudiera sostener una afirmación clínica.

---

*Documento generado como material de redacción. Cifras verificadas contra los archivos del
repositorio (rutas citadas). Referencias bibliográficas pendientes de localizar y verificar
por el autor (no inventar DOIs).*
