# Borrador de memoria — auditoría de fiabilidad y experimento honesto

> Documento generado tras la fase de auditoría del 2026-05-28 a 2026-05-31.
> Pensado como columna vertebral para los capítulos de **Resultados** y
> **Discusión** de la memoria. Las cifras vienen todas de ejecuciones reales
> con seed fija y trazables a sus runs y JSONs.

---

## Resumen ejecutivo

El sistema CNN 3D entrenado sobre la mezcla multi-fuente (BraTS + UPENN-GBM
como positivos y IXI + NKI Rockland como negativos) alcanza **AUC ≈ 1.0** en
test, resultado **incompatible con detección clínica realista de tumor a partir
de RM cerebral**. Se diseñó y ejecutó una **auditoría sistemática** que
demuestra que dicho rendimiento se debe a un **confound estructural de
dominio**: la etiqueta está perfectamente correlacionada con el dataset de
origen (`label == 1 ⟺ dataset ∈ {BraTS, UPENN}`). Las pruebas realizadas (1)
una *tiny baseline* lineal sobre estadísticos de intensidad obtiene **AUC 1.0**
sin red neuronal ni información espacial; (2) un test de generalización
cross-dataset (LODO) hace caer la CNN a **AUC 0.62** en una dirección y
**0.20** en la opuesta (con inversión de signo); y (3) un experimento honesto
intra-dominio sobre BTC_preop (ds001226), donde tumor y sano provienen del
mismo escáner, sitúa el rendimiento de la CNN y de la *tiny baseline* en torno
al azar (**AUC 0.40 IC95% [0.21, 0.62]** y **0.55 IC95% [0.32, 0.79]**
respectivamente). Concluimos que el AUC ≈ 1.0 inicial era **íntegramente
atribuible al confound** y no constituye evidencia de capacidad de detección
de tumor.

---

## 1. Tabla maestra de resultados

| # | Experimento | Modelo | Modalidades | Test AUC | IC95% / Notas |
|---|---|---|---|---|---|
| 1 | Mezcla aleatoria de 4 datasets | Tiny baseline LogReg (16 features intensidad) | T1+T2 | **1.0000** | atajo trivial |
| 2 | Mezcla aleatoria de 4 datasets | Tiny baseline RandomForest | T1+T2 | 0.9989 | idem |
| 3 | Mezcla aleatoria de 4 datasets | **CNN 3D 2-canales (modelo del TFG)** | T1+T2 | **1.0000** | confounded |
| 4 | LODO A: BraTS+IXI → UPENN+NKI | Tiny baseline LogReg | T1+T2 | 0.9952 | transfer espurio |
| 5 | LODO A | **CNN 3D 2-canales** | T1+T2 | **0.6236** | sen 0.68 / spe 0.46 |
| 6 | LODO B: UPENN+NKI → BraTS+IXI | Tiny baseline LogReg | T1+T2 | 0.3184 | invertido |
| 7 | LODO B | **CNN 3D 2-canales** | T1+T2 | **0.2012** | sen 0.01 / spe 0.97 (colapso) |
| 8 | **Ghent intra-dominio** (n=36, 5-fold) | Tiny baseline LogReg | T1 | **0.5491** | **IC95% [0.32, 0.79]** — cruza 0.5 |
| 9 | Ghent intra-dominio | Tiny baseline RandomForest | T1 | 0.4055 | IC95% [0.22, 0.62] |
| 10 | **Ghent intra-dominio** | **CNN 3D 1-canal** | T1 | **0.4036** | **IC95% [0.21, 0.62]** — al azar |

> **Filas en negrita**: las que conviene reportar en la memoria como cifras
> de cabecera. Filas 3, 5, 7 y 10 son las cuatro lecturas críticas.

---

## 2. La narrativa en cuatro pasos

### Paso 1 — El resultado anómalo
Con el pipeline 2-canales (T1+T2) entrenado sobre el split aleatorio del pool
multi-fuente, se obtiene AUC = 1.000 en validación y AUC = 0.9999 en test
(`outputs/evaluation/cnn3d_test_results.json`, checkpoint
`outputs/checkpoints/20260527_152619/best.pt`). Sensibilidad 0.994,
especificidad 1.000. *Per dataset*, la varianza intra-dataset de los scores es
prácticamente nula (UPENN ±0.0005, NKI ±0.0005), lo que en la literatura suele
ser síntoma de detección de dominio más que de lesión. Resulta además
imposible calcular AUC intra-dataset porque cada fuente aporta una sola clase.

### Paso 2 — Demostración del confound (tiny baseline)
Se reduce cada volumen a 16 estadísticos de intensidad **no clínicos**
(fracción de vóxeles no-cero, media, desviación típica y percentiles 1/25/50/
75/99 de T1 y T2 sobre tejido) y se entrena una regresión logística con el
mismo split agrupado por sujeto. **AUC test = 1.0000**, accuracy 0.994
(`docs/audit/audit_leakage.json`). Es decir, la etiqueta es completamente
decodificable sin red neuronal y sin información espacial. La *feature* más
informativa es `nz_frac_t1` (importancia RF 0.17), una firma directa del
*skull-stripping*; le siguen percentiles de intensidad. Ninguna es clínica.

### Paso 3 — Test de estrés cross-dataset (LODO)
Para confirmar que la CNN no aprende tumor sino dominio, se reentrena
manteniendo intacta la arquitectura y los hiperparámetros pero **partiendo por
dataset**: tren+val sobre un par tumor+sano y test sobre los otros dos,
nunca vistos. Resultados (`outputs/evaluation/lodo_A|B/`):

- **LODO A** (BraTS+IXI → UPENN+NKI): val AUC 1.000 (dominios vistos), **test
  AUC 0.6236**. Sensibilidad 0.68, especificidad 0.46.
- **LODO B** (UPENN+NKI → BraTS+IXI): val AUC 1.000 desde la época 1, **test
  AUC 0.2012**. Sensibilidad 0.01, especificidad 0.97. **AUC < 0.5 = signo
  invertido**: el patrón de intensidad aprendido para "tumor" en UPENN apunta a
  "sano" cuando se aplica a BraTS.

La asimetría entre A y B y la inversión en B son **incompatibles con detección
real de tumor** y plenamente consistentes con un detector de firma de
adquisición/preprocesado. El paralelo con la *tiny baseline* (filas 4 y 6 de
la tabla maestra) refuerza la conclusión: incluso un modelo lineal sobre
intensidad reproduce el patrón.

### Paso 4 — Experimento honesto intra-dominio (BTC_preop)
Para obtener una cifra **no contaminada** se incorpora un dataset adicional,
**BTC_preop (OpenNeuro ds001226, Aerts et al., licencia CC0)**: 36 sujetos
escaneados en el mismo equipo (Ghent University Hospital), 25 pacientes con
tumor (glioma o meningioma) y 11 controles sanos. Preprocesado idéntico al
resto: HD-BET → reorient RAS → resample 1 mm³ → crop/pad 192×224×192 →
z-score Otsu. Como ds001226 no incluye T2, el experimento se hace **T1-only**
(1 canal); todos los demás componentes (modelo, hiperparámetros,
augmentation) son los del run confundido. Validación cruzada **5-fold por
sujeto, estratificada**, con agregación de predicciones e IC95% por bootstrap
(2000 resamples).

- **CNN 3D 1-canal**: **AUC 0.4036, IC95% [0.21, 0.62]** (cruza 0.5 → no
  significativo). Sensibilidad 0.60, especificidad 0.36 a umbral 0.5.
- **Tiny baseline LogReg**: AUC 0.5491, IC95% [0.32, 0.79] (también cruza 0.5).
- **Tiny baseline RandomForest**: AUC 0.4055, IC95% [0.22, 0.62].

Observación cualitativa de los scores por sujeto en cada fold: la red emite
valores prácticamente constantes en torno a 0.500 (variaciones en la 3ª-4ª
decimal). Es decir, con n=29 sujetos de entrenamiento por fold el modelo **no
extrae señal**, lo que se confirma por el plateau del *train loss* en ~0.43
a lo largo de las 20 épocas.

**Lectura conjunta de los pasos 1-4:** la totalidad del AUC ≈ 1.0 obtenido en
el escenario multi-fuente es atribuible al confound dominio↔clase. Cuando el
confound se elimina por construcción (mismo escáner, ambas clases), tanto el
modelo profundo como el modelo lineal triviales rinden indistinguiblemente del
azar al nivel de muestra disponible.

### Paso 5 — Evidencia en el espacio latente de la CNN (embeddings)
Para visualizar *qué* codifica internamente el modelo confundido se extrajo,
para los 2267 volúmenes, el vector de 96 dimensiones de la última capa
convolucional (salida de `AdaptiveAvgPool3d`, antes del clasificador) y se
proyectó a 2D mediante PCA y t-SNE
(`docs/audit/figures/embeddings_{pca,tsne}.png`). Se cuantificó la compacidad
del agrupamiento con el coeficiente *silhouette*
(`docs/audit/embeddings_silhouette.json`):

- *silhouette* por **etiqueta** = 0.754
- *silhouette* por **dataset** = 0.366

La separación por clase es nítida (0.754); sin embargo, dado que
`clase ≡ dataset`, esa separación **no permite distinguir** si el modelo
codifica "tumor" o "procedencia", y **no debe interpretarse como evidencia de
capacidad de detección**. El *silhouette* global por dataset (0.366) está
parcialmente diluido porque agrega la separación inter-clase con la
intra-clase; para aislar la huella de procedencia es necesario el análisis
refinado que sigue.

**Análisis refinado: separabilidad de datasets DENTRO de cada clase.** La
pregunta diagnóstica precisa es si la red distingue dos cohortes de la *misma*
clase. Se mide entrenando un clasificador (regresión logística, validación
cruzada 5-fold) sobre los embeddings, restringido a cada clase
(`docs/audit/embeddings_intraclass.json`):

| Pregunta | *silhouette* | LogReg CV-AUC |
|---|---|---|
| IXI vs NKI (ambos **sanos**) | 0.561 | **0.998** |
| BraTS vs UPENN (ambos **tumor**) | 0.186 | **0.991** |
| Clasificar el dataset (4 clases) desde embeddings | — | **acc 0.982** (azar 0.25) |

Dados dos sujetos **igualmente sanos**, uno de IXI y otro de NKI, el espacio
latente de la red permite identificar su cohorte de origen con AUC = 0.998 —
una capacidad que **nada tiene que ver con la presencia de tumor**. La
representación interna funciona, en la práctica, como un identificador del
centro de adquisición (clasificación de dataset al 98 % frente a un 25 % de
azar). Esto aísla la huella de procedencia que el *silhouette* global, dominado
por la separación de clase, enmascaraba. La menor separabilidad BraTS-UPENN
(0.186) frente a IXI-NKI (0.561) es coherente con que BraTS y UPENN comparten
un preprocesado de tipo *challenge*, mientras IXI y NKI son cohortes de
investigación con protocolos más dispares.

> **Nota de honestidad metodológica.** Se reportan tanto el *silhouette* global
> (0.366, diluido) como el análisis intra-clase (limpio). El segundo no
> sustituye al primero: lo explica. La medición intra-clase es la pregunta
> correctamente planteada desde el inicio (¿se separan cohortes de la misma
> clase?), no una métrica seleccionada a posteriori por conveniencia.

### Pasos auxiliares — confound en datos crudos y matrices de confusión
- **Distribución de intensidades por dataset**
  (`docs/audit/figures/intensity_by_dataset.png`): los cuatro datasets son
  separables ya en estadísticos triviales (fracción de vóxeles no-cero, media
  y percentiles de intensidad), confirmando que el confound existe **antes**
  de entrar al modelo, en los propios píxeles. Esto explica por qué la *tiny
  baseline* lineal del Paso 2 alcanza AUC 1.0.
- **Matrices de confusión por régimen**
  (`docs/audit/figures/confusion_matrices.png`): comparan visualmente
  confounded (diagonal perfecta), LODO A, LODO B (colapso/inversión) y Ghent
  intra-dominio (dispersión propia del azar).

---

## 3. Hipótesis alternativas descartadas

Las siguientes hipótesis se evaluaron y se descartan con evidencia
documentada:

- **Leakage por partición**: la auditoría de `data/splits.json`
  (`src/data/audit_splits.py`) muestra **0 solapamiento de subject_id** entre
  train, val y test. Estratificación por (dataset, label) controlada y
  `seed=42` fijo.
- **Duplicados exactos cruzando splits**: `src/audit/audit_leakage.py`
  computa SHA-1 de los arrays t1+t2 de cada fichero y detecta **0 grupos
  duplicados** cruzando train/val/test.
- **Error de cálculo de métricas**: `src/evaluation/evaluate_3d.py` calcula
  AUC sobre probabilidades sigmoides (no clases), preserva el orden
  `y_true`/`y_pred` del DataLoader y reporta sensibilidad/especificidad
  /PR-AUC con confusion matrix completa.
- **Leakage de test en el entrenamiento**: el checkpoint se selecciona por
  balanced_accuracy sobre val con sensibilidad mínima 0.80; el test se toca
  una sola vez al final del entrenamiento (`src/training/train_3d.py`,
  líneas 417-431).

---

## 4. Limitaciones que se reportan con honestidad

1. **Tamaño muestral del experimento intra-dominio**: n = 36 sujetos (25
   tumor / 11 control) es suficiente para *cuantificar* la magnitud del
   confound pero **insuficiente para una afirmación clínica positiva**. Los
   IC95% sobre el AUC son anchos y cruzan 0.5. No podemos concluir "el modelo
   no detecta tumor", sólo "con estos datos no detectamos tumor por encima
   del azar".
2. **Modalidad reducida (T1-only) en el experimento honesto**: ds001226 no
   incluye T2 estructural. Se discute en §6.1 la pérdida de información
   asociada y las vías para recuperarla.
3. **Ausencia de validación externa real**: la cohorte original prevista
   como conjunto externo (UPENN-GBM) se incorporó al pool de entrenamiento
   por decisión metodológica explícita (ver `docs/evolucion_respecto_primera_entrega.md §1.2`).
   Hasta disponer de otra cohorte con tumor y sano del mismo dominio (p. ej.,
   Edinburgh SN-851861, requiere solicitud) o de tamaño superior a las
   docenas, la validez externa permanece pendiente.

---

## 5. Frases listas para incorporar al cuerpo de la memoria

> *"El presente trabajo formula la priorización automática de RM cerebrales
> como una tarea de clasificación binaria a nivel de estudio. Al evaluar el
> sistema sobre el pool multi-fuente construido (BraTS 2021, UPENN-GBM, IXI y
> NKI Rockland), se obtuvieron métricas de rendimiento incompatibles con la
> dificultad clínica del problema (AUC = 0.9999, sensibilidad = 0.994,
> especificidad = 1.000). Este resultado motivó una auditoría sistemática
> cuyo objetivo era determinar si las cifras reflejaban detección de lesión o
> alguna forma de fuga metodológica."*

> *"La auditoría confirmó la existencia de un confound estructural: la
> etiqueta de clase coincide en grado del 100 % con el dataset de origen
> (P(tumor | BraTS o UPENN) = 1, P(tumor | IXI o NKI) = 0). Una prueba de
> control consistente en entrenar una regresión logística sobre 16
> estadísticos de intensidad no clínicos alcanza AUC = 1.0000 sobre el mismo
> conjunto de test, lo que demuestra que la etiqueta es decodificable sin
> red neuronal ni información anatómica. Por lo tanto, las métricas
> aparentemente perfectas de la CNN no pueden interpretarse como evidencia
> de detección de lesión tumoral."*

> *"Para cuantificar la dependencia del modelo respecto al dataset de
> origen se realizó una prueba de generalización cross-dataset bajo el
> esquema leave-one-dataset-out por clase. Entrenando con BraTS+IXI y
> evaluando en UPENN+NKI, el AUC en test cae a 0.6236; en la dirección
> opuesta (entrenando con UPENN+NKI, evaluando en BraTS+IXI) el AUC se
> sitúa en 0.2012, valor inferior al azar e interpretable como inversión de
> signo de la regla de decisión. Estas asimetrías son inconsistentes con
> un detector de lesión genuino y consistentes con un clasificador de
> firma de adquisición/preprocesado."*

> *"Como evaluación honesta del rendimiento de detección de tumor con el
> confound de dominio controlado, se incorporó el dataset BTC_preop
> (Aerts et al., OpenNeuro ds001226, licencia CC0), que contiene 25
> pacientes con tumor y 11 controles adquiridos en el mismo equipo de RM.
> El pipeline se reentrena en T1 (única modalidad disponible) con
> validación cruzada 5-fold a nivel de sujeto. El AUC agregado es
> 0.4036, con intervalo de confianza al 95% [0.21, 0.62] obtenido por
> bootstrap (2000 resamples). El intervalo cruza el valor 0.5, lo que
> indica que el rendimiento es indistinguible del azar al tamaño muestral
> disponible. Una prueba de control con regresión logística sobre los
> mismos estadísticos de intensidad sitúa el AUC en 0.5491 [0.32, 0.79],
> igualmente al azar. Concluimos que el AUC ≈ 1.0 obtenido en el pool
> multi-fuente era atribuible íntegramente al confound de dominio."*

> *"El trabajo aporta, por tanto, una caracterización metodológica
> reproducible de un caso de aprendizaje espurio en clasificación 3D de RM
> cerebral, junto con un protocolo de auditoría (tiny baseline, LODO,
> validación intra-dominio) reutilizable para sistemas análogos. La
> validación clínica del sistema como herramienta de triaje queda
> pendiente de una cohorte intra-dominio de mayor tamaño y, deseablemente,
> multi-céntrica con ambas clases representadas en cada centro."*

> *"El análisis del espacio latente de la red refuerza esta conclusión: a
> partir del vector de características de la última capa convolucional, un
> clasificador lineal distingue las dos cohortes de sujetos sanos (IXI frente a
> NKI Rockland) con AUC = 0.998, y predice el dataset de origen entre las cuatro
> fuentes con una exactitud del 98 % (frente a un 25 % de azar). Es decir, la
> representación aprendida codifica la procedencia del estudio con independencia
> de la presencia de lesión, lo que constituye evidencia directa, a nivel de
> representación interna, del confound de dominio identificado."*

---

## 5-bis. Inventario de figuras para la memoria

Todas en `docs/audit/figures/`. La columna "mensaje" sirve como pie de figura.

| Figura | Mensaje que transmite |
|---|---|
| `auc_summary.png` | El AUC se desploma de ~1.0 a azar al eliminar el confound (barras con IC95%). Figura resumen. |
| `roc_curves.png` | Confounded pega al techo; LODO se aleja; intra-dominio cae a la diagonal. |
| `score_hist_confound.png` | El run confundido da scores bimodales extremos por dataset, no por contenido. |
| `score_hist_lodo.png` | En cross-dataset los scores se descolocan (LODO A y B). |
| `confusion_matrices.png` (E3) | Diagonal perfecta (confounded) → colapso/inversión (LODO) → dispersión (intra-dominio). |
| `btc_kfold_bars.png` | Intra-dominio: AUC por fold disperso en torno al azar, IC95% ancho. |
| `embeddings_tsne.png` (E1) | **Figura estrella**: el latente agrupa por procedencia; IXI y NKI (ambos sanos) no se fusionan. |
| `embeddings_pca.png` (E1) | Versión lineal del anterior, con varianza explicada. |
| `intensity_by_dataset.png` (E2) | El confound existe ya en los píxeles crudos (estadísticos de intensidad separables por dataset). |
| `gradcam/confound/*.png` | El mapa de atención cae en fondo/bordes, no en la lesión. |
| `gradcam/lodo_{A,B}/*.png` | Apoyo: a qué atiende el modelo cross-dataset. |

**Selección mínima si hay que recortar:** `auc_summary`, `embeddings_tsne`,
`confusion_matrices`, un `gradcam/confound`. Esas cuatro cuentan la historia
completa.

---

## 6. Anticipo de preguntas del tribunal

### 6.1 *"¿Por qué solo T1 en Ghent y no T1+T2?"*
ds001226 fue diseñado para *brain network modelling* y solo incluye T1 + DWI
+ rs-fMRI por sujeto, sin T2 estructural. Las alternativas con T1+T2 y ambas
clases dentro del mismo dominio son raras en datos públicos; la candidata
más sólida (Edinburgh, UK Data Service SN-851861) requiere registro y
aceptación del *End User License*, sin garantía de plazo. Optamos por
ejecutar el experimento honesto en T1-only para no bloquear la entrega y
declarar la extensión a T1+T2 como trabajo futuro inmediato.

### 6.2 *"¿No es muy pequeño n = 36 para una CNN 3D desde cero?"*
Sí. La consecuencia es un IC95% ancho y un modelo que esencialmente no
aprende (se observa plateau del *train loss* en ~0.43 y scores casi
constantes en torno a 0.5). Esto **no permite afirmar que el modelo no
detecte tumor**; permite afirmar que **con n = 36 no se observa señal por
encima del azar**, lo que es suficiente para concluir que el rendimiento
previo de AUC ≈ 1.0 era confound y no detección.

### 6.3 *"¿Por qué considerar AUC 0.40, que está por debajo de 0.5, como 'al azar'?"*
Porque el IC95% del bootstrap (`[0.21, 0.62]`) incluye 0.5 con holgura. Un
valor por debajo de 0.5 con IC95% que cruza 0.5 es estadísticamente
indistinguible del azar. La aparente "inversión" sugiere variabilidad por
muestreo, no señal real anti-correlacionada.

### 6.4 *"¿Han considerado domain-adversarial training o armonización?"*
Sí, brevemente. Se discuten en la sección de trabajo futuro. La razón por
la que **no resuelven el problema en este dataset** es estructural: si
domain ≡ label al 100 %, eliminar la información de dominio elimina
también la de etiqueta. Los métodos de adaptación de dominio asumen al
menos descorrelación parcial entre dominio y clase, que aquí no existe.

### 6.5 *"¿No habría bastado con un split por dataset desde el inicio?"*
No por sí solo. Un split por dataset (LODO) detecta el síntoma pero no
sana la patología: tanto train como test siguen teniendo `label = domain`,
por lo que el rendimiento alto en test cruzado puede deberse a transferencia
de firmas de familia-de-dominio (p. ej., los dos datasets de tumor comparten
características de adquisición que los diferencian de los dos sanos). La
única forma de evaluar detección de tumor sin confound es disponer de
ambas clases dentro del mismo dominio, que es lo que aporta el experimento
intra-dominio sobre BTC_preop.

---

## 7. Trabajo futuro inmediato

1. **Ampliar el experimento intra-dominio a T1+T2** con el dataset de
   Edinburgh (UK Data Service SN-851861, **acceso ya aprobado**): replicar la
   validación intra-dominio en un segundo dominio independiente y con las dos
   modalidades, condicionado a verificar que el depósito incluye los controles
   sanos además de los pacientes.
2. **Reunir o solicitar una cohorte intra-dominio de mayor tamaño** (objetivo
   n ≥ 100 con ambas clases en el mismo centro) para reducir la anchura del
   IC95%.
3. **Reportar leave-one-dataset-out completo** (configuraciones C y D del
   protocolo LODO) si se dispone de tiempo de cómputo.
4. **Calibración del clasificador final** (Platt scaling, reliability
   diagrams) sobre la cohorte intra-dominio cuando alcance tamaño suficiente.

---

## 8. Trazabilidad de las cifras (para defensa)

| Cifra | Fichero / run |
|---|---|
| Confounded CNN AUC 1.0 | `outputs/evaluation/cnn3d_test_results.json` (checkpoint `outputs/checkpoints/20260527_152619/best.pt`) |
| Tiny baseline mezcla AUC 1.0 | `docs/audit/audit_leakage.json` |
| LODO tiny baseline (4 configs) | `docs/audit/audit_lodo.json` |
| LODO A CNN AUC 0.6236 | `outputs/evaluation/lodo_A/cnn3d_test_results.json` |
| LODO B CNN AUC 0.2012 | `outputs/evaluation/lodo_B/cnn3d_test_results.json` |
| Ghent tiny baseline AUC + IC | `docs/audit/btc_intradomain_tinybaseline.json` |
| Ghent CNN AUC + IC + per-fold | `outputs/evaluation/btc_intradomain/cnn_kfold_results.json` |
| Embeddings + silhouette global (E1) | `docs/audit/embeddings.npz`, `docs/audit/embeddings_silhouette.json` |
| Silhouette intra-clase + clasificador de dataset desde embeddings | `docs/audit/embeddings_intraclass.json` (`src/audit/embeddings_intraclass.py`) |
| Clasificador de dataset acc 0.985 (C) | `docs/audit/audit_leakage.json` (`dataset_origin_classifier`) |
| Composición ds001226 | `data/raw/btc_preop/participants.tsv` (DOI 10.18112/openneuro.ds001226.v5.0.0) |
| Scripts auditoría | `src/audit/audit_leakage.py`, `audit_lodo.py`, `btc_tiny_baseline.py`, `btc_cnn_kfold.py`, `embeddings_tsne.py`, `make_extra_figures.py`, `make_plots.py`, `gradcam_3d.py` |

Todos los runs son reproducibles con `seed = 42` (split y entrenamiento) y
`seed = 0` (bootstrap de IC95% y proyecciones).

---

_Borrador generado el 2026-05-31 tras la finalización exitosa de la cadena
BTC. Las cifras son reales y se han verificado contra los JSONs originales._
