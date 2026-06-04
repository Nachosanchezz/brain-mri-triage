# Auditoría de resultados sospechosos — triaje de RM cerebral

> **Auditor:** revisión técnica de ML médico / metodología TFG.
> **Fecha:** 2026-05-28.
> **Objeto:** determinar si las métricas casi perfectas (AUC ≈ 1.0,
> sensitivity ≈ 0.994, specificity = 1.0, varianza intra-dataset ≈ 0)
> son defendibles o reflejan contaminación metodológica.
> **Run auditado:** `outputs/checkpoints/20260527_152619/best.pt`,
> evaluado sobre `data/splits.json` (test, n=340).

---

## 1. Resumen ejecutivo

**Los resultados NO son defendibles como evidencia de detección de tumor.**
El modelo está midiendo, en grado prácticamente total, **el dataset/protocolo
de origen de cada volumen, no la presencia de lesión.**

La causa no es un bug de partición ni duplicados (ambos descartados con
pruebas, ver §5 y §2). Es un **confound estructural en la composición del
dataset**: la etiqueta es *idéntica* al origen de los datos.

- **Positivos (tumor):** únicamente BraTS + UPENN-GBM.
- **Negativos (sano):** únicamente IXI + NKI Rockland.

Es decir, `label == 1 ⟺ dataset ∈ {brats, upenn}` y
`label == 0 ⟺ dataset ∈ {ixi, nki}`, con correlación del **100 %**.
Verificado por código:
`label_equals_dataset = {brats:[1], upenn:[1], ixi:[0], nki:[0]}`
(`docs/audit/audit_leakage.json`).

**Prueba definitiva (tiny baseline):** una **regresión logística** entrenada
sobre 16 estadísticos triviales de intensidad (fracción de vóxels no-cero,
media, std y percentiles de T1/T2) — **sin red neuronal, sin información
espacial, sin localizar ningún tumor** — alcanza:

| Modelo trivial | AUC test | Acc test |
|---|---|---|
| Regresión logística (16 features) | **1.0000** | 0.994 |
| Random Forest (16 features) | 0.9989 | 0.988 |

Si un modelo lineal sobre estadísticos globales de brillo separa las clases
perfectamente, entonces el AUC ≈ 1.0 de la CNN **no requiere detección de
tumor en absoluto**: la señal está en la firma de intensidad/preprocesado de
cada fuente.

**Clasificación de riesgo global: CRÍTICO.** Las métricas actuales no pueden
presentarse como rendimiento de triaje tumoral sin inducir a error.

---

## 2. Reconocimiento del proyecto y flujo del pipeline

Scripts principales identificados:

| Función | Fichero |
|---|---|
| Preprocesado común (RAS, resample, crop, z-score) | `src/preprocessing/base_preprocessing.py` |
| Preprocesado por dataset | `preprocess_brats.py`, `preprocess_ixi.py`, `preprocess_upenn.py`, `preprocess_nki_rockland.py` |
| Creación del split | `src/data/dataset_3d.py::create_splits` |
| Carga de datos / augmentation | `src/data/dataset_3d.py::BrainMRI3DDataset` |
| Entrenamiento | `src/training/train_3d.py` |
| Evaluación + métricas por dataset | `src/evaluation/evaluate_3d.py` |
| Análisis de umbral | `src/evaluation/threshold_analysis.py` |
| Auditoría de splits | `src/data/audit_splits.py` |
| Modelo | `src/models/cnn3d.py` |
| Config | `configs/train_3d.yaml` |

**Flujo:** `data/raw/*` → preprocesado (reorient RAS → resample 1 mm³ →
crop/pad a 192×224×192 → z-score sobre máscara de tejido) → `.npz` en
`data/processed/{positives,negatives}/` → `create_splits` genera
`data/splits.json` (70/15/15) → `BrainMRI3DDataset` recorta a 128×160×128 y
augmenta → CNN 3D → `evaluate_3d.py` produce métricas globales y por dataset.

**Composición real de los datos** (`data/processed/`):

| Carpeta | Dataset | n | label |
|---|---|---|---|
| positives | BraTS2021 | 580 | 1 |
| positives | UPENN-GBM | 587 | 1 |
| negatives | IXI | 577 | 0 |
| negatives | NKI Rockland | 523 | 0 |

→ 1167 positivos / 1100 negativos. **Ningún dataset aporta ambas clases.**

---

## 3. Auditoría de particiones

| Comprobación | Resultado | Evidencia |
|---|---|---|
| Nivel del split | Por fichero, agrupado por `subject_id` | `dataset_3d.py:93-165` |
| ¿Mismo sujeto en train y test? | **No.** `train∩val=0, train∩test=0, val∩test=0` | `audit_splits.py` (ejecutado) |
| `n_samples == n_subjects` | Sí en cada (split,dataset,label) → 1 fichero/sujeto | salida de `audit_splits` |
| `random_state` controlado | Sí, `seed=42`, `np.random.default_rng` | `dataset_3d.py:119` |
| Estratificación | Por **(dataset, label)** conjunta | `dataset_3d.py:120-135` |
| Split fijo vs recalculado | **Recalculado en cada run** (`recreate_splits: true`) pero determinista por seed | `configs/train_3d.yaml:17`, `train_3d.py:71-76` |

**Diagnóstico de splits: "No hay evidencia de leakage en splits."** La
partición está bien hecha (agrupada por sujeto, estratificada, sin
solapamiento, seed fijo). Riesgo *menor*: `recreate_splits: true` regenera el
split en cada ejecución; si se añaden/quitan datos el split cambia
silenciosamente. Recomendación: congelar `splits.json` y poner
`recreate_splits: false` para reproducibilidad (no es un leak, es higiene).

**Importante:** el split correcto **no salva** el experimento, porque el
problema no está en cómo se reparten los ficheros sino en qué representa la
etiqueta (§4).

---

## 4. Auditoría de sesgo de dataset / origen  ← causa raíz

**Diagnóstico: leakage de etiqueta CONFIRMADO por confound de dominio.**

Evidencias (todas reproducibles):

1. **Correlación clase ↔ dataset = 100 %.** Ningún dataset contiene las dos
   clases. Por eso `evaluate_3d.py` reporta `auc = NaN` por dataset en los
   cuatro (no se puede calcular AUC con una sola clase). *Esa imposibilidad de
   medir AUC intra-dataset es, en sí misma, la prueba de que el AUC global es
   un clasificador de dominio.*

2. **Scores agrupados por dataset, no por contenido**
   (`outputs/evaluation/cnn3d_test_results.json`):

   | Dataset | clase | score medio | std |
   |---|---|---|---|
   | upenn | tumor | 0.9995 | ±0.0005 |
   | brats | tumor | 0.9860 | ±0.098 |
   | ixi | sano | 0.0070 | ±0.018 |
   | nki | sano | 0.0008 | ±0.0005 |

   Varianza intra-dataset casi nula (UPENN ±0.0005, NKI ±0.0005): el modelo
   asigna prácticamente el mismo score a *todas* las imágenes de un dataset.
   Un detector de lesión real mostraría variabilidad (tumores sutiles vs.
   evidentes). Esto es comportamiento de **detector de dominio**.

3. **Tiny baseline (prueba B).** LogReg sobre 16 features de intensidad →
   **AUC test = 1.0000**. Features más informativas (RandomForest):
   `nz_frac_t1` (0.17), `p75_t1` (0.17), `p25_t2` (0.13), `nz_frac_t2` (0.10).
   La fracción de vóxels no-cero (firma de skull-stripping/preprocesado) y los
   percentiles de intensidad bastan para separar las clases. **Ninguna de esas
   variables es clínica.**

4. **Clasificador de origen (prueba C).** RandomForest prediciendo el dataset
   (4 clases, azar ≈ 25 %) → **acc test = 0.985**. El dominio es trivialmente
   identificable desde las mismas features.

**Conclusión §4:** el modelo puede alcanzar AUC ≈ 1.0 resolviendo
"¿esta imagen viene de BraTS/UPENN o de IXI/NKI?" en lugar de
"¿hay tumor?". Como ambas preguntas tienen la misma respuesta en estos datos,
las métricas no distinguen entre las dos hipótesis. **Riesgo: CRÍTICO.**

---

## 5. Auditoría de duplicados y similitud

Implementado en `src/audit/audit_leakage.py` (sha1 de los arrays t1+t2 por
fichero, comparado entre splits). Resultado:

- **0 grupos de duplicados exactos.**
- **0 duplicados cruzando splits.**

→ El leakage **no** procede de duplicados ni de copias del mismo estudio en
particiones distintas. Esta hipótesis queda **descartada limpiamente**.
CSV de features+hashes: `docs/audit/audit_features.csv`. **Riesgo: BAJO.**

---

## 6. Auditoría de métricas y evaluación

`src/evaluation/evaluate_3d.py` está, en lo técnico, **bien implementado**:

| Comprobación | Resultado |
|---|---|
| ¿Evalúa sobre test independiente? | Sí, `splits["test"]`, sin shuffle (`evaluate_3d.py:252-264`) |
| AUC usa probabilidades, no clases | Sí, `sigmoid(logits)` (`positive_probability`, l.86-91) |
| Alineación y_true / y_pred | Correcta (mismo orden de loader, l.200-210) |
| Sens/spec/PPV/F1/PR-AUC/conf-matrix | Correctas (l.135-162) |
| Threshold elegido en val, aplicado en test | Sí (`threshold_analysis.py`) |
| Checkpoint correcto | `20260527_152619/best.pt` (no `cnn3d_best.pt` por defecto; se pasó `--checkpoint`) |

**No hay errores de cálculo de métricas.** El cálculo es honesto; el problema
es **qué** se está midiendo (datos contaminados), no **cómo**.

Carencias menores (no inflan resultados, pero conviene para el TFG):
- No hay intervalos de confianza / bootstrap sobre AUC y sensitivity.
- `threshold=0.5` es defendible aquí solo porque los scores son bimodales
  extremos (0.00 vs 0.99) — pero esa bimodalidad es justamente síntoma del
  confound, no de buena calibración.

**Riesgo del código de evaluación: BAJO.** **Riesgo de interpretación: ALTO.**

---

## 7. Auditoría del entrenamiento

`src/training/train_3d.py`:

| Aspecto | Estado | Nota |
|---|---|---|
| Loss | `BCEWithLogitsLoss`, `pos_weight=1.0` | Clases ~equilibradas; correcto |
| Augmentation | flips + gamma + ruido sobre máscara de cerebro | `dataset_3d.py:202-232`; intenta reducir domain shift de intensidad pero NO rompe el confound |
| Early stopping | balanced_accuracy con `sen ≥ 0.80` | `train_3d.py:380-407`; criterio sensato |
| Selección de modelo | por **val**, no por test | Correcto, sin leak de test |
| ¿Se usó test para decidir? | **No** | test solo se toca al final (l.417-431) |
| Overfitting | val_auc = 1.0 desde épocas tempranas | No es overfitting clásico: val es separable porque val *también* tiene cada dataset mono-clase |

**No hay leakage de test en el entrenamiento.** La selección de modelo y el
early stopping son metodológicamente correctos. Insisto: el problema está
aguas arriba, en la composición de los datos. **Riesgo de entrenamiento: BAJO**
(salvo que hereda el confound de §4).

---

## 8. Pruebas negativas realizadas

| Prueba | Estado | Resultado | Lectura |
|---|---|---|---|
| **B. Tiny baseline** (LogReg/RF sobre intensidad) | ✅ ejecutada | AUC test **1.0000** / 0.999 | Etiqueta decodificable sin red ni señal clínica → contaminación confirmada |
| **C. Clasificador de dataset de origen** | ✅ ejecutada | acc **0.985** (4 clases) | Dominio trivialmente identificable → sesgo fuerte |
| **Duplicados / hashes** | ✅ ejecutada | 0 duplicados | Leakage no es por duplicados |
| **E. Patient-level split** | ✅ verificada | 0 overlap de sujeto | Split correcto (no es la causa) |
| **A. Label shuffle** | ⏳ propuesta | — | Ver nota |
| **D. Leave-one-dataset-out** | ⚠️ no aplicable tal cual | — | Ver nota |
| **F. Permutation sanity** | ⏳ propuesta | — | Equivalente a A |

Scripts: `src/audit/audit_leakage.py`. Salidas: `docs/audit/audit_leakage.json`,
`docs/audit/audit_features.csv`.

**Notas sobre A/D/F:**
- **Label shuffle / permutation (A, F):** con los datos actuales, barajar
  etiquetas *dentro de cada dataset* es imposible (cada dataset es mono-clase),
  así que el único shuffle posible rompe también la correlación con el dominio;
  el baseline tonto ya demuestra el punto de forma más fuerte. Vale la pena
  como confirmación pero el tiny baseline es concluyente.
- **Leave-one-dataset-out (D):** *no permite medir detección de tumor* aquí,
  porque al dejar fuera, p.ej., IXI, el modelo no ve negativos de ese dominio y
  no hay forma de separar "cambió el dominio" de "cambió la clase". LODO solo
  sería informativo si **cada dominio tuviera ambas clases**. Es justamente la
  carencia estructural del dataset.

---

## 9. Problemas enumerados y clasificados por gravedad

| # | Problema | Gravedad | Fichero / evidencia |
|---|---|---|---|
| 1 | **Confound clase ↔ dataset (label leakage de dominio)** | **CRÍTICO** | composición `data/processed/`; tiny baseline AUC=1.0 |
| 2 | Imposibilidad de medir AUC intra-dataset (todos NaN) | **ALTO** | `cnn3d_test_results.json` |
| 3 | Métricas casi perfectas presentadas sin caveat suficiente en resúmenes | ALTO | tablas de resultados |
| 4 | No hay evaluación externa cross-dataset (UPENN movido a train) | ALTO | `evolucion_respecto_primera_entrega.md §1.2` |
| 5 | `recreate_splits: true` → split no congelado | MEDIO | `configs/train_3d.yaml:17` |
| 6 | Sin intervalos de confianza / bootstrap | BAJO | `evaluate_3d.py` |
| 7 | Sin calibración | BAJO | declarado pendiente |
| 8 | Duplicados | NINGUNO (descartado) | `audit_leakage.json` |
| 9 | Leakage de split por sujeto | NINGUNO (descartado) | `audit_splits` |

---

## 10. Cambios propuestos (NO aplicados — requieren tu decisión)

> Por tu instrucción, **no he tocado el pipeline de entrenamiento/evaluación
> ni el preprocesado**. Solo he añadido scripts de auditoría nuevos
> (`src/audit/`). Lo siguiente son propuestas a debatir.

**Cambios mínimos (higiene, bajo coste):**
1. Congelar el split: `recreate_splits: false` y commitear `data/splits.json`.
2. Añadir bootstrap de IC95 % para AUC/sensitivity en `evaluate_3d.py`.
3. En todo informe/resumen, sustituir "AUC ≈ 1.0 detectando tumor" por la
   redacción honesta de §11.

**Cambios estructurales (los que de verdad arreglan el experimento):**
4. **Romper el confound** — la única solución real. Opciones, de mejor a peor:
   - **(a) Negativos del mismo dataset que los positivos.** BraTS y UPENN-GBM
     son todo-tumor, pero existen cohortes con sanos y tumor del mismo
     protocolo, o se pueden usar negativos "sin lesión" del mismo escáner.
     Esto permitiría por fin medir tumor *dentro* de un dominio.
   - **(b) Validación cross-dataset honesta:** entrenar y reportar de forma que
     el test provenga de fuentes no vistas *y con ambas clases*.
   - **(c) Si (a)/(b) son inviables a tiempo:** reformular el TFG como
     *"clasificación multi-fuente con confound de dominio reconocido"* y
     reportar el tiny-baseline como límite superior trivial (honestidad
     metodológica), sin afirmar capacidad de detección de tumor.
5. **Grad-CAM** sobre aciertos: si el mapa de atención no cae sobre la lesión
   sino sobre bordes/fondo, confirma visualmente el confound (material muy
   defendible en la memoria).

No recomiendo aplicar nada de §10 sin que decidas la vía (a/b/c), porque
cambia el alcance del TFG.

---

## 11. Criterio final

**¿Son los resultados actuales defendibles?**
No como detección de tumor. Son defendibles únicamente como
*"el modelo distingue las fuentes de datos"*, que es trivial y no era el
objetivo clínico.

**¿Hay indicios de data leakage?**
Sí: **leakage de etiqueta vía confound de dominio** (la clase coincide con el
dataset). No hay leakage de split ni de duplicados (descartados con pruebas).

**¿Hay indicios de sesgo de dataset?**
Sí, máximo posible: correlación clase↔origen del 100 %, clasificador de
dominio con acc 0.985, tiny baseline con AUC 1.0.

**¿Qué cambiarías antes de entregar el TFG?**
Como mínimo, declarar el confound de forma central (no como nota al pie) y
añadir el tiny baseline como evidencia. Idealmente, conseguir **negativos y
positivos del mismo dominio** para una medición honesta, aunque sea sobre un
subconjunto pequeño.

**Frase metodológica para la memoria (reconoce la limitación sin hundir el trabajo):**

> *"En la configuración actual del conjunto de datos, la clase positiva
> (tumor) procede exclusivamente de BraTS y UPENN-GBM y la negativa (control
> sano) de IXI y NKI Rockland, de modo que la etiqueta está perfectamente
> correlacionada con el dataset de origen. Una prueba de control con un
> clasificador lineal sobre estadísticos de intensidad no clínicos alcanza
> AUC = 1.0, lo que demuestra que las métricas casi perfectas obtenidas por la
> CNN 3D son atribuibles, en grado indeterminado, a la firma de
> adquisición/preprocesado de cada fuente y no pueden interpretarse como
> capacidad de detección de lesión. Por ello, los resultados se presentan como
> prueba de concepto del pipeline (preprocesado homogéneo, partición por
> sujeto, entrenamiento estable) y la validación clínica del poder
> discriminativo queda condicionada a una evaluación con ambas clases
> representadas dentro de cada dominio, que se plantea como trabajo
> inmediato."*

**Experimento mínimo a rehacer para una evaluación más honesta:**
Reunir un conjunto — aunque sea pequeño (p.ej. 40–60 sujetos) — en el que
**tumor y sano provengan del mismo dataset/protocolo** (mismo escáner, mismo
preprocesado), y reportar AUC/sensitivity/specificity *dentro de ese dominio*.
Si ahí el AUC cae sustancialmente respecto a 1.0, queda cuantificado cuánto del
rendimiento anterior era confound. Acompañar con el tiny baseline (ya
ejecutado) como cota trivial y, si es posible, Grad-CAM sobre 5–10 aciertos.

---

### Reproducir esta auditoría

```bash
python -m src.data.audit_splits                 # overlap de sujetos, composición
python -m src.audit.audit_leakage               # tiny baseline + clf de origen + duplicados
# Salidas: docs/audit/audit_leakage.json, docs/audit/audit_features.csv
```
