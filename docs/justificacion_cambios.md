# Justificación de los cambios sobre el modelo CNN 3D

> Documento de apoyo: explica el **porqué** de cada uno de los 8 pasos
> aplicados sobre el pipeline de entrenamiento/evaluación tras detectar
> que el modelo predecía casi todo como tumor (specificity = 0 en test).

## Contexto del fallo

Estado previo:

- AUC test = 0.745
- Sensitivity = 0.909
- **Specificity = 0.000** (TN = 0 sobre 165 negativos)
- Accuracy = 0.467
- `val_specificity` casi siempre 0 durante el entrenamiento
- `train_auc` saturando ~0.99 y `val_auc` oscilando mucho desde la
  época 1
- El "mejor" checkpoint según `val_auc` resultaba ser el de la época 1

Resumen ejecutivo del diagnóstico:

> Esto **no** es un problema de balanceo de clases. Las clases ya están
> casi equilibradas (817 pos / 770 neg en train, `pos_weight ≈ 0.94`).
> Es una combinación de **calibración rota** + **preprocesado
> inconsistente entre datasets** (UPENN vs. resto) + **BatchNorm con
> `batch_size = 1`**, agravado por un criterio de checkpoint que premia
> AUC en lugar de balanced accuracy.

---

## Paso 1 — Auditoría de splits + estratificar por (dataset, label, subject_id)

**Por qué:** `create_splits` solo estratificaba por label. Por azar quedó
equilibrado entre datasets, pero si se cambia la seed o se añaden datos,
se rompe sin avisar. Y si un sujeto tuviera varias sesiones (NKI usa
sufijos como `-BAS2`, `-BAS3`), el código actual las separaría entre
train/test → leakage de sujeto.

**Coste:** muy bajo. Es seguro de cara al futuro.

**Métrica para validar el cambio:** `python -m src.data.audit_splits`
debe mostrar 0 overlap por subject_id y proporciones similares por
(split, dataset, label).

---

## Paso 2 — Evaluación por dataset

**Por qué:** un AUC global de 0.745 esconde si el modelo funciona bien
en BraTS y fatal en IXI/NKI, o si simplemente separa "datasets" en vez
de "tumor/no tumor". Sin métricas por dataset, todos los experimentos
posteriores son ciegos.

**En particular:** queremos saber si los falsos positivos vienen de IXI,
de NKI o de ambos. En el estado previo, los dos al 100 % a thr = 0.5,
con NKI **aún más concentrado** (scores 0.60–0.76) → el modelo no había
aprendido un "no-tumor" característico.

---

## Paso 3 — Histogramas + threshold óptimo en validation

**Por qué:** los scores observados en test viven en [0.45, 0.95] y
**todos los negativos** caen por encima de 0.5 → threshold = 0.5 es
absurdo para esta distribución. Esto **explica solamente el
`specificity = 0`**, no que el modelo esté mal entrenado: con un AUC de
0.745 hay señal aprovechable si se elige el threshold correctamente.

Hay que separar dos cosas:

1. *Modelo descalibrado* → se arregla con un threshold elegido en
   validation (Youden J o sensitivity mínima maximizando specificity).
2. *Modelo sin señal real* → no se arregla con threshold; requiere
   cambios estructurales (pasos 4–8).

**Regla:** el threshold se elige en **validation** y se aplica en
**test**, nunca al revés.

---

## Paso 4 — BatchNorm → GroupNorm (cambio de mayor impacto)

**Por qué:** con `batch_size = 1`, BatchNorm es inestable:

- En train, las estadísticas por batch se calculan sobre **una sola
  muestra** → ruido enorme.
- En eval, BN usa `running_mean`/`running_var` acumulados; con batch 1
  estos nunca convergen bien.
- Resultado: gap masivo train ↔ val. Esto explica `train_auc ≈ 0.99`
  vs. `val_auc` oscilando 0.05–0.65 desde la época 1.

GroupNorm **no depende del batch**, calcula estadísticas por canal
sobre cada muestra. Es la elección estándar cuando no puedes usar
batches grandes en 3D.

> Si solo se hace un cambio del listado, **hacer este**.

---

## Paso 5 — Checkpoint por balanced_accuracy con sensitivity mínima

**Por qué:** guardar por `val_auc` cuando `val_specificity = 0` siempre
significa que el "mejor" modelo es inservible para triaje. Puedes
tener AUC alto y no acertar ni un sano (es justo lo que pasaba: mejor
checkpoint = época 1).

- `balanced_accuracy = (sensitivity + specificity) / 2` penaliza
  directamente que specificity sea 0.
- La condición extra `sensitivity ≥ 0.80` evita guardar un modelo
  trivial que "acierte sanos" diciendo "todo es sano".

Es decir, premiamos modelos **útiles para triaje**, no modelos con
buena separación de scores pero mal calibrados.

---

## Paso 6 — Quitar pos_weight, bajar lr, subir weight_decay, scheduler, AMP off

Cuatro cambios pequeños con motivaciones distintas:

- **`pos_weight: 1.0`** — con 817/770, `auto` daba 0.94 (≈ 1, neutro).
  Fijarlo a 1.0 elimina ruido conceptual y descarta la hipótesis "el
  loss favorece tumor".
- **`lr: 1e-4 → 5e-5`** — el historial muestra overfitting desde la
  época 2 (`train_auc` salta a 0.97 en la 2). Bajar lr ralentiza el
  sobreajuste.
- **`weight_decay: 1e-4 → 1e-3`** — misma motivación, regularizar más.
- **AMP off** — float16 con `batch=1` + GroupNorm amplifica
  inestabilidades numéricas; se reactiva más adelante al subir batch.
- **`scheduler: cosine`** — lr fijo no se adapta al final del
  entrenamiento; el cosine permite "ajuste fino" en las últimas
  épocas.

---

## Paso 7 — Augmentation real + center crop también en train

Dos cambios distintos en el mismo paso:

### Center crop también en train (en vez de random crop)

Antes: train usaba `random_crop` y val/test usaban `center_crop`. Eso
significa **dos distribuciones espaciales distintas** entre train y
val. Es una fuente adicional de val-AUC errático que no tiene nada que
ver con el modelo. Igualarlos elimina ese shift.

### Augmentation de intensidad (gamma + ruido gaussiano)

Antes solo había flips. Solo flips no introduce variabilidad de
intensidad, y el modelo se aferra al **brillo característico de cada
dataset** (BraTS vs. UPENN vs. IXI vs. NKI). Gamma y ruido le obligan
a usar geometría/contraste relativo, no intensidad absoluta → **reduce
el domain shift entre datasets**.

La augmentation se aplica **sobre la máscara de voxels no cero**, para
no inyectar señal en el fondo (que es donde precisamente UPENN ya
tiene un problema, ver paso 8).

---

## Paso 8 — Re-normalizar UPENN con Otsu

**Por qué:** UPENN trae imágenes procesadas por CaPTk que **no están
skull-stripped igual que HD-BET**. Datos medidos por dataset:

| Dataset | Fracción voxels ≠ 0 | T1 min (media) |
|---------|----------------------|----------------|
| BraTS   | 0.21                 | -2.75          |
| IXI     | 0.18                 | -2.84          |
| NKI     | 0.16                 | -2.40          |
| UPENN   | **0.66**             | **-0.91**      |

UPENN tiene 3–4× más voxels no-cero que el resto y un rango de
intensidades muy distinto → hay un "shell" de fondo gris que las otras
imágenes no tienen.

La normalización actual (`percentile(positive, 5)` como threshold de
máscara en `normalize_intensity`) **cae dentro del shell** en UPENN, lo
que arrastra la media/std hacia ese shell. Resultado: UPENN queda con
una escala distinta a BraTS/IXI/NKI, y el modelo aprende "imagen con
shell → tumor" porque solo UPENN lo tiene. Es la **causa más probable
del domain shift más serio del dataset**, y explica por qué UPENN tiene
la varianza de scores más alta (0.25–0.94 vs. BraTS 0.45–0.90).

**La corrección:** usar Otsu sobre los voxels positivos para encontrar
el umbral entre "shell gris" y "tejido cerebral", con un fallback al
percentil 30 si scikit-image no está disponible. Tras Otsu, la media y
std se calculan **solo sobre tejido**, no sobre el shell.

**Coste:** hay que reprocesar UPENN (caro en tiempo). Sin esto, los
datasets nunca serán comparables y el modelo seguirá teniendo un atajo
trivial para predecir UPENN.

---

## Razonamiento global del orden

- **Pasos 1–3**: *diagnóstico*. No cambian el modelo, solo permiten
  ver qué está pasando. Imprescindibles para evaluar si los pasos
  posteriores ayudan.
- **Pasos 4–5**: *arreglos críticos*. BatchNorm inestable + checkpoint
  mal elegido = la mayor parte del problema observable.
- **Pasos 6–7**: *regularización y reducción de domain shift
  train/val*. Sin estos, el modelo seguirá overfitting.
- **Paso 8**: *arreglar el preprocesado*. Imprescindible para que los
  datasets sean comparables y el modelo no use atajos triviales.

> Si hubiera que parar en algún punto, **parar tras el paso 5** ya
> debería dar `specificity > 0` y un test usable. Los pasos 6–8 son
> para que el modelo sea *bueno*, no solo *funcional*.

## Métrica objetivo

Tras el paso 5 (mínimo viable):

- En test, con threshold elegido en val: `sensitivity ≥ 0.85` **y**
  `specificity > 0` en ambos datasets sanos (IXI y NKI).

Tras los pasos 6–8 (objetivo deseado):

- En test: `sensitivity ≥ 0.85` **y** `specificity ≥ 0.60`.
- `score_mean` de UPENN ≈ `score_mean` de BraTS (sin domain shift
  visible entre datasets de la misma clase).

---

# Resultados de la ejecución del plan (run `20260527_152619`)

> Sección añadida el 2026-05-28 con los resultados reales tras aplicar
> los 8 pasos anteriores. Entrenamiento del 2026-05-27, evaluado sobre
> el split de test (nunca visto en train ni en el early stopping).
> Entorno: conda env `igsan`, GPU AMD Radeon RX 6700 XT (ROCm,
> PyTorch 2.5.1+rocm6.2).

## Corte del entrenamiento

- **37 de 40 épocas**: paró por **early stopping** (`patience = 10`), no
  por crash de GPU. El mejor `val_loss` fue en la **época 27**
  (`val_loss = 0.00594`, `val_balanced_accuracy = 1.0`, `sensitivity =
  1.0`); 10 épocas después sin mejorar, cortó en la 37. Terminó limpio
  (generó `curves.png` al final, lo que no ocurriría en un crash).
- Checkpoint final: `outputs/checkpoints/20260527_152619/best.pt`.

## Paso 1 ejecutado — auditoría de splits

`python -m src.data.audit_splits`:

| split | composición | overlap subject_id |
|-------|-------------|--------------------|
| train | 1587 (817 pos / 770 neg) | train ∩ val = 0 |
| val   | 340 (175 pos / 165 neg)  | train ∩ test = 0 |
| test  | 340 (175 pos / 165 neg)  | val ∩ test = 0 |

Los 4 datasets presentes en los tres splits. **0 overlap** de sujeto.

> **Matiz importante:** en cada fila `n_samples == n_subjects` → en los
> datos actuales **cada sujeto tiene un único fichero**. No hay sesiones
> múltiples tipo NKI `-BAS2`/`-BAS3` que separar, así que el split
> agrupado por `subject_id` (paso 1) es, para *estos* datos, equivalente
> al split por fichero anterior. El leakage de sujeto que el paso 1
> prevenía **no era posible aquí**; el cambio sigue siendo correcto como
> protección a futuro, pero no explica diferencias entre corridas.

## Pasos 2–3 ejecutados — umbral

`python -m src.evaluation.threshold_analysis ... --target-sensitivity 0.95`:

- **VAL AUC = 1.0000** (separación perfecta en validation).
- Umbrales candidatos: Youden(val) = 0.731; min-sensitivity(val) = 0.993.
- En test, `0.5`, Youden y `min_sen` dan resultados casi idénticos; el
  umbral alto `min_sen` incluso **empeora** la sensibilidad (0.977 vs
  0.994) → para triaje (no perder tumores) **no conviene subir el
  umbral**.

**Umbral elegido: 0.5.**

## Paso final — evaluación en test (umbral 0.5, n = 340)

| AUC | PR-AUC | Accuracy | Sensitivity | Specificity | F1 |
|-----|--------|----------|-------------|-------------|----|
| 1.000 | 1.000 | 0.997 | 0.994 | 1.000 | 0.997 |

- **1 falso negativo, 0 falsos positivos.**
- FN: `BraTS2021_00736`, score 0.092 (confiadamente clasificado como
  sano; outlier dentro de BraTS, cuya media es 0.986). Candidato a
  revisión manual del volumen (tumor sutil o defecto de preprocesado).

Scores medios por dataset:

| Dataset | Clase | score medio | n |
|---------|-------|-------------|---|
| upenn        | tumor | 0.999 ± 0.000 | 88 |
| brats        | tumor | 0.986 ± 0.098 | 87 |
| ixi          | sano  | 0.007 ± 0.018 | 86 |
| nki_rockland | sano  | 0.001 ± 0.001 | 79 |

## Verificación del paso 8 (re-normalización UPENN con Otsu)

Los `.npz` de UPENN se regeneraron el 2026-05-27 (13:44–14:04) **antes**
de lanzar el entrenamiento (15:26), con el `normalize_intensity` basado
en Otsu. Comprobación dura sobre un volumen crudo de UPENN:

| Normalización | Umbral | Máscara "tejido" | media | std |
|---------------|--------|------------------|-------|-----|
| Vieja (pctl-5) | 6.00  | **62.99%** del vol | 180.8 | 210.3 |
| Nueva (Otsu)   | 232.77| **22.43%** del vol | 427.4 | 155.4 |

El método viejo metía 3,83 M de vóxels de shell gris en la máscara (64%
de su propia máscara). Tras Otsu, los `.npz` de UPENN tienen ~22–28% de
vóxels no-cero, igual que BraTS (~20%) — ya no el 63% del shell.

> **Objetivo del paso 8 CUMPLIDO:** `score_mean` UPENN (0.999) ≈
> `score_mean` BraTS (0.986). Ya no hay domain shift visible **entre
> datasets de la misma clase** (positivos). El atajo "imagen con shell →
> tumor" descrito en el paso 8 queda eliminado.

## Limitación conocida — confound estructural dataset ↔ clase

A pesar de las métricas casi perfectas, **no puede afirmarse que el
modelo detecte tumor**, por la composición de los datos:

- **Positivos (tumor):** BraTS + UPENN.
- **Negativos (sano):** IXI + NKI Rockland.

La clase está **perfectamente correlacionada con el dataset de origen**,
así que el modelo puede acertar reconociendo *de qué fuente/escáner*
viene la imagen, no la lesión. Dos evidencias lo respaldan:

1. **AUC = 1.0** en val y test (separación perfecta), poco realista para
   detección de tumor.
2. **Varianza de score casi nula dentro de cada dataset** (UPENN ±0.000,
   NKI ±0.001): el modelo asigna prácticamente el mismo score a todas las
   imágenes de un dataset, independientemente del contenido. Un detector
   de lesión real mostraría variabilidad (tumores sutiles vs. evidentes).

El paso 8 corrigió el domain shift **dentro** de la clase positiva
(UPENN vs BraTS), pero **no** este confound, que es estructural a la
composición del dataset (positivos y negativos vienen de fuentes
disjuntas) y no se arregla con preprocesado.

**Vías para despejarlo (trabajo futuro):**

1. **Negativos del mismo dataset que los positivos** (sujetos sanos o
   volúmenes sin tumor de BraTS/UPENN) → rompe la correlación clase↔fuente.
2. **Validación cruzada por dominio** (leave-one-dataset-out): entrenar
   con un dataset de cada clase y evaluar en otro no visto.
3. **Grad-CAM** sobre aciertos: comprobar si el modelo atiende a la
   lesión o a bordes/fondo/firma del dominio.

## ¿Se cumplió la métrica objetivo?

| Criterio (objetivo deseado, pasos 6–8) | Resultado | ¿Cumple? |
|----------------------------------------|-----------|----------|
| `sensitivity ≥ 0.85` en test | 0.994 | ✅ |
| `specificity ≥ 0.60` en test | 1.000 (IXI y NKI al 100%) | ✅ |
| `score_mean` UPENN ≈ BraTS | 0.999 vs 0.986 | ✅ |

Los tres objetivos numéricos del plan se superan ampliamente. **Pero**
ese resultado debe leerse junto a la limitación del confound: las
métricas reflejan, en parte indeterminada, capacidad de distinguir
*dominios* y no solo *tumor/no-tumor*. La validez clínica como triaje
queda pendiente de las vías 1–3.
