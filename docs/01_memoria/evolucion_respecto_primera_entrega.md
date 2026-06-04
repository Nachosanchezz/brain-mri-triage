# Evolución del proyecto respecto a la primera entrega

**Documento:** Comparación entre el planteamiento inicial (1ª entrega, aprobada por el director) y el estado actual del repositorio.  
**Basado en:** análisis del código, configuraciones, checkpoints y resultados existentes en el repositorio.  
**Fecha de elaboración:** mayo 2026

---

> ## ⚠️ NOTA DE ACTUALIZACIÓN (estado posterior)
>
> **Este documento describe un estado INTERMEDIO del proyecto y ha quedado
> parcialmente desactualizado.** Se conserva como registro histórico de la
> evolución respecto a la primera entrega, pero las siguientes afirmaciones que
> aparecen más abajo **ya NO reflejan el estado final** del repositorio:
>
> | Afirmación en este documento | Estado real actual | Fuente vigente |
> |---|---|---|
> | «El modelo 3D **no ha sido entrenado todavía**» / «No hay checkpoint» | La CNN 3D **sí se entrenó**: run `20260527_152619`, test AUC ≈ 0.99997 | `outputs/evaluation/cnn3d_test_results.json`, `docs/justificacion_cambios.md` |
> | «Los scripts de evaluación (`evaluate_3d.py`) **no existen todavía**» | `src/evaluation/evaluate_3d.py` **existe y se ha ejecutado** | `src/evaluation/evaluate_3d.py` |
> | «negatives → **IXI únicamente** (577)» | Negativos = **IXI (577) + NKI Rockland (523) = 1100** | `data/processed/preprocessing_summary.json` |
> | Parámetros de la CNN 3D «**~200K**» | **504 553 parámetros** (verificado instanciando el modelo; 504 229 en la variante 1-canal) | `src/models/cnn3d.py` |
> | Referencia a `docs/revision_tecnica.md` | Ese archivo **no existe**; el informe equivalente es `docs/auditoria_resultados_sospechosos.md` | `docs/auditoria_resultados_sospechosos.md` |
>
> **Hallazgo posterior no recogido aquí.** Tras entrenar la CNN 3D se detectó que
> su rendimiento casi perfecto se debe a un **confound estructural dominio↔clase**
> (la etiqueta coincide al 100 % con el dataset de origen) y **no** a detección
> real de tumor. La auditoría completa (tiny baseline, LODO, validación
> intra-dominio BTC, embeddings, Grad-CAM) está en `docs/audit/` y la
> reconstrucción consolidada en `docs/reconstruccion_evolucion_tfg.md`. **Para el
> estado y las conclusiones finales, consultar esos documentos, no este.**

---

## 0. Contexto

La primera entrega planteaba una clasificación binaria a nivel de estudio (tumor sí/no), con BraTS 2021 como clase positiva, IXI como clase negativa y UPENN-GBM como conjunto de evaluación externa. El sistema debía usar T1 y T2, aplicar preprocesado homogéneo, partir datos a nivel de paciente y usar sensibilidad como métrica principal.

Esta tabla resume la evolución completa:

| Apartado | Primera entrega | Estado actual | Cambio |
|---|---|---|---|
| Datasets de entrenamiento | BraTS (pos) + IXI (neg) | BraTS + IXI — igual | Sin cambio |
| Dataset externo | UPENN-GBM (solo test) | UPENN incorporado al entrenamiento (decisión B); evaluación externa pendiente de nuevo dataset | **Cambio crítico — decisión tomada** |
| Datasets de negativos | IXI únicamente | IXI + NKI Rockland (multi-fuente) | **Ampliación** |
| Modalidades | T1 + T2 | T1 + T2 | Sin cambio |
| Preprocesado | Homogéneo previsto | Implementado y ejecutado completamente | Completado |
| Skull-stripping IXI | Previsto | Ejecutado con HD-BET y verificado | Completado |
| Baseline | CNN sencilla | ResNet18 2D pseudo-multicanal (entrenado, evaluado) | Completado |
| Modelo principal | CNN 3D previsto | CNN 3D propia implementada (no entrenada aún) | En curso |
| Métricas | Sensibilidad, AUC, PR-AUC, F1, calibración | Parcialmente implementadas — ver sección 6 | Incompleto |
| Evaluación externa | UPENN-GBM | No ejecutada | Pendiente |
| Sesgo de dominio | Identificado, mitigación prevista | Mitigado + analizado empíricamente | En curso |

---

## 1. Cambios en los datasets

### 1.1 Datasets presentes en el repositorio

| Dataset | Ruta | Papel actual | Papel en la 1ª entrega |
|---|---|---|---|
| BraTS 2021 | `data/processed/positives/BraTS2021_*.npz` (580) | Clase positiva — entrenamiento | Clase positiva — entrenamiento |
| IXI | `data/processed/negatives/IXI*.npz` (577) | Clase negativa — entrenamiento | Clase negativa — entrenamiento |
| UPENN-GBM | `data/processed/positives/UPENN-GBM-*.npz` (587) | **Clase positiva — en `positives/`** | Solo test externo |

### 1.2 Decisión adoptada: UPENN-GBM incorporado al entrenamiento

Respecto al planteamiento de la primera entrega, **se ha tomado la decisión explícita de incorporar UPENN-GBM al pool de entrenamiento** en lugar de reservarlo como conjunto de evaluación externa cross-dataset. UPENN-GBM ha sido preprocesado con `label=1` y colocado en `data/processed/positives/` junto a BraTS.

**Reformulación del objetivo:** este cambio implica desplazar parcialmente el foco metodológico desde "evaluación cross-dataset con un único conjunto externo" hacia "**aprendizaje de un clasificador robusto frente a variabilidad multi-fuente**". La justificación es que entrenar con dos fuentes positivas distintas (BraTS y UPENN, con protocolos de adquisición y poblaciones diferentes) reduce el riesgo de que el modelo aprenda firmas específicas de un único dataset y mejora su capacidad de generalización dentro de la distribución de entrenamiento.

**Limitación reconocida explícitamente:** la consecuencia directa es que el repositorio actual **no dispone de un conjunto de evaluación externa cross-dataset**. La validación externa queda como trabajo pendiente, condicionada a la identificación y descarga de un dataset adicional con tumores cerebrales (candidatos posibles: TCGA-GBM, REMBRANDT, datasets de Kaggle específicos) que no haya sido visto durante el entrenamiento. Esta limitación debe declararse de forma explícita en la memoria como una desviación consciente respecto al planteamiento original, motivada por la decisión de priorizar el volumen y la heterogeneidad de los datos de entrenamiento.

**Evidencia en el código:** `data/processed/preprocessing_summary.json` registra:
```json
"positives": 1167  (580 BraTS + 587 UPENN)
"negatives": 1100  (577 IXI + 523 NKI Rockland)
```

**Estado del splits.json:** `configs/train_3d.yaml` mantiene `recreate_splits: true`, por lo que la próxima ejecución de `train_3d.py` regenerará los splits incorporando los 587 UPENN al pool de positivos y los 523 NKI Rockland al pool de negativos. Tras este reentrenamiento, conviene cambiar a `recreate_splits: false` y commitear el `splits.json` resultante para garantizar reproducibilidad entre ejecuciones.

### 1.3 Incorporación de NKI Rockland como segundo dataset de negativos

Como reacción al desbalance que introdujo UPENN (1167 positivos frente a 577 negativos, ratio ~2:1), se ha incorporado **NKI Rockland** como segundo dataset de sujetos sanos. Los scripts `src/preprocessing/download_nki_rockland.py` y `src/preprocessing/preprocess_nki_rockland.py` descargan desde el bucket S3 público de FCP-INDI, filtrando exclusivamente sujetos con T1w y T2w en la misma sesión BIDS, y aplican el mismo pipeline de preprocesado (HD-BET → RAS → 1 mm³ → crop 192×224×192 → z-score) que el resto de datasets. El resultado es de 523 volúmenes adicionales con `label=0`.

**Estado actual del balance de clases:** con NKI Rockland incorporado, la ratio se acerca al equilibrio (1167:1100, aproximadamente 1.06:1), lo que reduce la dependencia del modelo respecto al `pos_weight` calculado automáticamente. El sistema mantiene `pos_weight: auto` en la configuración por consistencia, pero su efecto correctivo es ahora marginal.

**Refuerzo del argumento multi-fuente:** la presencia de dos datasets de negativos (IXI y NKI Rockland), con protocolos de adquisición y características demográficas diferentes, refuerza la línea metodológica de robustez multi-fuente adoptada en la sección 1.2. Tanto el lado positivo (BraTS + UPENN) como el negativo (IXI + NKI Rockland) están ahora representados por dos fuentes independientes cada uno.

---

## 2. Cambios en las modalidades de entrada

**No hay cambio.** Ambos datasets de entrenamiento (BraTS, IXI) y UPENN-GBM se procesan con T1 y T2 exclusivamente.

**Evidencia:** `data/processed/positives/BraTS2021_00000.npz` contiene keys `t1`, `t2`, `label`. Idem para IXI y UPENN. `src/data/dataset_3d.py` línea 199: `t1 = sample["t1"]`, `t2 = sample["t2"]`.

La decisión de restringirse a T1+T2 por ser las modalidades comunes sigue siendo correcta y sigue justificada en la memoria tal y como se planteó.

---

## 3. Cambios en el preprocesado

### 3.1 Pipeline sobre los volúmenes (sin cambio funcional)

El pipeline de preprocesado es el mismo que se planteó:

| Paso | Script | Estado |
|---|---|---|
| Conversión DICOM→NIfTI (BraTS) | `src/preprocessing/conversion/dicom_to_nifti.py` | Ejecutado (580/585) |
| Skull-stripping IXI con HD-BET | `src/preprocessing/preprocess_ixi.py` | **Completado** (verificado: 80-84% voxels a cero) |
| Conversión DICOM→NIfTI (UPENN) | `src/preprocessing/convert_dicom_to_nifti_upenn.py` | En curso |
| Reorientación RAS | `src/preprocessing/base_preprocessing.py` | Ejecutado |
| Remuestreo 1mm isótropo | `base_preprocessing.py` | Ejecutado |
| Crop/pad a (192, 224, 192) | `base_preprocessing.py` | Ejecutado |
| Normalización z-score (voxels > 0) | `base_preprocessing.py` | Ejecutado |
| Guardado `.npz` T1+T2+label | scripts por dataset | Ejecutado (1744 muestras) |

### 3.2 Refactorización del código de preprocesado

En la primera entrega existía un solo script monolítico `preprocess_volumes.py`. Ahora el código se ha refactorizado en:

- `src/preprocessing/base_preprocessing.py` — funciones comunes (reorient, resample, crop, normalize)
- `src/preprocessing/preprocess_brats.py` — pipeline BraTS
- `src/preprocessing/preprocess_ixi.py` — pipeline IXI
- `src/preprocessing/preprocess_upenn.py` — pipeline UPENN-GBM
- `src/preprocessing/conversion/dicom_to_nifti.py` — conversión DICOM→NIfTI

El `preprocess_volumes.py` original sigue existiendo (no fue eliminado). Hay duplicidad de código entre el monolítico y los nuevos scripts modulares. Para la memoria, referencia los nuevos scripts modulares como la implementación definitiva.

### 3.3 Nuevo paso: crop de sub-volumen en el dataset

El `src/data/dataset_3d.py` aplica un recorte adicional a **(128, 160, 128)** durante la carga (línea 63 de `train_3d.py`, parámetro `crop_shape`). Esto reduce el volumen de `(192, 224, 192)` a `(128, 160, 128)` para que quepa en memoria GPU con batch_size=1.

Este paso NO estaba en el preprocesado original. Es un paso de reducción de dimensionalidad implícito en el dataset. En train se aplica con `random_crop=True` (posición aleatoria); en val/test con `random_crop=False` (centrado). Esto implica que el modelo no ve siempre la totalidad del cerebro durante el entrenamiento.

**Impacto metodológico:** el crop aleatorio actúa como augmentation implícito y reduce la memoria necesaria. Puede excluir regiones tumorales periféricas si el tumor no está centrado. Debería mencionarse en la memoria como trade-off entre cobertura espacial y limitaciones de memoria.

---

## 4. Cambios en la arquitectura del modelo

### 4.1 Baseline 2D → completado y evaluado

El baseline planteado en la primera entrega (CNN sencilla como referencia) fue implementado como **ResNet18 pseudo-2D** (`src/models/baseline_2d.py`, ya no existe como fichero — solo el checkpoint `outputs/checkpoints/baseline_2d_best.pt`).

**Resultados del baseline** (`outputs/evaluation/test_results.json`):
```
AUC = 0.9996
PR-AUC = 0.9996
Sensibilidad a threshold=0.5: 1.000
Especificidad a threshold=0.5: 0.965
Sensibilidad al 95%: 0.989 → especificidad: 1.000
```

Estos resultados son del modelo entrenado sobre BraTS vs IXI skull-stripped (174 muestras de test: 87 BraTS + 87 IXI). El `domain_analysis` confirma que existe solapamiento ligero de scores (`overlap_exists: true`, gap=-0.147), lo que indica que el sesgo de dominio se ha reducido respecto al primer entrenamiento (AUC=1.0 trivial).

### 4.2 Modelo 3D → implementado, no entrenado

El modelo principal es la **CNN 3D propia** `BrainTumorCNN3D` (`src/models/cnn3d.py`):

```
Input: (B, 2, 128, 160, 128)   ← T1+T2, crop reducido
  ↓ ConvBlock3D(2 → 12)  + MaxPool3d(2)
  ↓ ConvBlock3D(12 → 24) + MaxPool3d(2)
  ↓ ConvBlock3D(24 → 48) + MaxPool3d(2)
  ↓ ConvBlock3D(48 → 96) + AdaptiveAvgPool3d(1)
  ↓ FC(96→96) + ReLU + Dropout(0.25) + FC(96→2)
Output: logits (B, 2)
```

**Diferencias respecto a la 1ª entrega:**

| Aspecto | Baseline 2D (1ª entrega) | CNN 3D (modelo principal) |
|---|---|---|
| Dimensión | Pseudo-2D (16 slices como canales) | 3D volumétrico completo |
| Inicialización | Transfer learning ImageNet | **Desde cero (random init)** |
| Parámetros aprox. | ResNet18: 11M (pretrained) | **504 553** (compacto, no pretrained; ver nota de actualización) |
| Input shape | (B, 32, 192, 224) | (B, 2, 128, 160, 128) |
| Volumen cubierto | 16 slices axiales centrales | Sub-volumen 3D completo |

La CNN 3D es más correcta metodológicamente para volúmenes cerebrales que el slice-stacking, pero abandona el transfer learning de ImageNet. Para la memoria, explicar que la arquitectura 3D responde a la necesidad de capturar la estructura espacial completa del tumor.

**Estado:** No hay checkpoint `cnn3d_best.pt` ni `cnn3d_history.json`. El modelo 3D **no ha sido entrenado todavía**.

---

## 5. Cambios en entrenamiento y validación

### 5.1 Parámetros de entrenamiento

| Parámetro | train.py (baseline 2D) | train_3d.py (CNN 3D) |
|---|---|---|
| Optimizador | Adam | **AdamW** |
| LR | 1e-4 | 1e-4 |
| Weight decay | 1e-4 | 1e-4 |
| Loss | CrossEntropy (sin pesos) | CrossEntropy (**class_weights auto**) |
| Batch size | 16 | **1** (por memoria GPU) |
| Epochs máx. | 50 | **30** |
| Early stopping | val_AUC, patience=10 | **val_loss, patience=8** |
| LR scheduler | ReduceLROnPlateau | Ninguno |
| AMP | No | **Sí** (mixed precision) |
| Crop | Volumen completo (192,224,192) | Sub-volumen (128,160,128) |

### 5.2 Splits de datos

La partición 70/15/15 estratificada a nivel de paciente se mantiene. El mecanismo cambia:
- Original: `sklearn.train_test_split` con stratify
- Actual: `dataset_3d.py::create_splits()` — shuffle por clase por separado con `numpy.random.default_rng`

Ambos garantizan estratificación. La partición sigue siendo a nivel de fichero (un `.npz` = un paciente), no a nivel de slice ni de volumen.

**Riesgo pendiente:** `recreate_splits: true` en `configs/train_3d.yaml` hace que cada vez que se ejecuta `train_3d.py` se regeneren los splits. Si en ese momento `positives/` contiene tanto BraTS como UPENN, el split mezclará ambos. Cambiar `recreate_splits: false` antes de entrenar si se quiere preservar el split actual (solo BraTS + IXI).

### 5.3 Métricas durante entrenamiento — regresión importante

`train_3d.py` solo calcula **loss y accuracy** durante el entrenamiento (`run_epoch`, líneas 102-148). No calcula AUC, sensibilidad ni especificidad durante las épocas.

La 1ª entrega planteaba sensibilidad como métrica principal y la función `evaluate()` del `train.py` calculaba AUC, sensibilidad y especificidad. Esa función ya no existe en el código 3D.

Esto significa:
- El modelo 3D se optimiza por val_loss (no val_AUC) → el early stopping no refleja la métrica clínica
- No hay forma de saber durante el entrenamiento si la sensibilidad mejora
- Habrá que escribir un script de evaluación post-entrenamiento (ver sección 6)

---

## 6. Cambios en métricas y análisis de resultados

### 6.1 Métricas implementadas para el baseline 2D

Los ficheros en `outputs/evaluation/` contienen los resultados del baseline:

| Métrica | Implementado | Resultado baseline |
|---|---|---|
| ROC-AUC | Sí | 0.9996 |
| PR-AUC | Sí | 0.9996 |
| Sensibilidad | Sí | 1.000 (a 0.5) |
| Especificidad | Sí | 0.965 (a 0.5) |
| F1-score | Sí | 0.983 |
| Umbral al 95% sensibilidad | Sí | threshold=0.730 → spec=1.000 |
| Análisis de dominio BraTS vs IXI | Sí | overlap=True, gap=-0.147 |
| Tests Kruskal-Wallis / Mann-Whitney IXI | Sí | `domain_bias_analysis.json` |
| Calibración | **No** | Pendiente |
| Análisis por institución BraTS | Parcial | `domain_bias_analysis.json` |

### 6.2 Métricas del modelo 3D — pendiente completo

Los scripts de evaluación correspondientes al modelo 3D (`evaluate_3d.py`, equivalente al viejo `evaluate.py`) **no existen todavía**. `train_3d.py` solo reporta accuracy al final del test (`test_metrics = run_epoch(...)`).

Para la memoria, reportar métricas completas del modelo 3D requerirá:
1. Entrenar el modelo (`python src/training/train_3d.py`)
2. Escribir un `evaluate_3d.py` con las mismas métricas que tenía `evaluate.py`

### 6.3 Calibración — no implementada

La primera entrega mencionaba calibración (Platt scaling, reliability plots) como parte del análisis. Ningún script actual la implementa. Esto debe mencionarse como limitación en la memoria.

---

## 7. Cambios en el objetivo clínico/metodológico

### 7.1 La tarea principal no ha cambiado

El problema sigue siendo una clasificación binaria a nivel de estudio: tumor sí/no. La formulación clínica (triage, sensibilidad como métrica principal, no sustituir al radiólogo) se mantiene intacta.

### 7.2 Evolución técnica coherente con el plan

La secuencia real de desarrollo es coherente con lo planteado:
1. **Baseline 2D** → implementado, entrenado, evaluado con métricas completas ✓
2. **Diagnóstico de sesgo de dominio** → detectado (AUC=1.0 trivial), analizado, corregido (HD-BET) ✓
3. **Modelo principal 3D** → implementado (`cnn3d.py`), pendiente de entrenar

### 7.3 Desviaciones que requieren justificación en la memoria

1. **UPENN-GBM incorporado al entrenamiento**: Decisión adoptada (opción B). Se reformula el objetivo metodológico hacia "robustez multi-fuente" en lugar de "evaluación cross-dataset con conjunto externo único". La evaluación externa queda como trabajo pendiente, condicionada a la incorporación de un dataset adicional con tumores cerebrales no visto durante el entrenamiento (candidatos: TCGA-GBM, REMBRANDT, Kaggle).

2. **NKI Rockland como segundo dataset de negativos**: Ampliación no contemplada en la 1ª entrega. Justificable por dos motivos complementarios: (a) restablece el equilibrio de clases roto al incorporar UPENN, y (b) refuerza la línea multi-fuente al introducir variabilidad demográfica y de protocolo en el lado negativo.

3. **No hay evaluación externa cross-dataset todavía**: Consecuencia directa de la decisión 1. Declarar explícitamente como limitación y proponer la validación externa como trabajo futuro inmediato.

4. **No hay calibración**: Mencionarla como trabajo pendiente / limitación del estudio.

5. **AUC=0.9996 del baseline**: Resultado real, no trivial (hay solapamiento de scores, overlap=True). Puede reportarse con la caveat de que persiste riesgo de sesgo residual (el gap es pequeño: -0.147) y de que ese resultado corresponde a un entrenamiento sobre BraTS+IXI únicamente, no sobre la composición multi-fuente actual.

6. **3D CNN sin transfer learning**: Diferente a ResNet18 pretrained. Justificable por la diferencia de dominio (volúmenes MRI vs imágenes naturales) y la disponibilidad de datos suficientes (>2000 muestras con la composición actual).

---

## 8. Estructura actual del repositorio

```
brain-mri-triage/
├── src/
│   ├── preprocessing/
│   │   ├── base_preprocessing.py          ← Funciones comunes (RAS, resample, crop, z-score)
│   │   ├── preprocess_brats.py            ← Pipeline BraTS
│   │   ├── preprocess_ixi.py              ← Pipeline IXI (skull-stripped)
│   │   ├── preprocess_upenn.py            ← Pipeline UPENN-GBM (modificado)
│   │   ├── preprocess_volumes.py          ← Script monolítico original (redundante)
│   │   ├── skull_strip_ixi.py             ← HD-BET sobre IXI
│   │   ├── convert_dicom_to_nifti_upenn.py ← Conversión UPENN DICOM→NIfTI
│   │   ├── conversion/dicom_to_nifti.py   ← Conversión BraTS/UPENN DICOM→NIfTI (refactorizado)
│   │   └── dataset/summarize_processed.py ← Resumen de data/processed/
│   ├── data/
│   │   ├── dataset_3d.py                  ← Dataset 3D actual (en uso)
│   │   ├── dataset.py                     ← Dataset 2D original (puede no existir)
│   │   ├── transforms.py                  ← Augmentation MONAI (puede no existir)
│   │   └── filter_upenn_manifest.py       ← Filtrado manifest TCIA UPENN
│   ├── models/
│   │   ├── cnn3d.py                       ← Modelo 3D actual (no entrenado)
│   │   └── baseline_2d.py                 ← Modelo 2D original (puede no existir)
│   ├── training/
│   │   ├── train_3d.py                    ← Entrenamiento 3D actual
│   │   └── train.py                       ← Entrenamiento 2D original (puede no existir)
│   └── evaluation/
│       ├── evaluate.py                    ← Evaluación 2D completa (puede no existir)
│       └── analyze_domain_bias.py         ← Análisis sesgo (puede no existir)
├── configs/
│   ├── train_3d.yaml                      ← Configuración 3D (activo)
│   └── train.yaml                         ← Configuración 2D (legacy)
├── outputs/
│   ├── checkpoints/
│   │   ├── baseline_2d_best.pt            ← Checkpoint baseline 2D (entrenado)
│   │   └── baseline_2d_history.json       ← Historial baseline 2D (11 épocas)
│   └── evaluation/
│       ├── test_results.json              ← Resultados baseline 2D en test
│       ├── test_predictions.json          ← Predicciones por muestra
│       └── domain_bias_analysis.json      ← Análisis sesgo BraTS vs IXI
├── data/
│   ├── processed/
│   │   ├── positives/                     ← 1167 .npz (580 BraTS + 587 UPENN)
│   │   ├── negatives/                     ← 577 .npz (IXI skull-stripped)
│   │   └── preprocessing_summary.json
│   ├── raw/
│   │   ├── brats/                         ← NIfTI BraTS (intermedios)
│   │   ├── ixi_stripped/                  ← NIfTI IXI skull-stripped
│   │   └── upenn/                         ← NIfTI UPENN (pendiente de generar)
│   └── splits.json                        ← Split actual (solo BraTS+IXI, 976 muestras)
├── notebooks/
│   └── explorar_processed_dinamico.ipynb  ← Exploración interactiva del dataset
└── docs/
    ├── revision_tecnica.md
    └── evolucion_respecto_primera_entrega.md (este fichero)
```

**Ficheros redundantes / pendientes de limpiar:**
- `preprocess_volumes.py` — monolítico original, funciones duplicadas en `base_preprocessing.py`
- `convert_dicom_to_nifti_upenn.py` (raíz de `preprocessing/`) — puede solaparse con `conversion/dicom_to_nifti.py`
- Scripts 2D (`baseline_2d.py`, `train.py`, `evaluate.py`, `analyze_domain_bias.py`, `transforms.py`, `dataset.py`) — probablemente eliminados; solo quedan sus outputs en `outputs/`

---

## 9. Tabla resumen de cambios

| Apartado del TFG | 1ª entrega | Estado actual | Cambio | Impacto | Cómo explicar en la memoria |
|---|---|---|---|---|---|
| Dataset positivo | BraTS 2021 (580) | BraTS 2021 (580) | Sin cambio | — | Igual que en la 1ª entrega |
| Dataset negativo | IXI (577) | IXI skull-stripped (577) | Skull-stripping completado | Corrección de sesgo crítico | "Se verificó que HD-BET homogeneiza la distribución de fondo entre BraTS e IXI" |
| Dataset externo | UPENN-GBM (solo test) | UPENN preprocesado en positives/ | **Cambio mayor** | Elimina evaluación externa si se mezcla | Decisión pendiente: mantener separado o justificar incorporación |
| Modalidades | T1+T2 | T1+T2 | Sin cambio | — | Igual que en la 1ª entrega |
| Preprocesado pipeline | Previsto | Ejecutado y verificado | Completado | Positivo | Describir pipeline completo como implementado |
| Skull-stripping IXI | Previsto | Completado, verificado empíricamente | Completado | Corrección de sesgo | Incluir verificación cuantitativa (% voxels a cero) |
| Balanceo de clases | ~1:1 (580 vs 577) | 2:1 si UPENN en train (1167 vs 577) | Desbalanceo introducido | Requiere corrección | Mencionar class_weights auto en train_3d.py |
| Baseline 2D | CNN sencilla | ResNet18 pseudo-2D, evaluado | Completado y mejorado | Positivo | Reportar AUC=0.9996, sens=1.0 post-corrección sesgo |
| Modelo 3D | Previsto | CNN 3D propia implementada | En curso (no entrenado) | Positivo si se entrena | Describir arquitectura BrainTumorCNN3D |
| Transfer learning | No especificado | Baseline usó ImageNet; 3D desde cero | Cambio en 3D | Justificable | Justificar por diferencia de dominio MRI vs imágenes naturales |
| Loss | CrossEntropy | CrossEntropy (ponderada en 3D) | Mejora | Positivo | Mencionar class_weights para compensar desbalanceo |
| Optimizador | Adam | AdamW (3D) | Mejora menor | Positivo | Mencionar AdamW como variante con regularización incorporada |
| Batch size | 16 | 1 (3D, por memoria) | Cambio técnico forzado | Limitación GPU | "La naturaleza volumétrica impone batch_size=1 con AMP" |
| Métricas durante train | AUC, sens, spec | Solo accuracy + loss (3D) | **Regresión** | Negativo — debe corregirse | Pendiente escribir evaluate_3d.py |
| Early stopping | val_AUC | val_loss (3D) | Cambio menor | Diverge de métrica clínica | Mención como limitación técnica |
| Evaluación completa | Prevista | Implementada para baseline 2D | Parcial | Baseline evaluado correctamente | Reportar resultados del baseline como referencia |
| Calibración | Prevista | No implementada | Pendiente | Limitación | Mencionar como trabajo futuro |
| Evaluación externa | UPENN-GBM | No ejecutada | Pendiente | — | Pendiente tras entrenar modelo 3D |
| Análisis por institución | BraTS por hospital | Implementado para baseline | Parcial | — | Mencionar resultados del domain_bias_analysis.json |

---

## 10. Texto para la memoria: "Evolución del planteamiento respecto a la primera entrega"

El presente apartado describe los cambios producidos en el desarrollo del sistema respecto al planteamiento aprobado en la primera entrega del trabajo.

### 10.1 Preprocesado y corrección del sesgo de dominio

El aspecto más relevante completado durante la fase de implementación ha sido la identificación y corrección del sesgo de dominio entre BraTS 2021 e IXI. En el planteamiento inicial se reconocía este riesgo de forma explícita y se proponía el preprocesado homogéneo como medida de mitigación. Durante los primeros experimentos de entrenamiento se constató empíricamente que el modelo baseline alcanzaba AUC=1.0 desde la primera época, resultado que el análisis posterior atribuyó a la diferencia en el porcentaje de voxels de fondo entre ambos datasets: BraTS llegaba con skull-stripping de fábrica (aproximadamente 80% de voxels a cero) mientras que IXI no había sido procesado con este método (aproximadamente 18% de voxels a cero). El modelo no aprendía señal tumoral, sino la presencia o ausencia de tejido craneal.

Para corregir este problema se aplicó skull-stripping a todas las imágenes IXI mediante HD-BET, herramienta basada en deep learning específicamente diseñada para resonancia magnética cerebral. Tras este proceso, la distribución de voxels de fondo en IXI pasó a ser comparable a la de BraTS (81-84% de voxels a cero), eliminando la diferencia más obvia entre dominios. La verificación cuantitativa de esta corrección sobre una muestra de los ficheros procesados confirmó que el preprocesado homogéneo se había aplicado correctamente.

El preprocesado definitivo consiste en cinco pasos aplicados a todos los volúmenes de todos los datasets: reorientación a espacio RAS mediante nibabel, remuestreo a resolución de 1 mm isótropo mediante interpolación lineal con scipy.ndimage.zoom, recorte o relleno con ceros hasta alcanzar una forma fija de 192×224×192 vóxeles centrada en el volumen original, y normalización de intensidades mediante z-score calculado exclusivamente sobre vóxeles no nulos. Cada volumen resultante se almacena en formato NumPy comprimido (.npz) con las modalidades T1 y T2 y la etiqueta binaria correspondiente.

### 10.2 Implementación y evaluación del baseline

El modelo baseline planteado en la primera entrega fue implementado como una adaptación de ResNet18 con pesos preentrenados en ImageNet. Dado que ResNet18 espera imágenes 2D con 3 canales, se adoptó una estrategia de apilamiento de cortes axiales: se extraen 16 cortes axiales centrales de cada modalidad (T1 y T2) y se apilan como 32 canales de entrada, reformulando el problema volumétrico como una clasificación 2D multi-canal. La primera capa convolucional de la red se adaptó de 3 a 32 canales replicando los pesos pretrained de forma ponderada. El resto de la red se fine-tuneó completamente.

Tras la corrección del sesgo de dominio mediante skull-stripping homogéneo, el baseline fue reentrenado y evaluado con métricas completas sobre el conjunto de test (174 muestras: 87 BraTS y 87 IXI). Los resultados obtenidos fueron: ROC-AUC de 0.9996, PR-AUC de 0.9996, sensibilidad de 1.000 y especificidad de 0.965 al umbral por defecto de 0.5. Al fijar el umbral para garantizar un mínimo del 95% de sensibilidad, la especificidad resultante fue de 1.000. El análisis de distribución de scores mostró que existe un ligero solapamiento entre las predicciones de BraTS e IXI (gap negativo de -0.147), indicando que el modelo no discrimina trivialmente por dominio de procedencia, si bien la separación sigue siendo elevada y no puede descartarse la existencia de algún sesgo residual.

### 10.3 Implementación del modelo principal 3D

En respuesta al objetivo de desarrollar un modelo principal de mayor capacidad espacial que el baseline 2D, se implementó una CNN tridimensional propia denominada BrainTumorCNN3D. La arquitectura consta de cuatro bloques convolucionales tridimensionales, cada uno compuesto por dos capas Conv3D con BatchNorm3D y ReLU, seguidos de MaxPool3D para reducción progresiva de resolución. El bloque final utiliza AdaptiveAvgPool3D para colapsar el mapa de características a un vector de dimensión fija, que es procesado por un clasificador fully-connected con dropout. El modelo opera sobre volúmenes de (2, 128, 160, 128) vóxeles, donde los 2 canales corresponden a T1 y T2 y el recorte a 128×160×128 se aplica para adecuarse a las limitaciones de memoria de la GPU disponible.

A diferencia del baseline, este modelo no emplea pesos preentrenados, lo cual está justificado por la naturaleza fundamentalmente diferente de las imágenes de resonancia magnética cerebral respecto a las imágenes naturales de ImageNet, así como por el volumen de datos disponibles (superior a 1.000 muestras con ambos datasets). El entrenamiento utiliza AdamW con tasa de aprendizaje 1×10⁻⁴, pesos de clase automáticos calculados inversamente proporcionales a la frecuencia de cada clase, y mixed precision (AMP) para optimizar el uso de memoria. El modelo principal se encuentra implementado y validado en términos de inferencia, pero pendiente de entrenamiento completo y evaluación formal.

### 10.4 Incorporación de UPENN-GBM y NKI Rockland: reformulación hacia robustez multi-fuente

Durante el desarrollo se ha producido una desviación significativa respecto al planteamiento de la primera entrega en lo relativo a la composición del conjunto de datos. UPENN-GBM, que originalmente se contemplaba como conjunto de evaluación externa cross-dataset, ha sido incorporado al pool de entrenamiento tras su preprocesado homogéneo. Los 587 volúmenes resultantes se han colocado en `data/processed/positives/` con etiqueta positiva, junto a los 580 volúmenes de BraTS 2021.

Esta decisión responde a una reformulación parcial del objetivo metodológico. En lugar de validar la generalización del sistema mediante un único conjunto externo, se ha optado por entrenar un clasificador expuesto desde el inicio a la variabilidad inter-dataset, con dos fuentes positivas independientes que presentan diferencias en protocolos de adquisición, características demográficas e instituciones de origen. La hipótesis subyacente es que un modelo entrenado sobre una distribución más heterogénea aprenderá representaciones menos dependientes de firmas específicas de un dataset concreto, lo que constituye una forma alternativa, aunque más débil que la cross-dataset estricta, de abordar el problema del sesgo de dominio.

De forma coherente con esta línea, el lado negativo del dataset se ha ampliado mediante la incorporación de NKI Rockland, un dataset de sujetos sanos descargado del bucket público de FCP-INDI mediante el script `download_nki_rockland.py`, que filtra exclusivamente sujetos con T1w y T2w en la misma sesión BIDS. Tras el skull-stripping con HD-BET y el preprocesado homogéneo, se han añadido 523 volúmenes adicionales con etiqueta negativa, resultando en una distribución final de 1.167 positivos (BraTS + UPENN) y 1.100 negativos (IXI + NKI Rockland) sobre un total de 2.267 muestras. Esta composición restablece el equilibrio de clases que la incorporación de UPENN había roto y aporta variabilidad multi-fuente también en el lado negativo.

La consecuencia metodológica directa, que se reconoce explícitamente como limitación de este trabajo, es que el repositorio no dispone actualmente de un conjunto de evaluación externa cross-dataset. La validación externa estricta queda planteada como línea de trabajo inmediata, condicionada a la identificación y preprocesado de un dataset adicional con tumores cerebrales que no haya sido visto durante el entrenamiento (candidatos contemplados: TCGA-GBM, REMBRANDT u otros datasets disponibles en TCIA y Kaggle). Hasta entonces, la generalización del sistema se evalúa exclusivamente sobre el conjunto de test interno extraído mediante partición estratificada del pool multi-fuente, lo que aporta evidencia sobre la consistencia del clasificador dentro de la distribución de entrenamiento pero no sobre su transferibilidad a dominios completamente nuevos.

### 10.5 Estado de la evaluación

La evaluación completa del baseline 2D ha sido implementada e incluye todas las métricas previstas a excepción de la calibración: ROC-AUC, PR-AUC, sensibilidad, especificidad, F1-score, análisis por umbral operativo y análisis de sesgo de dominio por distribución de scores y por institución de origen de IXI (tests de Kruskal-Wallis y Mann-Whitney). La calibración queda como tarea pendiente antes de la defensa. La evaluación del modelo 3D, que seguirá la misma estructura, está pendiente de su entrenamiento.
