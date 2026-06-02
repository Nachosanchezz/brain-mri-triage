# Plan de trabajo para remontar el TFG `brain-mri-triage`

> Guía de estudio y ejecución, pensada para seguir **en orden**, desde hoy.
> Objetivo: que **domines** el proyecto (no solo que lo entregues) y puedas
> defender cada número ante el tribunal.

---

## Principio rector: ¿por qué este orden y no otro?

1. **De la historia hacia los detalles, no al revés.** Primero te apropias de la
   *narrativa* (qué pasó y por qué); luego cada dato encuentra su sitio. Si empiezas
   por el código suelto o por papers, acumulas piezas sin saber dónde encajan.
2. **De la causa raíz hacia las consecuencias.** El problema vive en los **datos**
   (la etiqueta = el dataset). El modelo y la auditoría solo lo *confirman*. Por eso
   entiendes primero los datos, después el modelo, después las pruebas.
3. **Entender antes de escribir.** No redactes un apartado cuyo código/cifra no
   sepas explicar. La fuerza de este TFG es que el pipeline está **bien hecho** y aun
   así el resultado es espurio: para defenderlo tienes que poder demostrar que **no
   es un bug**, es un confound.
4. **Lo lento, en paralelo.** Leer papers consume tiempo de calendario. Arráncalo el
   día 1 y deja que avance en segundo plano mientras trabajas el código.

> Regla de una frase: **«primero la historia, luego los datos, luego el modelo,
> luego las pruebas, y la teoría en paralelo».**

---

## El flujo en 6 fases

| Fase | Objetivo (qué consigues) | Qué LEER (repo) | Qué CÓDIGO entender a fondo | Teoría / paper | Lo dominas si sabes explicar… |
|---|---|---|---|---|---|
| **0. La historia** *(hoy, 1–2 h)* | Contar el TFG en 5 frases | `reconstruccion_evolucion_tfg.md` §1, §11, §23, §24 · `auditoria_resultados_sospechosos.md` §1 y §11 | — (nada aún) | — | …por qué un AUC≈1.0 en este problema es **sospechoso**, no un éxito |
| **1. La causa raíz: datos y confound** | Saber **de dónde** sale el problema | `reconstruccion…` §6 y §7 · `src/preprocessing/README.md` · `audit_leakage.json` (`label_equals_dataset`) | `dataset_3d.py` (`create_splits`, `BrainMRI3DDataset`) · `base_preprocessing.py` (`normalize_intensity`/Otsu) · `audit_splits.py` | Geirhos 2020 (abstract+intro) | …por qué `label ≡ dataset` y por qué un split **perfecto** NO salva el experimento |
| **2. El modelo y el entrenamiento** | Defender que el pipeline está **bien hecho** | `justificacion_cambios.md` (los 8 pasos, entero) | `cnn3d.py` (entero, es corto) · `train_3d.py` (`run_epoch`, `binary_auc`, selección de checkpoint) · `configs/train_3d.yaml` | GroupNorm (Wu & He) · AUC = Mann–Whitney · `BCEWithLogitsLoss` | …la red capa a capa, por qué **GroupNorm** con batch=1, y cómo se elige el checkpoint |
| **3. La evaluación y por qué engaña** | Leer una métrica con ojo crítico | `reconstruccion…` §10 · `cnn3d_test_results.json` | `evaluate_3d.py` (`positive_probability`, métricas, `per_dataset`) · `threshold_analysis.py` | Sensibilidad/especificidad, PR-AUC, umbral operativo | …por qué la **AUC intra-dataset = NaN** y por qué la varianza ~0 dentro de cada dataset es una alarma |
| **4. La auditoría, pieza a pieza** *(el corazón)* | Entender cada prueba: qué hipótesis falsa descarta | ver desglose 4a–4e abajo | ver desglose 4a–4e | Bootstrap IC95 % · silhouette score | …para **cada** prueba, qué hipótesis intenta falsar y qué demostró |
| **5. Marco teórico** *(en paralelo desde fase 1)* | Encuadrar tu hallazgo en la literatura | `docs/referencias.md` | — | Geirhos 2020, Zech 2018, DeGrave 2021 | …que tu confound es un caso de *shortcut learning* y citar 2 antecedentes |
| **6. Escribir** | Memoria | `esquema_memoria_anotado.md` | — | — | …empezar por **Cap. 7 (Resultados)**, el más anclado en datos |

---

### Desglose de la Fase 4 (la auditoría) — en orden narrativo

| Sub | Prueba | Qué CÓDIGO | Qué demuestra (en una frase) | Cifra clave |
|---|---|---|---|---|
| **4a** | Tiny baseline + clf de dataset | `audit_leakage.py` | La etiqueta se decodifica **sin red ni anatomía** → la señal es de dominio | LogReg AUC **1.0**; clf dataset acc **0.985** |
| **4b** | LODO (fuera de dominio) | `make_lodo_splits.py` · `audit_lodo.py` · `run_lodo.sh` · `train_lodo.yaml` | El modelo **no generaliza**; en una dirección **invierte** la regla | CNN A **0.624** / B **0.201** |
| **4c** | Intra-dominio BTC | `preprocess_btc.py` · `btc_tiny_baseline.py` · `btc_cnn_kfold.py` · `run_btc_chain.sh` | Sin confound (ambas clases, mismo escáner), el rendimiento **cae al azar** | CNN **0.404**, IC95 [0.21, 0.62] |
| **4d** | Embeddings (espacio latente) | `embeddings_tsne.py` · `embeddings_intraclass.py` | La red codifica **procedencia** incluso dentro de una misma clase | IXI-vs-NKI (sanos) CV-AUC **0.998** |
| **4e** | Grad-CAM (interpretabilidad) | `gradcam_3d.py` | A qué atiende el modelo (apoyo **cualitativo**, con cautela) | — (figuras) |

> **Orden 4a→4e = orden del Cap. 7.** Es un embudo: primero descartas lo trivial
> (4a), luego pruebas que no generaliza (4b), luego mides sin trampa (4c), y por
> último abres la caja (4d–4e). Entiéndelo en este orden porque así lo vas a contar.

---

## Qué código entender, por niveles de profundidad

| Nivel | Qué significa | Archivos |
|---|---|---|
| **🔴 A fondo (línea a línea; te lo pueden preguntar)** | Saber explicar cada parte y por qué está así | `cnn3d.py` · `dataset_3d.py::create_splits` · `audit_leakage.py` · `btc_cnn_kfold.py` |
| **🟡 Lógica + entradas/salidas** | Saber qué hace, qué consume y qué produce | `train_3d.py` · `evaluate_3d.py` · `audit_lodo.py` · `make_lodo_splits.py` · `embeddings_tsne.py` · `embeddings_intraclass.py` · `base_preprocessing.py` (`normalize_intensity`) |
| **🟢 Saber qué hace (sin memorizar)** | Identificar su papel | `threshold_analysis.py` · `gradcam_3d.py` · `make_plots.py` · `make_extra_figures.py` · `consolidate_results.py` · `preprocess_*.py` · `download_nki_rockland.py` |

---

## Lista de lectura (papers) y cuándo

| Prioridad | Paper (clave en `referencias.md`) | Para qué lo lees | Cuándo |
|---|---|---|---|
| 🔴 1 | Geirhos 2020 — *Shortcut learning* | Es el marco exacto de tu tesis | Fase 1 (en paralelo) |
| 🔴 2 | Zech 2018 — confound entre hospitales (rayos X) | Antecedente clínico de tu mismo fenómeno | Fase 4 |
| 🔴 3 | DeGrave 2021 — atajos en COVID por imagen | Segundo antecedente fuerte | Fase 4 |
| 🟡 4 | HD-BET (Isensee 2019), dcm2niix, GroupNorm (Wu&He) | Citar herramientas que usaste | Fase 2 |
| 🟡 5 | Datasets: BraTS, UPENN-GBM, IXI, NKI, ds001226 | Citas obligatorias de datos | Fase 1 / al escribir Cap. 4 |
| 🟢 6 | Survey DL médico (Litjens) · triaje radiológico | Contexto de Intro y 3.3/3.6 | Antes del Cap. 3 |

> Lee primero **abstract + figuras + conclusiones** de cada paper; la lectura
> profunda solo de Geirhos, Zech y DeGrave. No te pierdas leyendo de más.

---

## Empieza HOY (sesión 0, 1–2 horas)

1. Lee `reconstruccion_evolucion_tfg.md` §1 (resumen) y §24 (conclusión).
2. Lee `auditoria_resultados_sospechosos.md` §1 (resumen ejecutivo).
3. Escribe **a mano, sin mirar**, 5 frases que respondan:
   - ¿Qué querías construir?
   - ¿Qué resultado dio?
   - ¿Por qué era sospechoso?
   - ¿Qué demostró la auditoría?
   - ¿Cuál es la aportación final del TFG?
4. Si puedes responder esas 5 sin titubear, has terminado la Fase 0 y estás listo
   para la Fase 1.

---

## Autoevaluación tipo tribunal (úsala al cerrar cada fase)

- **Tras Fase 1:** «¿Por qué tu split por sujeto, que es correcto, no resuelve el
  problema?» → porque el problema no es *cómo repartes*, sino *qué representa la
  etiqueta*.
- **Tras Fase 2:** «¿Cómo sabes que el AUC≈1.0 no es un bug de tu código?» → porque
  un baseline lineal trivial lo reproduce y la evaluación está verificada.
- **Tras Fase 3:** «¿Por qué no puedes dar la AUC por dataset?» → porque cada dataset
  es mono-clase; esa imposibilidad es, en sí, prueba del confound.
- **Tras Fase 4:** «Si el modelo no detecta tumor, ¿qué detecta?» → la procedencia /
  el dominio (LODO se invierte, intra-dominio cae al azar, el latente separa
  cohortes de la misma clase).
- **Tras Fase 5:** «¿Es esto un caso aislado?» → no; es *shortcut learning*, un
  fenómeno documentado (Zech, DeGrave).

> Si respondes estas cinco con soltura, **tienes el TFG defendido** aunque aún no
> esté escrito. Escribirlo es entonces solo trasladar lo que ya dominas.

---

## Ritmo sugerido (orientativo)

| Semana | Foco |
|---|---|
| 1 | Fases 0–1 + arrancar lectura de Geirhos |
| 2 | Fases 2–3 |
| 3 | Fase 4 (auditoría completa) + Zech/DeGrave |
| 4 | Fase 5 + empezar a redactar Cap. 7 (Resultados) |
| 5+ | Redacción: 7 → 6 → 5 → 4 → 8 → 3 → 1/2/9/10 (ver `esquema_memoria_anotado.md`) |

*El ritmo es ajustable; lo fijo es el ORDEN de las fases, no su duración.*
