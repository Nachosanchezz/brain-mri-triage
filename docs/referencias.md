# Referencias bibliográficas — candidatas (TFG `brain-mri-triage`)

> **AVISO IMPORTANTE.** Esta es una lista de **candidatas a citar**, no una
> bibliografía final. Las entradas se basan en obras conocidas, pero **los datos
> bibliográficos (autores, año, revista/conferencia, volumen, DOI) NO están
> verificados** y pueden contener errores de memoria. **Antes de citar cualquiera:
> localiza la fuente original, confirma todos los campos y NO cites de memoria.**
>
> Convención: `[clave]` sugerida para BibTeX · **Verificar:** qué comprobar ·
> **Uso:** apartado(s) de la memoria (ver `docs/esquema_memoria_anotado.md`).
>
> Prioridad:
> - **🔴 OBLIGATORIA** = describe datos o herramientas que realmente usaste; el
>   tribunal espera la cita. No es opcional.
> - **🟡 MARCO** = sustenta la discusión metodológica (confound, shortcut learning).
> - **🟢 CONTEXTO** = clínica/encuadre general; sustituible por equivalentes.

---

## 1. Datasets utilizados  🔴 OBLIGATORIAS

Estas debes citarlas sí o sí, porque son los datos sobre los que trabajaste.
Revisa además las **condiciones de cita exigidas por cada dataset** (muchos
challenges piden citar 2-3 artículos concretos y reconocer la fuente).

- `[brats_menze2015]` — Menze et al., «The Multimodal Brain Tumor Image Segmentation
  Benchmark (BRATS)», *IEEE TMI*.
  **Verificar:** año (2015), volumen/páginas, DOI. **Uso:** 3.4, 4.1.
- `[brats_bakas2017]` — Bakas et al., datos de segmentación y supervivencia
  (TCGA-GBM/LGG), *Scientific Data*.
  **Verificar:** año, DOI; si aplica a tu versión de BraTS. **Uso:** 4.1.
- `[brats_baid2021]` — Baid et al., «The RSNA-ASNR-MICCAI BraTS 2021 Benchmark…»,
  arXiv.
  **Verificar:** identificador arXiv, autores. **Uso:** 4.1 (es **tu** versión: BraTS 2021).
- `[upenn_bakas2022]` — Bakas et al., «The University of Pennsylvania glioblastoma
  (UPenn-GBM) cohort», *Scientific Data*.
  **Verificar:** año (2022), DOI; cita TCIA asociada. **Uso:** 4.1.
- `[tcia_clark2013]` — Clark et al., «The Cancer Imaging Archive (TCIA)…»,
  *J. Digital Imaging*.
  **Verificar:** año (2013), DOI. **Uso:** 4.1 (fuente de UPENN-GBM).
- `[ixi_dataset]` — IXI Dataset, Imperial College / brain-development.org.
  **Verificar:** forma de cita recomendada (suele ser la URL del proyecto; no hay
  paper único). **Uso:** 4.1.
- `[nki_nooner2012]` — Nooner et al., «The NKI-Rockland Sample…», *Frontiers in
  Neuroscience*.
  **Verificar:** año (2012), volumen, DOI; y cita de FCP-INDI / 1000 Functional
  Connectomes si aplica. **Uso:** 4.1.
- `[btc_aerts_ds001226]` — Aerts et al., dataset BTC_preop, **OpenNeuro ds001226**.
  **Verificar:** DOI del dataset (`10.18112/openneuro.ds001226.vX.X.X`), artículo(s)
  asociado(s) de Aerts et al. (estudios de *brain network modelling* en tumores),
  licencia CC0. **Uso:** 4.1, 6.6, 7.5.

---

## 2. Herramientas y métodos técnicos  🔴 OBLIGATORIAS

Las que forman parte de tu pipeline o de tus análisis.

- `[hdbet_isensee2019]` — Isensee et al., «Automated brain extraction of multisequence
  MRI using artificial neural networks» (HD-BET), *Human Brain Mapping*.
  **Verificar:** año (2019), DOI. **Uso:** 5.1 (skull-stripping de IXI/NKI/BTC).
- `[nnunet_isensee2021]` — Isensee et al., «nnU-Net: a self-configuring method…»,
  *Nature Methods*.
  **Verificar:** año (2021), DOI; solo si lo mencionas (HD-BET se apoya en nnU-Net).
  **Uso:** 5.1 (opcional).
- `[dcm2niix_li2016]` — Li et al., «The first step for neuroimaging data analysis:
  DICOM to NIfTI conversion» (dcm2niix), *J. Neuroscience Methods*.
  **Verificar:** año (2016), DOI. **Uso:** 5.1.
- `[pytorch_paszke2019]` — Paszke et al., «PyTorch: An Imperative Style,
  High-Performance Deep Learning Library», *NeurIPS*.
  **Verificar:** año (2019). **Uso:** 5.6.
- `[adamw_loshchilov2019]` — Loshchilov & Hutter, «Decoupled Weight Decay
  Regularization» (AdamW), *ICLR*.
  **Verificar:** año (2019). **Uso:** 5.3.
- `[groupnorm_wu2018]` — Wu & He, «Group Normalization», *ECCV*.
  **Verificar:** año (2018), DOI. **Uso:** 5.2 (justifica GroupNorm con batch pequeño).
- `[gradcam_selvaraju2017]` — Selvaraju et al., «Grad-CAM: Visual Explanations from
  Deep Networks…», *ICCV*.
  **Verificar:** año (2017), DOI. **Uso:** 6.7, 7.6.
- `[tsne_maaten2008]` — van der Maaten & Hinton, «Visualizing Data using t-SNE»,
  *JMLR*.
  **Verificar:** año (2008), volumen. **Uso:** 6.7, 7.6.
- `[sklearn_pedregosa2011]` — Pedregosa et al., «Scikit-learn: Machine Learning in
  Python», *JMLR*.
  **Verificar:** año (2011). **Uso:** 6.4 (tiny baseline, métricas).

---

## 3. Marco teórico: confound, *shortcut learning* y sesgo de dominio  🟡 MARCO

El núcleo conceptual de tu discusión. Sin estas, tu hallazgo parece anecdótico;
con ellas, es un caso de un fenómeno documentado.

- `[shortcut_geirhos2020]` — Geirhos et al., «Shortcut learning in deep neural
  networks», *Nature Machine Intelligence*.
  **Verificar:** año (2020), volumen, DOI. **Uso:** 3.5, 8.1, 8.4. **(La más importante.)**
- `[zech2018_confound]` — Zech et al., «Variable generalization performance of a deep
  learning model to detect pneumonia in chest radiographs: a cross-sectional study»,
  *PLoS Medicine*.
  **Verificar:** año (2018), DOI. **Uso:** 3.5, 8.4 (confound por hospital/dominio).
- `[degrave2021_covid]` — DeGrave et al., «AI for radiographic COVID-19 detection
  selects shortcuts over signal», *Nature Machine Intelligence*.
  **Verificar:** año (2021), DOI. **Uso:** 3.5, 8.4 (atajos en imagen médica).
- `[wynants / generalization]` — (opcional) revisión sobre fallos de generalización
  externa de modelos de DL médico.
  **Verificar:** elegir una revisión sólida y reciente; confirmar datos. **Uso:** 3.5.

---

## 4. Armonización / adaptación de dominio  🟡 MARCO (líneas futuras)

Para la discusión de por qué la armonización **no** resuelve un confound del 100 %.

- `[combat_johnson2007]` — Johnson et al., «Adjusting batch effects in microarray
  expression data using empirical Bayes methods» (ComBat original), *Biostatistics*.
  **Verificar:** año (2007), DOI. **Uso:** 10.2.
- `[combat_fortin2017]` — Fortin et al., «Harmonization of multi-site diffusion/
  cortical data with ComBat», *NeuroImage*.
  **Verificar:** año/título exacto (hay dos artículos, dMRI y grosor cortical), DOI.
  **Uso:** 10.2.
- `[dann_ganin2016]` — Ganin et al., «Domain-Adversarial Training of Neural Networks»,
  *JMLR*.
  **Verificar:** año (2016). **Uso:** 10.2 (domain-adversarial; y su límite si
  dominio≡clase).

---

## 5. IA en imagen médica y radiología  🟢 CONTEXTO

- `[litjens2017_survey]` — Litjens et al., «A survey on deep learning in medical
  image analysis», *Medical Image Analysis*.
  **Verificar:** año (2017), DOI. **Uso:** 3.3.
- `[esteva2017 / esteva2019]` — Esteva et al., trabajos de referencia sobre DL en
  imagen clínica.
  **Verificar:** cuál encaja mejor (2017 dermatología, *Nature*; 2019 guía,
  *Nat Med*). **Uso:** 3.3.
- `[triage_oref / worklist]` — referencia sobre **priorización de worklist** /
  triaje radiológico con IA (p. ej. detección de hemorragia intracraneal priorizando
  lectura).
  **Verificar:** elegir un estudio clínico o de validación regulatoria sólido.
  **Uso:** 3.6, 8.3.
- `[clinical_glioma]` — referencia clínica sobre gliomas/meningiomas y papel de la RM
  (p. ej. clasificación WHO de tumores del SNC).
  **Verificar:** edición vigente de la clasificación WHO CNS y/o guía clínica.
  **Uso:** 3.1, 3.2.

---

## Notas de uso

1. **Gestiona con BibTeX/`.bib`** si la plantilla del TFG lo permite; las claves de
   arriba son sugerencias para ese `.bib`.
2. **Condiciones de cita de datasets:** BraTS, UPENN-GBM (TCIA) y OpenNeuro suelen
   exigir citar artículos concretos y/o reconocer la financiación. Revisa la página
   de cada dataset y cumple su requisito exacto.
3. **No infles la bibliografía.** Para un TFG de 40–50 pp., ~25–40 referencias bien
   elegidas y verificadas es razonable. Mejor pocas y correctas que muchas y dudosas.
4. **Marca el estado** según las verifiques (p. ej. añade ✅ al inicio de la entrada
   cuando confirmes todos sus campos).
