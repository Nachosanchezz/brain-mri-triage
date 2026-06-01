# Preprocesado de datasets

Comandos para reconstruir `data/processed` con el formato comun `.npz`.

## Idea general

El objetivo del preprocesado es que todos los datasets tengan el mismo formato
antes de entrenar el modelo. Aunque BraTS, UPENN e IXI vienen de fuentes
distintas, al final cada caso debe quedar como una muestra `.npz` con:

```text
T1 + T2 + label
```

La red no debe aprender diferencias artificiales entre datasets, como la
presencia de craneo, orientaciones distintas, resoluciones distintas o tamanos
de volumen distintos. Por eso se aplica una pipeline comun:

```text
DICOM -> NIfTI -> RAS -> 1 mm isotropico -> crop/pad -> z-score -> .npz
```

El formato final guarda el volumen 3D completo. El modelo definitivo
(`src/models/cnn3d.py::BrainTumorCNN3D`) es una CNN **3D volumetrica** que
consume el par T1+T2 como dos canales; durante la carga, el `Dataset` de PyTorch
recorta un subvolumen a `(128, 160, 128)` por restricciones de memoria GPU.

## Que hace cada paso

**DICOM a NIfTI**

Los DICOM suelen venir como muchas imagenes 2D por serie. NIfTI (`.nii.gz`) es
mas comodo para neuroimagen porque guarda el volumen 3D completo y su geometria
espacial. Este paso se hace con `dcm2niix`.

**Skull-strip**

BraTS y UPENN ya vienen sin craneo o con versiones procesadas. IXI es un dataset
de sujetos sanos y puede venir con craneo. Si dejamos craneo en negativos y no
en positivos, el modelo podria aprender "craneo = no tumor" en vez de aprender
senales tumorales. Por eso IXI se pasa por HD-BET antes de crear los `.npz`.

**Orientacion RAS**

Distintos datasets pueden usar convenciones espaciales diferentes. Reorientar a
RAS hace que los ejes anatomicos sean consistentes entre muestras.

**Remuestreo a 1 mm isotropico**

Cada dataset puede tener distinto espaciado entre voxels. Se remuestrea todo a
`1.0 x 1.0 x 1.0 mm` para que una lesion tenga una escala comparable entre
datasets.

**Crop/pad a shape fija**

Las redes necesitan tensores del mismo tamano. Todos los volumenes se ajustan a:

```text
(192, 224, 192)
```

Si el volumen es mas grande, se recorta centrado. Si es mas pequeno, se rellena
con ceros.

**Normalizacion z-score**

Las intensidades MRI no tienen una escala absoluta comparable entre maquinas.
Se normaliza cada modalidad usando z-score sobre voxels no cero:

```text
(voxel - media) / desviacion_tipica
```

Esto se hace por separado para T1 y T2.

**Formato `.npz`**

El `.npz` es un contenedor NumPy comprimido. Es rapido de cargar y permite
guardar las dos modalidades y metadatos juntos.

## 1. Activar entorno

```powershell
env\Scripts\activate
```

## 2. Convertir DICOM a NIfTI

BraTS:

```powershell
python src\preprocessing\conversion\dicom_to_nifti.py brats
```

Salida:

```text
data/raw/brats/<paciente>/*_t1w.nii.gz
data/raw/brats/<paciente>/*_t2w.nii.gz
```

UPENN:

```powershell
python src\preprocessing\conversion\dicom_to_nifti.py upenn
```

Salida:

```text
data/raw/upenn/<paciente>/*_t1.nii.gz
data/raw/upenn/<paciente>/*_t2.nii.gz
```

## 3. Skull-strip de IXI

Si ya existe `data/raw/ixi_stripped`, se puede saltar este paso.

```powershell
python src\preprocessing\preprocess_ixi.py --skull-strip
```

Salida intermedia:

```text
data/raw/ixi_stripped/t1/*.nii.gz
data/raw/ixi_stripped/t2/*.nii.gz
```

## 4. Crear `.npz` procesados

BraTS positivos:

```powershell
python src\preprocessing\preprocess_brats.py
```

IXI negativos:

```powershell
python src\preprocessing\preprocess_ixi.py
```

UPENN positivos:

```powershell
python src\preprocessing\preprocess_upenn.py
```

NKI Rockland negativos:

Primero descarga solo sujetos con T1w y T2w completos en la misma sesion:

```powershell
python src\preprocessing\download_nki_rockland.py --aws-links C:\Users\1cnac\Downloads\aws_links.csv --out-dir data\raw\nki_rockland --include-json
```

Para probar antes sin descargar:

```powershell
python src\preprocessing\download_nki_rockland.py --aws-links C:\Users\1cnac\Downloads\aws_links.csv --out-dir data\raw\nki_rockland --limit 3 --dry-run --include-json
```

Despues crea los `.npz` como negativos:

```powershell
python src\preprocessing\preprocess_nki_rockland.py --skull-strip
```

Si el skull-strip ya esta hecho y quieres recrear solo los `.npz`:

```powershell
python src\preprocessing\preprocess_nki_rockland.py --use-stripped
```

Salida final:

```text
data/processed/positives/*.npz
data/processed/negatives/*.npz
```

Cada `.npz` contiene:

```text
t1         -> (192, 224, 192) float32
t2         -> (192, 224, 192) float32
label      -> 1 tumor, 0 no tumor
dataset    -> brats, upenn, ixi, nki_rockland
subject_id -> identificador del caso
source_t1  -> ruta NIfTI original
source_t2  -> ruta NIfTI original
```

Las clases son:

```text
label=1 -> tumor / masa tumoral
label=0 -> no tumor
```

Estado actual del pool principal (`data/processed/`, ver
`preprocessing_summary.json`):

```text
positives -> BraTS (580) + UPENN-GBM (587)   = 1167
negatives -> IXI (577) + NKI Rockland (523)  = 1100
total                                         = 2267
```

> Nota: el dataset intra-dominio BTC_preop (OpenNeuro ds001226) se preprocesa
> aparte con `preprocess_btc.py` a `data/processed_btc/` y es **T1-only**
> (no se usa en este pool de 2 canales).

## 5. Actualizar resumen

```powershell
python src\preprocessing\dataset\summarize_processed.py --write
```

Salida:

```text
data/processed/preprocessing_summary.json
```

El summary guarda el recuento total, el recuento por clase y el origen de cada
dataset. Tambien guarda ejemplos con rutas relativas como:

```text
positives/BraTS2021_00000.npz
positives/UPENN-GBM-00001.npz
negatives/IXI002-Guys-0828.npz
negatives/NKI-A00008326-BAS2.npz
```

## 6. Regenerar splits

```powershell
python -c "import sys; sys.path.insert(0, '.'); from src.data.dataset_3d import create_splits; create_splits(seed=42)"
```

Salida:

```text
data/splits.json
```

Los splits se hacen **estratificados por `(dataset, label)`** y **agrupados por
`subject_id`** (`create_splits` en `src/data/dataset_3d.py`):

```text
train -> 70 %
val   -> 15 %
test  -> 15 %
```

La estratificacion conjunta por dataset Y etiqueta evita que un re-split deje
un dataset sub/sobre-representado en alguna particion; la agrupacion por
`subject_id` evita el leakage de un mismo sujeto entre train/val/test.

## 7. Comandos utiles

Ver ayuda del conversor:

```powershell
python src\preprocessing\conversion\dicom_to_nifti.py --help
```

Filtrar manifest TCIA de UPENN para quedarse con T1/T2:

```powershell
python src\preprocessing\preprocess_upenn.py --filter-manifest
```

Filtrar manifest sin pedir confirmacion:

```powershell
python src\preprocessing\preprocess_upenn.py --filter-manifest --yes
```

Hacer una prueba pequena sin procesar todo:

```powershell
python src\preprocessing\preprocess_brats.py --limit 2
python src\preprocessing\preprocess_ixi.py --limit 2
python src\preprocessing\preprocess_upenn.py --limit 2
```
