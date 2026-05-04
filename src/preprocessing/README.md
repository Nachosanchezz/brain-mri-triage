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

El formato final esta pensado para una CNN 2.5D: se guarda el volumen 3D
completo y despues el `Dataset` de PyTorch puede extraer cortes o grupos de
cortes 2D alrededor de una posicion.

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
dataset    -> brats, upenn, ixi
subject_id -> identificador del caso
source_t1  -> ruta NIfTI original
source_t2  -> ruta NIfTI original
```

Las clases son:

```text
label=1 -> tumor / masa tumoral
label=0 -> no tumor
```

Actualmente:

```text
positives -> BraTS + UPENN
negatives -> IXI
```

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
```

## 6. Regenerar splits

```powershell
@'
import sys
from pathlib import Path
sys.path.insert(0, str(Path("src/data").resolve()))
from dataset import create_splits
create_splits(seed=42)
'@ | python -
```

Salida:

```text
data/splits.json
```

Los splits se hacen estratificados por clase:

```text
train -> 70 %
val   -> 15 %
test  -> 15 %
```

Esto intenta conservar la proporcion de positivos y negativos en cada particion.

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
