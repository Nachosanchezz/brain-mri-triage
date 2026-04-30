"""
base_preprocessing.py
---------------------
Utilidades comunes para convertir volumenes NIfTI T1/T2 a muestras .npz.

Cada dataset debe aportar solamente una lista de DatasetSample. Esta base se
encarga de reorientar, remuestrear, normalizar, guardar y resumir resultados.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_TARGET_SHAPE = (192, 224, 192)
DEFAULT_TARGET_SPACING = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class PreprocessingConfig:
    target_shape: tuple[int, int, int] = DEFAULT_TARGET_SHAPE
    target_spacing: tuple[float, float, float] = DEFAULT_TARGET_SPACING


@dataclass(frozen=True)
class DatasetSample:
    dataset: str
    subject_id: str
    t1_path: Path
    t2_path: Path
    label: int
    output_subdir: str | None = None

    @property
    def class_subdir(self) -> str:
        if self.output_subdir:
            return self.output_subdir
        return "positives" if self.label == 1 else "negatives"

    @property
    def output_name(self) -> str:
        return f"{self.subject_id}.npz"


@dataclass
class DatasetResult:
    dataset: str
    label: int
    found: int = 0
    written: int = 0
    existing: int = 0
    skipped: int = 0
    errors: list[dict[str, str]] = field(default_factory=list)

    @property
    def available(self) -> int:
        return self.written + self.existing

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "found": self.found,
            "written": self.written,
            "existing": self.existing,
            "available": self.available,
            "skipped": self.skipped,
            "errors": self.errors,
        }


def reorient_to_ras(img):
    from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform

    current_ornt = io_orientation(img.affine)
    target_ornt = axcodes2ornt(("R", "A", "S"))
    transform = ornt_transform(current_ornt, target_ornt)
    return img.as_reoriented(transform)


def resample_volume(
    data: np.ndarray,
    current_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
) -> np.ndarray:
    from scipy.ndimage import zoom

    zoom_factors = [current / target for current, target in zip(current_spacing, target_spacing)]
    return zoom(data, zoom_factors, order=1)


def crop_or_pad(data: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    result = np.zeros(target_shape, dtype=data.dtype)

    starts_data: list[int] = []
    ends_data: list[int] = []
    starts_result: list[int] = []
    ends_result: list[int] = []

    for axis in range(3):
        if data.shape[axis] >= target_shape[axis]:
            offset = (data.shape[axis] - target_shape[axis]) // 2
            starts_data.append(offset)
            ends_data.append(offset + target_shape[axis])
            starts_result.append(0)
            ends_result.append(target_shape[axis])
        else:
            offset = (target_shape[axis] - data.shape[axis]) // 2
            starts_data.append(0)
            ends_data.append(data.shape[axis])
            starts_result.append(offset)
            ends_result.append(offset + data.shape[axis])

    result[
        starts_result[0]:ends_result[0],
        starts_result[1]:ends_result[1],
        starts_result[2]:ends_result[2],
    ] = data[
        starts_data[0]:ends_data[0],
        starts_data[1]:ends_data[1],
        starts_data[2]:ends_data[2],
    ]
    return result


def normalize_intensity(data: np.ndarray, source_path: Path | str = "") -> np.ndarray:
    mask = data > 0
    if mask.sum() == 0:
        print(f"  WARNING: volumen sin voxels no-cero ({source_path}), devuelto sin normalizar")
        return data.astype(np.float32, copy=False)

    mean_val = data[mask].mean()
    std_val = data[mask].std()
    if std_val < 1e-8:
        print(f"  WARNING: std ~0 en ({source_path}), devuelto sin normalizar")
        return data.astype(np.float32, copy=False)

    normalized = np.zeros_like(data, dtype=np.float32)
    normalized[mask] = (data[mask] - mean_val) / std_val
    return normalized


def preprocess_single_volume(nifti_path: Path, config: PreprocessingConfig) -> np.ndarray:
    import nibabel as nib

    img = nib.load(str(nifti_path))
    img = reorient_to_ras(img)

    data = img.get_fdata().astype(np.float32)
    spacing = img.header.get_zooms()[:3]
    data = resample_volume(data, spacing, config.target_spacing)
    data = crop_or_pad(data, config.target_shape)
    return normalize_intensity(data, source_path=nifti_path)


def output_path_for_sample(sample: DatasetSample, output_root: Path) -> Path:
    return output_root / sample.class_subdir / sample.output_name


def save_sample_npz(
    sample: DatasetSample,
    output_root: Path,
    config: PreprocessingConfig,
    overwrite: bool = False,
) -> tuple[Path, bool]:
    output_path = output_path_for_sample(sample, output_root)
    if output_path.exists() and not overwrite:
        return output_path, False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    t1_data = preprocess_single_volume(sample.t1_path, config)
    t2_data = preprocess_single_volume(sample.t2_path, config)

    np.savez_compressed(
        str(output_path),
        t1=t1_data,
        t2=t2_data,
        label=np.int64(sample.label),
        dataset=np.array(sample.dataset),
        subject_id=np.array(sample.subject_id),
        source_t1=np.array(str(sample.t1_path)),
        source_t2=np.array(str(sample.t2_path)),
    )
    return output_path, True


def process_dataset(
    samples: Iterable[DatasetSample],
    output_root: Path = DEFAULT_OUTPUT_DIR,
    config: PreprocessingConfig | None = None,
    overwrite: bool = False,
    desc: str | None = None,
) -> DatasetResult:
    config = config or PreprocessingConfig()
    samples = list(samples)
    dataset = samples[0].dataset if samples else "unknown"
    label = samples[0].label if samples else -1
    result = DatasetResult(dataset=dataset, label=label, found=len(samples))

    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        def tqdm(iterable, desc=None):
            return iterable

    for sample in tqdm(samples, desc=desc or dataset):
        try:
            _, wrote_file = save_sample_npz(sample, output_root, config, overwrite=overwrite)
            if wrote_file:
                result.written += 1
            else:
                result.existing += 1
        except Exception as exc:
            result.skipped += 1
            result.errors.append({"subject_id": sample.subject_id, "error": str(exc)})
            print(f"\n  Error procesando {sample.dataset}/{sample.subject_id}: {exc}")

    return result


def _scalar_to_python(value):
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def infer_dataset_from_path(path: Path, files: list[str]) -> str:
    name = path.stem.lower()
    parent = path.parent.name.lower()

    if "dataset" in files:
        try:
            with np.load(path) as sample:
                return str(_scalar_to_python(sample["dataset"]))
        except Exception:
            pass

    if parent == "upenn" or name.startswith("upenn"):
        return "upenn"
    if name.startswith("brats"):
        return "brats"
    if name.startswith("ixi"):
        return "ixi"
    if parent in {"positives", "negatives"}:
        return "unknown"
    return parent


def infer_label_from_path(path: Path, files: list[str]) -> int | None:
    if "label" in files:
        try:
            with np.load(path) as sample:
                return int(_scalar_to_python(sample["label"]))
        except Exception:
            pass

    parent = path.parent.name.lower()
    if parent == "positives":
        return 1
    if parent == "negatives":
        return 0
    return None


def scan_processed(output_root: Path = DEFAULT_OUTPUT_DIR) -> dict:
    output_root = Path(output_root)
    by_dataset: dict[str, dict] = {}
    by_output_dir: dict[str, int] = {}
    label_counts = {"0": 0, "1": 0, "unknown": 0}

    for path in sorted(output_root.rglob("*.npz")):
        rel_parent = str(path.parent.relative_to(output_root))
        by_output_dir[rel_parent] = by_output_dir.get(rel_parent, 0) + 1

        try:
            with np.load(path) as sample:
                files = list(sample.files)
        except Exception:
            files = []

        dataset = infer_dataset_from_path(path, files)
        label = infer_label_from_path(path, files)

        ds_info = by_dataset.setdefault(
            dataset,
            {
                "total": 0,
                "labels": {"0": 0, "1": 0, "unknown": 0},
                "output_dirs": {},
            },
        )
        ds_info["total"] += 1
        ds_info["output_dirs"][rel_parent] = ds_info["output_dirs"].get(rel_parent, 0) + 1

        label_key = str(label) if label in {0, 1} else "unknown"
        ds_info["labels"][label_key] += 1
        label_counts[label_key] += 1

    total = sum(label_counts.values())
    return {
        "total": total,
        "positives": label_counts["1"],
        "negatives": label_counts["0"],
        "unknown_label": label_counts["unknown"],
        "datasets": by_dataset,
        "output_dirs": by_output_dir,
    }


def write_preprocessing_summary(
    output_root: Path = DEFAULT_OUTPUT_DIR,
    config: PreprocessingConfig | None = None,
    run_results: Iterable[DatasetResult] | None = None,
) -> Path:
    config = config or PreprocessingConfig()
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    scan = scan_processed(output_root)
    summary = {
        "target_spacing": list(config.target_spacing),
        "target_shape": list(config.target_shape),
        "total": scan["total"],
        "positives": scan["positives"],
        "negatives": scan["negatives"],
        "unknown_label": scan["unknown_label"],
        "datasets": scan["datasets"],
        "output_dirs": scan["output_dirs"],
    }

    if run_results is not None:
        summary["last_run"] = {result.dataset: result.as_dict() for result in run_results}

    summary_path = output_root / "preprocessing_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary_path
