"""
dataset_3d.py
-------------
Dataset PyTorch para clasificacion binaria con volumenes MRI 3D en .npz.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_FILE = DATA_DIR / "splits.json"


def center_crop_or_pad(volume: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Ajusta un volumen 3D a target_shape con crop/pad centrado."""
    result = np.zeros(target_shape, dtype=volume.dtype)

    src_slices = []
    dst_slices = []
    for axis, target_size in enumerate(target_shape):
        size = volume.shape[axis]
        if size >= target_size:
            start = (size - target_size) // 2
            src_slices.append(slice(start, start + target_size))
            dst_slices.append(slice(0, target_size))
        else:
            start = (target_size - size) // 2
            src_slices.append(slice(0, size))
            dst_slices.append(slice(start, start + size))

    result[tuple(dst_slices)] = volume[tuple(src_slices)]
    return result


def random_crop_or_pad(volume: np.ndarray, target_shape: tuple[int, int, int], rng: np.random.Generator) -> np.ndarray:
    """Ajusta un volumen 3D a target_shape con crop aleatorio y pad centrado si hace falta."""
    result = np.zeros(target_shape, dtype=volume.dtype)

    src_slices = []
    dst_slices = []
    for axis, target_size in enumerate(target_shape):
        size = volume.shape[axis]
        if size >= target_size:
            start = int(rng.integers(0, size - target_size + 1))
            src_slices.append(slice(start, start + target_size))
            dst_slices.append(slice(0, target_size))
        else:
            start = (target_size - size) // 2
            src_slices.append(slice(0, size))
            dst_slices.append(slice(start, start + size))

    result[tuple(dst_slices)] = volume[tuple(src_slices)]
    return result


def list_processed_files(processed_dir: Path = PROCESSED_DIR) -> tuple[list[str], list[int]]:
    pos_files = sorted((processed_dir / "positives").glob("*.npz"))
    neg_files = sorted((processed_dir / "negatives").glob("*.npz"))

    files = [str(path) for path in pos_files + neg_files]
    labels = [1] * len(pos_files) + [0] * len(neg_files)
    return files, labels


def _scalar(value):
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _extract_meta(path: str) -> tuple[str, str, int]:
    p = Path(path)
    with np.load(p) as sample:
        ds = str(_scalar(sample["dataset"])) if "dataset" in sample.files else "unknown"
        sid = str(_scalar(sample["subject_id"])) if "subject_id" in sample.files else p.stem
        lbl = int(sample["label"]) if "label" in sample.files else (
            1 if p.parent.name == "positives" else 0
        )
    return ds, sid, lbl


def create_splits(
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    processed_dir: Path = PROCESSED_DIR,
    splits_file: Path = SPLITS_FILE,
) -> dict:
    """Crea split estratificado por (dataset, label), agrupado por subject_id.

    - Estratificacion conjunta por dataset Y label evita que un re-split
      por azar deje un dataset sub/sobrerrepresentado en val/test.
    - Agrupacion por subject_id evita leakage cuando un mismo sujeto
      aparece en multiples ficheros (p.ej. NKI-...-BAS2, BAS3).
    """
    files, _ = list_processed_files(processed_dir)
    if not files:
        raise FileNotFoundError(f"No se encontraron .npz en {processed_dir}")

    # 1) Indexar por subject_id (los ficheros del mismo sujeto deben caer en el mismo split)
    by_subject: dict[str, dict] = {}
    for f in files:
        ds, sid, lbl = _extract_meta(f)
        info = by_subject.setdefault(sid, {"dataset": ds, "label": lbl, "files": []})
        info["files"].append(f)

    # 2) Estratificar por (dataset, label) sobre la lista de sujetos unicos
    rng = np.random.default_rng(seed)
    groups: dict[tuple[str, int], list[str]] = {}
    for sid, info in by_subject.items():
        groups.setdefault((info["dataset"], info["label"]), []).append(sid)

    train_subjects: list[str] = []
    val_subjects: list[str] = []
    test_subjects: list[str] = []
    for key, sids in groups.items():
        sids = sorted(sids)  # estabilidad reproducible
        rng.shuffle(sids)
        n = len(sids)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        train_subjects.extend(sids[:n_train])
        val_subjects.extend(sids[n_train:n_train + n_val])
        test_subjects.extend(sids[n_train + n_val:])

    def expand(sids: list[str]) -> list[str]:
        out: list[str] = []
        for sid in sids:
            out.extend(by_subject[sid]["files"])
        rng.shuffle(out)
        return out

    splits = {
        "seed": seed,
        "train": expand(train_subjects),
        "val": expand(val_subjects),
        "test": expand(test_subjects),
    }

    splits_file.parent.mkdir(parents=True, exist_ok=True)
    with open(splits_file, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    print(f"Splits guardados en {splits_file} (estratificados por dataset+label, agrupados por subject_id)")
    for name, split_files, sids in [
        ("train", splits["train"], train_subjects),
        ("val", splits["val"], val_subjects),
        ("test", splits["test"], test_subjects),
    ]:
        n_pos = sum(1 for f in split_files if Path(f).parent.name == "positives")
        n_neg = len(split_files) - n_pos
        print(f"  {name:5s}: {len(split_files):4d} files ({n_pos} pos, {n_neg} neg) | {len(sids)} subjects")

    return splits


def load_splits(splits_file: Path = SPLITS_FILE) -> dict:
    if not splits_file.exists():
        raise FileNotFoundError(f"No existe {splits_file}")
    with open(splits_file, encoding="utf-8") as f:
        return json.load(f)


class BrainMRI3DDataset(Dataset):
    """Devuelve tensores (2, D, H, W) con canales T1 y T2."""

    def __init__(
        self,
        file_paths: list[str | Path],
        crop_shape: tuple[int, int, int] | None = (128, 160, 128),
        random_crop: bool = False,
        augment: bool = False,
        seed: int = 42,
    ):
        self.file_paths = [Path(path) for path in file_paths]
        self.crop_shape = crop_shape
        self.random_crop = random_crop
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.file_paths)

    def _fit_shape(self, volume: np.ndarray) -> np.ndarray:
        if self.crop_shape is None:
            return volume
        if self.random_crop:
            return random_crop_or_pad(volume, self.crop_shape, self.rng)
        return center_crop_or_pad(volume, self.crop_shape)

    def _augment(self, volume: np.ndarray) -> np.ndarray:
        if not self.augment:
            return volume
        # Flips espaciales compartidos para T1/T2.
        for axis in (1, 2, 3):
            if self.rng.random() < 0.5:
                volume = np.flip(volume, axis=axis).copy()
        # Augmentation de intensidad aplicada SOLO sobre voxels no-cero
        # (mascara de cerebro). Asi no se inyecta señal en el fondo
        # — clave en UPENN, que trae shell de fondo gris.
        # Objetivo: forzar al modelo a usar contraste relativo y no
        # depender del brillo absoluto caracteristico de cada dataset.
        for c in range(volume.shape[0]):
            channel = volume[c]
            mask = channel != 0
            if not mask.any():
                continue
            # Gamma correction sobre la magnitud (preservando el signo,
            # porque el preprocesado deja valores z-score con negativos).
            if self.rng.random() < 0.5:
                gamma = float(self.rng.uniform(0.8, 1.25))
                values = channel[mask]
                sign = np.sign(values)
                magnitude = np.abs(values) + 1e-6
                channel[mask] = (sign * (magnitude ** gamma)).astype(np.float32)
            # Ruido gaussiano de baja amplitud sobre el cerebro.
            if self.rng.random() < 0.5:
                noise = self.rng.normal(0.0, 0.03, size=int(mask.sum())).astype(np.float32)
                channel[mask] = channel[mask] + noise
            volume[c] = channel
        return volume

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.file_paths[idx]
        with np.load(path) as sample:
            t1 = sample["t1"].astype(np.float32, copy=True)
            t2 = sample["t2"].astype(np.float32, copy=True)
            label = int(sample["label"])

        t1 = self._fit_shape(t1)
        t2 = self._fit_shape(t2)
        volume = np.stack([t1, t2], axis=0)
        volume = self._augment(volume)

        return torch.from_numpy(volume.copy()), torch.tensor(label, dtype=torch.float32)
