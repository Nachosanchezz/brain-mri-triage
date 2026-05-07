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


def create_splits(
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    processed_dir: Path = PROCESSED_DIR,
    splits_file: Path = SPLITS_FILE,
) -> dict:
    """Crea split estratificado train/val/test y lo guarda en JSON."""
    files, labels = list_processed_files(processed_dir)
    if not files:
        raise FileNotFoundError(f"No se encontraron .npz en {processed_dir}")

    rng = np.random.default_rng(seed)
    class_to_files: dict[int, list[str]] = {0: [], 1: []}
    for path, label in zip(files, labels):
        class_to_files[int(label)].append(path)

    train_files: list[str] = []
    val_files: list[str] = []
    test_files: list[str] = []
    train_labels: list[int] = []
    val_labels: list[int] = []
    test_labels: list[int] = []

    for label, class_files in class_to_files.items():
        shuffled = list(class_files)
        rng.shuffle(shuffled)
        n_total = len(shuffled)
        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))

        split_train = shuffled[:n_train]
        split_val = shuffled[n_train:n_train + n_val]
        split_test = shuffled[n_train + n_val:]

        train_files.extend(split_train)
        val_files.extend(split_val)
        test_files.extend(split_test)
        train_labels.extend([label] * len(split_train))
        val_labels.extend([label] * len(split_val))
        test_labels.extend([label] * len(split_test))

    def shuffle_together(split_files: list[str], split_labels: list[int]) -> tuple[list[str], list[int]]:
        indices = np.arange(len(split_files))
        rng.shuffle(indices)
        return [split_files[i] for i in indices], [split_labels[i] for i in indices]

    train_files, train_labels = shuffle_together(train_files, train_labels)
    val_files, val_labels = shuffle_together(val_files, val_labels)
    test_files, test_labels = shuffle_together(test_files, test_labels)

    splits = {
        "seed": seed,
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }

    splits_file.parent.mkdir(parents=True, exist_ok=True)
    with open(splits_file, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    def count_split(split_labels: list[int]) -> tuple[int, int]:
        n_pos = int(sum(split_labels))
        n_neg = int(len(split_labels) - n_pos)
        return n_pos, n_neg

    print(f"Splits guardados en {splits_file}")
    for name, split_files, split_labels in [
        ("train", train_files, train_labels),
        ("val", val_files, val_labels),
        ("test", test_files, test_labels),
    ]:
        n_pos, n_neg = count_split(split_labels)
        print(f"  {name:5s}: {len(split_files):4d} ({n_pos} pos, {n_neg} neg)")

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

        return torch.from_numpy(volume.copy()), torch.tensor(label, dtype=torch.long)
