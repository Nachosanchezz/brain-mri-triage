"""
evaluate_3d.py
--------------
Evaluacion de la CNN 3D tumor/no tumor sobre un split guardado.

Guarda:
  - outputs/evaluation/cnn3d_<split>_results.json
  - outputs/evaluation/cnn3d_<split>_predictions.json
  - outputs/evaluation/cnn3d_<split>_predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset_3d import BrainMRI3DDataset, load_splits
from src.models.cnn3d import build_cnn3d
from src.training.train_3d import configure_gpu_performance, parse_shape


DEFAULT_CONFIG = REPO_ROOT / "configs" / "train_3d.yaml"
DEFAULT_CHECKPOINT = REPO_ROOT / "outputs" / "checkpoints" / "cnn3d_best.pt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "evaluation"


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def torch_load(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def scalar_to_python(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def strip_compile_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(key.startswith("_orig_mod.") for key in state_dict):
        return state_dict
    return {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}


def build_model_from_checkpoint(checkpoint: Any, fallback_config: dict, device: torch.device) -> torch.nn.Module:
    checkpoint_config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    config = checkpoint_config or fallback_config
    model_cfg = config.get("model", {})

    model = build_cnn3d(
        in_channels=int(model_cfg.get("in_channels", 2)),
        n_classes=int(model_cfg.get("n_classes", 1)),
        base_channels=int(model_cfg.get("base_channels", 12)),
        dropout=float(model_cfg.get("dropout", 0.25)),
    ).to(device)

    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    state_dict = strip_compile_prefix(state_dict)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def positive_probability(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2 and logits.shape[1] == 2:
        return torch.softmax(logits, dim=1)[:, 1]
    if logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits.squeeze(1)
    return torch.sigmoid(logits)


def binary_auc(y_true: list[int], y_score: list[float]) -> float:
    labels = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(y_score, dtype=np.float64)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end

    sum_pos_ranks = float(ranks[labels == 1].sum())
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(y_true: list[int], y_score: list[float]) -> float:
    labels = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(y_score, dtype=np.float64)
    n_pos = int((labels == 1).sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def metrics_at_threshold(y_true: list[int], y_score: list[float], threshold: float) -> dict[str, float | int]:
    y_pred = [int(score >= threshold) for score in y_score]
    tp = sum(int(pred == 1 and true == 1) for pred, true in zip(y_pred, y_true))
    tn = sum(int(pred == 0 and true == 0) for pred, true in zip(y_pred, y_true))
    fp = sum(int(pred == 1 and true == 0) for pred, true in zip(y_pred, y_true))
    fn = sum(int(pred == 0 and true == 1) for pred, true in zip(y_pred, y_true))

    total = tp + tn + fp + fn
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) else float("nan")

    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": (tp + tn) / total if total else 0.0,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "npv": npv,
        "f1": f1,
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
    }


def threshold_for_target_sensitivity(
    y_true: list[int],
    y_score: list[float],
    target_sensitivity: float,
) -> dict[str, float | int] | None:
    candidates = sorted(set(float(score) for score in y_score), reverse=True)
    candidates.append(0.0)
    valid = [
        metrics_at_threshold(y_true, y_score, threshold)
        for threshold in candidates
        if metrics_at_threshold(y_true, y_score, threshold)["sensitivity"] >= target_sensitivity
    ]
    if not valid:
        return None
    return max(valid, key=lambda row: (float(row["specificity"]), float(row["threshold"])))


def read_npz_metadata(path: Path) -> dict[str, str]:
    with np.load(path) as sample:
        dataset = scalar_to_python(sample["dataset"]) if "dataset" in sample.files else "unknown"
        subject_id = scalar_to_python(sample["subject_id"]) if "subject_id" in sample.files else path.stem
    return {"dataset": str(dataset), "subject_id": str(subject_id)}


def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    channels_last_3d: bool,
) -> tuple[list[int], list[float]]:
    y_true: list[int] = []
    y_score: list[float] = []

    with torch.inference_mode():
        for volumes, labels in loader:
            volumes = volumes.to(device, non_blocking=True)
            if channels_last_3d:
                volumes = volumes.contiguous(memory_format=torch.channels_last_3d)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(volumes)
                probs = positive_probability(logits)

            y_score.extend(float(score) for score in probs.detach().cpu().tolist())
            y_true.extend(int(label) for label in labels.cpu().tolist())

    return y_true, y_score


def write_predictions(predictions: list[dict], json_path: Path, csv_path: Path) -> None:
    json_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")

    fieldnames = ["file", "dataset", "subject_id", "label", "score", "prediction", "correct"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalua la CNN 3D tumor/no tumor.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--target-sensitivity", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"No existe el checkpoint: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    perf_cfg = configure_gpu_performance(config, device)
    channels_last_3d = device.type == "cuda" and bool(perf_cfg.get("channels_last_3d", True))
    use_amp = bool(config.get("training", {}).get("amp", True)) and device.type == "cuda"

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    splits = load_splits()
    file_paths = splits[args.split]
    data_cfg = config.get("data", {})
    crop_shape = parse_shape(data_cfg.get("crop_shape", [128, 160, 128]))
    dataset = BrainMRI3DDataset(file_paths, crop_shape=crop_shape, random_crop=False, augment=False)
    batch_size = args.batch_size or int(data_cfg.get("batch_size", 1))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )

    checkpoint = torch_load(args.checkpoint, device)
    model = build_model_from_checkpoint(checkpoint, config, device)
    if channels_last_3d:
        model = model.to(memory_format=torch.channels_last_3d)

    y_true, y_score = run_inference(model, loader, device, use_amp, channels_last_3d)

    predictions: list[dict] = []
    for path_str, label, score in zip(file_paths, y_true, y_score):
        path = Path(path_str)
        metadata = read_npz_metadata(path)
        pred = int(score >= args.threshold)
        predictions.append(
            {
                "file": str(path.relative_to(REPO_ROOT)) if path.is_absolute() and path.is_relative_to(REPO_ROOT) else str(path),
                "dataset": metadata["dataset"],
                "subject_id": metadata["subject_id"],
                "label": label,
                "score": score,
                "prediction": pred,
                "correct": int(pred == label),
            }
        )

    results = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "n_samples": len(y_true),
        "n_positives": int(sum(y_true)),
        "n_negatives": int(len(y_true) - sum(y_true)),
        "auc": binary_auc(y_true, y_score),
        "pr_auc": average_precision(y_true, y_score),
        "metrics_at_threshold": metrics_at_threshold(y_true, y_score, args.threshold),
        "metrics_at_target_sensitivity": threshold_for_target_sensitivity(
            y_true,
            y_score,
            args.target_sensitivity,
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"cnn3d_{args.split}"
    results_path = args.output_dir / f"{prefix}_results.json"
    predictions_json_path = args.output_dir / f"{prefix}_predictions.json"
    predictions_csv_path = args.output_dir / f"{prefix}_predictions.csv"

    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_predictions(predictions, predictions_json_path, predictions_csv_path)

    m = results["metrics_at_threshold"]
    print(
        f"{args.split} | n={results['n_samples']} "
        f"auc={results['auc']:.4f} pr_auc={results['pr_auc']:.4f} "
        f"acc={m['accuracy']:.4f} sen={m['sensitivity']:.4f} "
        f"spe={m['specificity']:.4f} f1={m['f1']:.4f}"
    )
    print(f"Resultados guardados en: {results_path}")
    print(f"Predicciones guardadas en: {predictions_json_path}")
    print(f"Predicciones CSV guardadas en: {predictions_csv_path}")


if __name__ == "__main__":
    main()
