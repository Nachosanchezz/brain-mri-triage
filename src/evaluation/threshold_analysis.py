"""
threshold_analysis.py
---------------------
Calcula threshold optimo en VAL y lo aplica a TEST. Tambien dibuja
histogramas de probabilidades por clase real.

Uso:
    python -m src.evaluation.threshold_analysis \\
        --checkpoint outputs/checkpoints/<run>/best.pt \\
        --target-sensitivity 0.90
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset_3d import BrainMRI3DDataset, load_splits
from src.evaluation.evaluate_3d import (
    binary_auc,
    build_model_from_checkpoint,
    metrics_at_threshold,
    read_npz_metadata,
    run_inference,
    torch_load,
)
from src.training.train_3d import configure_gpu_performance, load_config, parse_shape


def best_threshold_youden(y_true: list[int], y_score: list[float]) -> float:
    """Threshold que maximiza balanced_accuracy (equivale a Youden J)."""
    candidates = sorted(set(y_score)) + [0.0, 1.0]
    return max(
        candidates,
        key=lambda t: metrics_at_threshold(y_true, y_score, t)["balanced_accuracy"],
    )


def best_threshold_min_sensitivity(
    y_true: list[int], y_score: list[float], min_sensitivity: float
) -> float | None:
    """Threshold mas alto que mantiene sensitivity >= min_sensitivity,
    maximizando specificity. Devuelve None si no hay solucion factible.
    """
    candidates = sorted(set(y_score), reverse=True)
    valid = [
        t for t in candidates
        if metrics_at_threshold(y_true, y_score, t)["sensitivity"] >= min_sensitivity
    ]
    if not valid:
        return None
    return max(
        valid,
        key=lambda t: metrics_at_threshold(y_true, y_score, t)["specificity"],
    )


def plot_hist(y_true: list[int], y_score: list[float], out_path: Path, title: str, threshold: float | None = None) -> None:
    scores = np.asarray(y_score)
    labels = np.asarray(y_true)
    plt.figure(figsize=(7, 4))
    plt.hist(scores[labels == 0], bins=40, alpha=0.6, label="no tumor", color="steelblue")
    plt.hist(scores[labels == 1], bins=40, alpha=0.6, label="tumor", color="firebrick")
    plt.axvline(0.5, ls="--", color="black", label="thr=0.5")
    if threshold is not None and threshold != 0.5:
        plt.axvline(threshold, ls=":", color="green", label=f"thr={threshold:.3f}")
    plt.title(title)
    plt.xlabel("score (P[tumor])")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_hist_per_dataset(predictions: list[dict], out_path: Path) -> None:
    datasets = sorted({p["dataset"] for p in predictions})
    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 4), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        sub = [p for p in predictions if p["dataset"] == ds]
        scores = np.asarray([p["score"] for p in sub])
        labels = np.asarray([p["label"] for p in sub])
        if (labels == 0).any():
            ax.hist(scores[labels == 0], bins=20, alpha=0.6, label="no tumor", color="steelblue")
        if (labels == 1).any():
            ax.hist(scores[labels == 1], bins=20, alpha=0.6, label="tumor", color="firebrick")
        ax.axvline(0.5, ls="--", color="black")
        ax.set_title(f"{ds} (n={len(sub)})")
        ax.set_xlabel("score")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def infer_split(checkpoint: Path, config: dict, split: str) -> tuple[list[int], list[float], list[str]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    perf = configure_gpu_performance(config, device)
    channels_last_3d = device.type == "cuda" and bool(perf.get("channels_last_3d", True))
    use_amp = bool(config.get("training", {}).get("amp", True)) and device.type == "cuda"

    file_paths = load_splits()[split]
    data_cfg = config.get("data", {})
    crop_shape = parse_shape(data_cfg.get("crop_shape", [128, 160, 128]))
    dataset = BrainMRI3DDataset(file_paths, crop_shape=crop_shape, random_crop=False, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model_from_checkpoint(torch_load(checkpoint, device), config, device)
    if channels_last_3d:
        model = model.to(memory_format=torch.channels_last_3d)
    y_true, y_score = run_inference(model, loader, device, use_amp, channels_last_3d)
    return y_true, y_score, file_paths


def predictions_with_meta(file_paths: list[str], y_true: list[int], y_score: list[float], threshold: float) -> list[dict]:
    out = []
    for path_str, label, score in zip(file_paths, y_true, y_score):
        meta = read_npz_metadata(Path(path_str))
        out.append({
            "file": path_str,
            "dataset": meta["dataset"],
            "subject_id": meta["subject_id"],
            "label": int(label),
            "score": float(score),
            "prediction": int(score >= threshold),
        })
    return out


def metrics_per_dataset(predictions: list[dict], threshold: float) -> dict[str, dict]:
    by_ds: dict[str, dict[str, list]] = {}
    for p in predictions:
        bucket = by_ds.setdefault(p["dataset"], {"y": [], "s": []})
        bucket["y"].append(p["label"])
        bucket["s"].append(p["score"])
    out = {}
    for ds, bucket in by_ds.items():
        m = metrics_at_threshold(bucket["y"], bucket["s"], threshold)
        only_one_class = len(set(bucket["y"])) < 2
        m["auc"] = float("nan") if only_one_class else binary_auc(bucket["y"], bucket["s"])
        m["n"] = len(bucket["y"])
        m["n_pos"] = int(sum(bucket["y"]))
        m["n_neg"] = int(len(bucket["y"]) - sum(bucket["y"]))
        m["score_mean"] = float(np.mean(bucket["s"]))
        m["score_std"] = float(np.std(bucket["s"]))
        out[ds] = m
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Calcula threshold optimo en val y lo aplica a test.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "train_3d.yaml")
    parser.add_argument("--target-sensitivity", type=float, default=0.90)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "evaluation" / "threshold")
    args = parser.parse_args()

    config = load_config(args.config)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Inferencia en validation...")
    y_val, s_val, files_val = infer_split(args.checkpoint, config, "val")
    print("Inferencia en test...")
    y_te, s_te, files_te = infer_split(args.checkpoint, config, "test")

    # Thresholds calculados SOLO sobre validation
    thr_youden = best_threshold_youden(y_val, s_val)
    thr_min_sen = best_threshold_min_sensitivity(y_val, s_val, args.target_sensitivity)

    # Histogramas
    plot_hist(y_val, s_val, args.out_dir / "hist_val.png", "Validation scores", threshold=thr_youden)
    plot_hist(y_te, s_te, args.out_dir / "hist_test.png", "Test scores", threshold=thr_youden)
    preds_test = predictions_with_meta(files_te, y_te, s_te, thr_youden)
    plot_hist_per_dataset(preds_test, args.out_dir / "hist_test_per_dataset.png")

    # Reporte: metricas a thresholds 0.5, Youden(val), y min_sensitivity(val)
    report: dict = {
        "val": {
            "auc": binary_auc(y_val, s_val),
            "thr_youden_val": thr_youden,
            "thr_min_sensitivity_val": thr_min_sen,
            "metrics@0.5": metrics_at_threshold(y_val, s_val, 0.5),
            "metrics@youden": metrics_at_threshold(y_val, s_val, thr_youden),
        },
        "test": {
            "auc": binary_auc(y_te, s_te),
            "metrics@0.5": metrics_at_threshold(y_te, s_te, 0.5),
            "metrics@youden_val": {
                "threshold": thr_youden,
                **metrics_at_threshold(y_te, s_te, thr_youden),
            },
            "per_dataset@0.5": metrics_per_dataset(predictions_with_meta(files_te, y_te, s_te, 0.5), 0.5),
            "per_dataset@youden_val": metrics_per_dataset(preds_test, thr_youden),
        },
    }

    if thr_min_sen is not None:
        report["test"]["metrics@min_sensitivity_val"] = {
            "threshold": thr_min_sen,
            "target_sensitivity": args.target_sensitivity,
            **metrics_at_threshold(y_te, s_te, thr_min_sen),
        }
        preds_minsen = predictions_with_meta(files_te, y_te, s_te, thr_min_sen)
        report["test"]["per_dataset@min_sensitivity_val"] = metrics_per_dataset(preds_minsen, thr_min_sen)
    else:
        report["test"]["metrics@min_sensitivity_val"] = None

    out_path = args.out_dir / "thresholds.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nResultados en: {out_path}")
    print(f"Histogramas en: {args.out_dir}")
    print("\n--- resumen ---")
    print(f"VAL  AUC={report['val']['auc']:.4f}  thr_youden={thr_youden:.4f}  thr_min_sen={thr_min_sen}")
    m05 = report["test"]["metrics@0.5"]
    my = report["test"]["metrics@youden_val"]
    print(f"TEST@0.5         sen={m05['sensitivity']:.3f} spe={m05['specificity']:.3f} bal={m05['balanced_accuracy']:.3f}")
    print(f"TEST@youden_val  sen={my['sensitivity']:.3f} spe={my['specificity']:.3f} bal={my['balanced_accuracy']:.3f}")
    if thr_min_sen is not None:
        mm = report["test"]["metrics@min_sensitivity_val"]
        print(f"TEST@min_sen_val sen={mm['sensitivity']:.3f} spe={mm['specificity']:.3f} bal={mm['balanced_accuracy']:.3f}")


if __name__ == "__main__":
    main()
