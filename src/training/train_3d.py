"""
train_3d.py
-----------
Entrenamiento de CNN 3D para clasificacion tumor/no tumor.

Guarda en outputs/checkpoints/<timestamp>/:
  - best.pt
  - history.json
  - curves.png
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset_3d import BrainMRI3DDataset, SPLITS_FILE, create_splits, load_splits
from src.models.cnn3d import build_cnn3d


DEFAULT_CONFIG = REPO_ROOT / "configs" / "train_3d.yaml"


def configure_gpu_performance(config: dict, device: torch.device) -> dict:
    perf_cfg = config.get("performance", {})
    if device.type != "cuda":
        return perf_cfg

    torch.backends.cudnn.benchmark = bool(perf_cfg.get("cudnn_benchmark", True))
    torch.backends.cuda.matmul.allow_tf32 = bool(perf_cfg.get("allow_tf32", True))
    torch.backends.cudnn.allow_tf32 = bool(perf_cfg.get("allow_tf32", True))
    torch.set_float32_matmul_precision(str(perf_cfg.get("matmul_precision", "high")))
    return perf_cfg


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_shape(value: list[int] | tuple[int, int, int] | None) -> tuple[int, int, int] | None:
    if value is None:
        return None
    if len(value) != 3:
        raise ValueError("crop_shape debe tener 3 enteros")
    return tuple(int(v) for v in value)


def make_loaders(config: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_cfg = config["data"]
    split_cfg = data_cfg.get("splits", {})

    if not SPLITS_FILE.exists() or data_cfg.get("recreate_splits", False):
        create_splits(
            seed=int(data_cfg.get("seed", 42)),
            train_ratio=float(split_cfg.get("train_ratio", 0.70)),
            val_ratio=float(split_cfg.get("val_ratio", 0.15)),
        )

    splits = load_splits()
    crop_shape = parse_shape(data_cfg.get("crop_shape", [128, 160, 128]))

    train_ds = BrainMRI3DDataset(
        splits["train"],
        crop_shape=crop_shape,
        random_crop=bool(data_cfg.get("random_crop_train", True)),
        augment=bool(data_cfg.get("augment_train", True)),
        seed=int(data_cfg.get("seed", 42)),
    )
    val_ds = BrainMRI3DDataset(splits["val"], crop_shape=crop_shape, random_crop=False, augment=False)
    test_ds = BrainMRI3DDataset(splits["test"], crop_shape=crop_shape, random_crop=False, augment=False)

    loader_kwargs = {
        "batch_size": int(data_cfg.get("batch_size", 1)),
        "num_workers": int(data_cfg.get("num_workers", 0)),
        "pin_memory": torch.cuda.is_available(),
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def infer_pos_weight(loader: DataLoader) -> float:
    labels = []
    for path in loader.dataset.file_paths:
        with np.load(path) as sample:
            labels.append(int(sample["label"]))

    n_neg = labels.count(0)
    n_pos = labels.count(1)
    pos_weight = n_neg / n_pos if n_pos else 1.0
    print(f"Class counts train: [no tumor={n_neg}, tumor={n_pos}] -> pos_weight={pos_weight:.4f}")
    return pos_weight


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


def binary_metrics(y_true: list[int], y_score: list[float], threshold: float = 0.5) -> dict[str, float]:
    y_pred = [int(score >= threshold) for score in y_score]
    tp = sum(int(pred == 1 and true == 1) for pred, true in zip(y_pred, y_true))
    tn = sum(int(pred == 0 and true == 0) for pred, true in zip(y_pred, y_true))
    fp = sum(int(pred == 1 and true == 0) for pred, true in zip(y_pred, y_true))
    fn = sum(int(pred == 0 and true == 1) for pred, true in zip(y_pred, y_true))

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")

    return {
        "accuracy": accuracy,
        "auc": binary_auc(y_true, y_score),
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    channels_last_3d: bool = False,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    losses: list[float] = []
    y_true: list[int] = []
    y_score: list[float] = []

    for volumes, labels in loader:
        volumes = volumes.to(device, non_blocking=True)
        if channels_last_3d:
            volumes = volumes.contiguous(memory_format=torch.channels_last_3d)
        labels = labels.to(device, non_blocking=True).float()

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(volumes)
                loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        losses.append(float(loss.detach().cpu()))
        scores = torch.sigmoid(logits).detach().cpu().tolist()
        y_score.extend(float(score) for score in scores)
        y_true.extend(int(label) for label in labels.detach().cpu().tolist())

    metrics = binary_metrics(y_true, y_score)

    return {
        "loss": sum(losses) / max(len(losses), 1),
        **metrics,
    }


def save_curves(history: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0, 0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0, 0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0, 0].set_title("Loss"); axes[0, 0].set_xlabel("Epoch"); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(epochs, [row["train_auc"] for row in history], label="train")
    axes[0, 1].plot(epochs, [row["val_auc"] for row in history], label="val")
    axes[0, 1].set_title("AUC"); axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(epochs, [row["val_sensitivity"] for row in history], label="val sen", color="firebrick")
    axes[1, 0].plot(epochs, [row["val_specificity"] for row in history], label="val spe", color="steelblue")
    axes[1, 0].set_title("Val sensitivity / specificity"); axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylim(0, 1); axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    bal = [row.get("val_balanced_accuracy", float("nan")) for row in history]
    axes[1, 1].plot(epochs, bal, label="val balanced_acc", color="darkgreen")
    axes[1, 1].set_title("Val balanced accuracy"); axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylim(0, 1); axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena una CNN 3D para brain MRI triage.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    perf_cfg = configure_gpu_performance(config, device)
    channels_last_3d = device.type == "cuda" and bool(perf_cfg.get("channels_last_3d", True))
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, test_loader = make_loaders(config)
    print(f"Batches train/val/test: {len(train_loader)}/{len(val_loader)}/{len(test_loader)}")

    model_cfg = config["model"]
    model = build_cnn3d(
        in_channels=int(model_cfg.get("in_channels", 2)),
        n_classes=int(model_cfg.get("n_classes", 1)),
        base_channels=int(model_cfg.get("base_channels", 12)),
        dropout=float(model_cfg.get("dropout", 0.25)),
    ).to(device)
    if channels_last_3d:
        model = model.to(memory_format=torch.channels_last_3d)
    if device.type == "cuda" and bool(perf_cfg.get("compile", False)):
        model = torch.compile(model)

    training_cfg = config["training"]
    pos_weight = training_cfg.get("pos_weight", training_cfg.get("class_weights"))
    if pos_weight == "auto":
        pos_weight = infer_pos_weight(train_loader)
    pos_weight_tensor = torch.tensor(float(pos_weight), dtype=torch.float32, device=device) if pos_weight else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("lr", 1e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )

    scheduler_name = str(training_cfg.get("scheduler", "none")).lower()
    n_epochs = int(training_cfg.get("n_epochs", 30))
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    elif scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )

    use_amp = bool(training_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_checkpoint_dir = REPO_ROOT / config["paths"].get("checkpoint_dir", "outputs/checkpoints")
    checkpoint_dir = base_checkpoint_dir / run_ts
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_path = checkpoint_dir / "best.pt"
    history_path = checkpoint_dir / "history.json"
    curves_path = checkpoint_dir / "curves.png"
    print(f"Run directory: {checkpoint_dir}")

    history: list[dict] = []
    best_val_score = float("-inf")
    patience = int(training_cfg.get("patience", 8))
    epochs_without_improvement = 0
    min_sensitivity_for_save = float(training_cfg.get("min_sensitivity_for_save", 0.80))

    for epoch in range(1, n_epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler,
            use_amp,
            channels_last_3d,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            scaler=None,
            use_amp=use_amp,
            channels_last_3d=channels_last_3d,
        )

        sen = val_metrics["sensitivity"]
        spe = val_metrics["specificity"]
        # balanced_accuracy: NaN si alguna de las dos es NaN (no contar)
        bal = (sen + spe) / 2.0 if (sen == sen and spe == spe) else float("nan")

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_auc": train_metrics["auc"],
            "train_sensitivity": train_metrics["sensitivity"],
            "train_specificity": train_metrics["specificity"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_auc": val_metrics["auc"],
            "val_sensitivity": sen,
            "val_specificity": spe,
            "val_balanced_accuracy": bal,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={row['train_loss']:.4f} acc={row['train_accuracy']:.4f} auc={row['train_auc']:.4f} | "
            f"val loss={row['val_loss']:.4f} auc={row['val_auc']:.4f} "
            f"sen={sen:.4f} spe={spe:.4f} bal={bal:.4f} | lr={row['lr']:.2e}"
        )

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        save_curves(history, curves_path)

        # Step del scheduler (despues de la metrica de val para plateau)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if bal == bal:
                    scheduler.step(bal)
            else:
                scheduler.step()

        # Criterio de checkpoint: balanced_accuracy con sensitivity minima.
        # Razon: con val_specificity=0 el AUC podia subir y guardarse un modelo
        # inservible para triaje. Aqui exigimos que ademas spe>0 implicitamente
        # (bal > 0) y que se mantenga una sensitivity razonable.
        is_valid = (bal == bal) and (sen == sen) and (sen >= min_sensitivity_for_save)
        if is_valid and bal > best_val_score:
            best_val_score = bal
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "best_val_balanced_accuracy": bal,
                    "best_val_sensitivity": sen,
                    "best_val_specificity": spe,
                    "best_val_auc": row["val_auc"],
                    "best_val_loss": row["val_loss"],
                },
                best_path,
            )
            print(
                f"  Nuevo mejor checkpoint (bal={bal:.4f}, sen={sen:.4f}, spe={spe:.4f}): {best_path}"
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping tras {patience} epocas sin mejora en balanced_accuracy "
                    f"con sen >= {min_sensitivity_for_save:.2f}."
                )
                break

    if not best_path.exists():
        print(
            "AVISO: no se guardo ningun checkpoint que cumpliera la condicion "
            f"(sen>={min_sensitivity_for_save}). Evaluando con el modelo final."
        )
    else:
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_epoch(
        model,
        test_loader,
        criterion,
        device,
        optimizer=None,
        scaler=None,
        use_amp=use_amp,
        channels_last_3d=channels_last_3d,
    )
    print(
        f"Test | loss={test_metrics['loss']:.4f} acc={test_metrics['accuracy']:.4f} "
        f"auc={test_metrics['auc']:.4f} sen={test_metrics['sensitivity']:.4f} "
        f"spe={test_metrics['specificity']:.4f}"
    )
    print(f"Curvas guardadas en: {curves_path}")
    print(f"Historia guardada en: {history_path}")


if __name__ == "__main__":
    main()
