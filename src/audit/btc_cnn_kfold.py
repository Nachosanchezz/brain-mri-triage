"""
btc_cnn_kfold.py
----------------
CNN 3D 1-canal con k-fold por sujeto sobre BTC (ds001226), INTRA-DOMINIO.

Mismo modelo (`src/models/cnn3d.py`) y mismos hiperparametros que el run
confundido, pero T1-only y k-fold (n=36 -> no hay para train/val/test
convencional). NO toca train_3d.py ni dataset_3d.py.

Diseno:
  - k-fold estratificado por label, a nivel de sujeto (1 fichero = 1 sujeto).
  - Por cada fold: train fijo de N_EPOCHS epocas (sin early stopping para
    NO contaminar la seleccion con el test fold).
  - Evaluacion al final sobre el fold held-out.
  - Agregamos predicciones de los k folds -> AUC global + IC95% bootstrap.

Salida:
  outputs/evaluation/btc_intradomain/cnn_kfold_results.json
  outputs/checkpoints/btc_intradomain/fold_<k>/final.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.cnn3d import build_cnn3d  # arquitectura reutilizada SIN tocar

PROCESSED = REPO_ROOT / "data" / "processed_btc"
CHECKPOINT_DIR = REPO_ROOT / "outputs" / "checkpoints" / "btc_intradomain"
EVAL_DIR = REPO_ROOT / "outputs" / "evaluation" / "btc_intradomain"


def _scalar(v):
    if isinstance(v, np.ndarray):
        v = v.item()
    if isinstance(v, bytes):
        return v.decode("utf-8")
    return v


def center_crop_or_pad(volume: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    result = np.zeros(target_shape, dtype=volume.dtype)
    src_slices, dst_slices = [], []
    for axis, t in enumerate(target_shape):
        s = volume.shape[axis]
        if s >= t:
            start = (s - t) // 2
            src_slices.append(slice(start, start + t))
            dst_slices.append(slice(0, t))
        else:
            start = (t - s) // 2
            src_slices.append(slice(0, s))
            dst_slices.append(slice(start, start + s))
    result[tuple(dst_slices)] = volume[tuple(src_slices)]
    return result


class BTCDataset(Dataset):
    """T1-only. Devuelve (1, D, H, W). Augmentation simple en train."""

    def __init__(self, files: list[Path], crop_shape=(128, 160, 128),
                 augment: bool = False, seed: int = 42):
        self.files = [Path(p) for p in files]
        self.crop_shape = crop_shape
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.files)

    def _augment(self, vol: np.ndarray) -> np.ndarray:
        # flips espaciales en los 3 ejes
        for axis in (1, 2, 3):
            if self.rng.random() < 0.5:
                vol = np.flip(vol, axis=axis).copy()
        # augmentation de intensidad sobre voxels no-cero (mascara de cerebro)
        for c in range(vol.shape[0]):
            channel = vol[c]
            mask = channel != 0
            if not mask.any():
                continue
            if self.rng.random() < 0.5:
                gamma = float(self.rng.uniform(0.8, 1.25))
                vals = channel[mask]
                sign = np.sign(vals)
                mag = np.abs(vals) + 1e-6
                channel[mask] = (sign * (mag ** gamma)).astype(np.float32)
            if self.rng.random() < 0.5:
                noise = self.rng.normal(0.0, 0.03, size=int(mask.sum())).astype(np.float32)
                channel[mask] = channel[mask] + noise
            vol[c] = channel
        return vol

    def __getitem__(self, idx: int):
        with np.load(self.files[idx]) as s:
            t1 = s["t1"].astype(np.float32, copy=True)
            label = int(s["label"])
        t1 = center_crop_or_pad(t1, self.crop_shape)
        vol = t1[None, ...]  # (1, D, H, W)
        if self.augment:
            vol = self._augment(vol)
        return torch.from_numpy(vol.copy()), torch.tensor(label, dtype=torch.float32)


def list_btc_files() -> tuple[list[Path], np.ndarray, list[str]]:
    files = sorted(list((PROCESSED / "positives").glob("*.npz"))
                   + list((PROCESSED / "negatives").glob("*.npz")))
    labels = []
    sids = []
    for p in files:
        with np.load(p) as s:
            lbl = int(s["label"]) if "label" in s.files else (1 if p.parent.name == "positives" else 0)
            sid = str(_scalar(s["subject_id"])) if "subject_id" in s.files else p.stem
        labels.append(lbl)
        sids.append(sid)
    return files, np.array(labels, dtype=np.int64), sids


def stratified_kfold_subjects(labels: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for cls in (0, 1):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        for i, ix in enumerate(idx):
            folds[i % k].append(int(ix))
    return [np.array(sorted(f)) for f in folds]


def binary_auc(y_true, y_score) -> float:
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(y_score, dtype=np.float64)
    if len(set(y.tolist())) < 2:
        return float("nan")
    order = np.argsort(s)
    sorted_s = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    start = 0
    while start < len(s):
        end = start + 1
        while end < len(s) and sorted_s[end] == sorted_s[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    sum_pos_ranks = float(ranks[y == 1].sum())
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def bootstrap_ci(y_true, y_score, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    s = np.asarray(y_score)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(set(y[idx].tolist())) < 2:
            continue
        aucs.append(binary_auc(y[idx], s[idx]))
    if not aucs:
        return float("nan"), float("nan"), float("nan")
    aucs = np.array(aucs)
    return float(aucs.mean()), float(np.quantile(aucs, alpha / 2)), float(np.quantile(aucs, 1 - alpha / 2))


def train_fold(train_files: list[Path], device: torch.device, n_epochs: int,
               lr: float, weight_decay: float, seed: int,
               channels_last_3d: bool) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_cnn3d(in_channels=1, n_classes=1, base_channels=12, dropout=0.25).to(device)
    if channels_last_3d:
        model = model.to(memory_format=torch.channels_last_3d)
    # pos_weight desde el train fold
    labels = []
    for p in train_files:
        with np.load(p) as s:
            labels.append(int(s["label"]))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    pos_weight = float(n_neg) / float(max(n_pos, 1))
    pw_t = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    train_ds = BTCDataset(train_files, augment=True, seed=seed)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0,
                              pin_memory=(device.type == "cuda"))

    for epoch in range(1, n_epochs + 1):
        model.train()
        losses = []
        for vols, lbls in train_loader:
            vols = vols.to(device, non_blocking=True)
            if channels_last_3d:
                vols = vols.contiguous(memory_format=torch.channels_last_3d)
            lbls = lbls.to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            logits = model(vols)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        print(f"    epoch {epoch:02d}/{n_epochs}: train loss = {np.mean(losses):.4f}  lr = {optimizer.param_groups[0]['lr']:.2e}")
    return model


def eval_fold(model: nn.Module, test_files: list[Path], device: torch.device,
              channels_last_3d: bool) -> tuple[list[int], list[float]]:
    model.eval()
    ds = BTCDataset(test_files, augment=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                        pin_memory=(device.type == "cuda"))
    y_true: list[int] = []
    y_score: list[float] = []
    with torch.inference_mode():
        for vols, lbls in loader:
            vols = vols.to(device, non_blocking=True)
            if channels_last_3d:
                vols = vols.contiguous(memory_format=torch.channels_last_3d)
            logits = model(vols)
            probs = torch.sigmoid(logits).detach().cpu().tolist()
            y_score.extend(float(s) for s in probs)
            y_true.extend(int(l) for l in lbls.tolist())
    return y_true, y_score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5, help="numero de folds")
    ap.add_argument("--epochs", type=int, default=20, help="epocas FIJAS por fold (sin early stopping)")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels_last_3d = device.type == "cuda"
    if device.type == "cuda":
        # Liberar VRAM residual (importante si HD-BET dejo el bucket lleno)
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Device: {device}  (CUDA no disponible)")

    files, labels, sids = list_btc_files()
    print(f"BTC files: {len(files)}  positivos={int(labels.sum())}  negativos={int((labels==0).sum())}")
    if int(labels.sum()) < 2 or int((labels == 0).sum()) < 2:
        raise SystemExit("ERROR: pocas muestras por clase para k-fold.")

    k = min(args.k, int(min((labels == 1).sum(), (labels == 0).sum())))
    folds = stratified_kfold_subjects(labels, k, args.seed)
    print(f"k-fold = {k}-fold estratificado por sujeto. Tamaños test por fold: {[len(f) for f in folds]}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    all_y_true: list[int] = []
    all_y_score: list[float] = []
    per_fold = []
    t_global = time.time()

    for fi, test_idx in enumerate(folds):
        t0 = time.time()
        print(f"\n=== Fold {fi+1}/{k} ===")
        train_idx = np.array(sorted(set(range(len(files))) - set(test_idx.tolist())))
        train_files = [files[i] for i in train_idx]
        test_files = [files[i] for i in test_idx]
        print(f"  train n={len(train_files)} (pos={int(labels[train_idx].sum())}, neg={int((labels[train_idx]==0).sum())})")
        print(f"  test  n={len(test_files)} (pos={int(labels[test_idx].sum())}, neg={int((labels[test_idx]==0).sum())})")

        model = train_fold(train_files, device, n_epochs=args.epochs,
                           lr=args.lr, weight_decay=args.wd, seed=args.seed + fi,
                           channels_last_3d=channels_last_3d)
        fold_ckpt_dir = CHECKPOINT_DIR / f"fold_{fi+1}"
        fold_ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(),
                    "fold": fi + 1,
                    "test_indices": test_idx.tolist(),
                    "test_subject_ids": [sids[i] for i in test_idx]},
                   fold_ckpt_dir / "final.pt")

        y_true, y_score = eval_fold(model, test_files, device, channels_last_3d)
        all_y_true.extend(y_true)
        all_y_score.extend(y_score)

        fold_auc = binary_auc(y_true, y_score)
        per_fold.append({
            "fold": fi + 1,
            "n_test": len(test_files),
            "n_pos": int(sum(y_true)),
            "n_neg": int(len(y_true) - sum(y_true)),
            "test_subjects": [sids[i] for i in test_idx],
            "auc": fold_auc,
            "scores": y_score,
            "labels": y_true,
            "elapsed_s": round(time.time() - t0, 1),
        })
        print(f"  fold AUC = {fold_auc:.4f}  ({time.time()-t0:.0f}s)")

    overall_auc = binary_auc(all_y_true, all_y_score)
    mu, lo, hi = bootstrap_ci(all_y_true, all_y_score, n_boot=2000, seed=args.seed)
    elapsed_total = time.time() - t_global

    # sensitivity / specificity a thr=0.5
    y_pred = [1 if s >= 0.5 else 0 for s in all_y_score]
    tp = sum(int(p == 1 and t == 1) for p, t in zip(y_pred, all_y_true))
    tn = sum(int(p == 0 and t == 0) for p, t in zip(y_pred, all_y_true))
    fp = sum(int(p == 1 and t == 0) for p, t in zip(y_pred, all_y_true))
    fn = sum(int(p == 0 and t == 1) for p, t in zip(y_pred, all_y_true))
    sen = tp / (tp + fn) if (tp + fn) else float("nan")
    spe = tn / (tn + fp) if (tn + fp) else float("nan")
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)

    print("\n" + "=" * 60)
    print(f"AUC global agregado = {overall_auc:.4f}  IC95% [{lo:.4f}, {hi:.4f}]  (bootstrap mean {mu:.4f})")
    print(f"@thr=0.5: sen = {sen:.4f}, spe = {spe:.4f}, acc = {acc:.4f}  ({tp}TP/{tn}TN/{fp}FP/{fn}FN)")
    print(f"Tiempo total: {elapsed_total/60:.1f} min")
    print("=" * 60)

    out = {
        "n_samples": len(all_y_true),
        "n_positives": int(sum(all_y_true)),
        "n_negatives": int(len(all_y_true) - sum(all_y_true)),
        "k": k,
        "epochs_per_fold": args.epochs,
        "lr": args.lr,
        "weight_decay": args.wd,
        "seed": args.seed,
        "overall_auc": overall_auc,
        "overall_auc_ci95": [lo, hi],
        "metrics_at_thr_0.5": {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "sensitivity": sen, "specificity": spe, "accuracy": acc,
        },
        "per_fold": per_fold,
        "elapsed_min": round(elapsed_total / 60, 2),
    }
    out_path = EVAL_DIR / "cnn_kfold_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nJSON -> {out_path}")


if __name__ == "__main__":
    main()
