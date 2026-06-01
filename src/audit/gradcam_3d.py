"""
gradcam_3d.py
-------------
Grad-CAM 3D sobre BrainTumorCNN3D para visualizar a que atiende el modelo.

Hipotesis pre-registrada:
  - Modelo confounded -> CAM concentrada en bordes/craneo/firma de fondo.
  - LODO A/B          -> idem o difusa.
  - (Si hubiera honest model con senal real -> CAM centrada en la lesion.)

Uso:
  python -m src.audit.gradcam_3d \\
    --checkpoint outputs/checkpoints/20260527_152619/best.pt \\
    --tag confound \\
    --samples upenn:2 ixi:2 brats:1

Salida: docs/audit/figures/gradcam/<tag>/<dataset>_<subject>.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.cnn3d import build_cnn3d

PROCESSED = REPO_ROOT / "data" / "processed"
PROCESSED_BTC = REPO_ROOT / "data" / "processed_btc"
OUT_BASE = REPO_ROOT / "docs" / "audit" / "figures" / "gradcam"


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


def list_npz_by_dataset(processed_dirs: list[Path]) -> dict[str, list[Path]]:
    by_ds: dict[str, list[Path]] = {}
    for pdir in processed_dirs:
        if not pdir.exists():
            continue
        for sub in ("positives", "negatives"):
            for p in sorted((pdir / sub).glob("*.npz")):
                with np.load(p) as s:
                    ds = str(_scalar(s["dataset"])) if "dataset" in s.files else "?"
                by_ds.setdefault(ds, []).append(p)
    return by_ds


def load_volume(path: Path, in_channels: int, crop_shape=(128, 160, 128)) -> tuple[np.ndarray, np.ndarray, int, str, str]:
    """Devuelve (volume (C,D,H,W) np float32 cropped, t1_raw (D,H,W), label, dataset, subject_id)."""
    with np.load(path) as s:
        t1 = s["t1"].astype(np.float32, copy=True)
        t1_c = center_crop_or_pad(t1, crop_shape)
        if in_channels == 2 and "t2" in s.files:
            t2 = s["t2"].astype(np.float32, copy=True)
            t2_c = center_crop_or_pad(t2, crop_shape)
            vol = np.stack([t1_c, t2_c], axis=0)
        else:
            vol = t1_c[None, ...]
        lbl = int(s["label"]) if "label" in s.files else (1 if path.parent.name == "positives" else 0)
        ds = str(_scalar(s["dataset"])) if "dataset" in s.files else "?"
        sid = str(_scalar(s["subject_id"])) if "subject_id" in s.files else path.stem
    return vol, t1_c, lbl, ds, sid


class GradCAM3D:
    """Grad-CAM 3D enganchado al ultimo bloque convolucional antes del AdaptiveAvgPool3d."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        # En BrainTumorCNN3D, features = Sequential(ConvBlock, MaxPool, ConvBlock, ...,
        # ConvBlock(c3->c4), AdaptiveAvgPool3d). El ultimo ConvBlock es features[6].
        self.target_layer = self.model.features[6]  # ConvBlock3D(c3->c4)
        self.fh = self.target_layer.register_forward_hook(self._fwd_hook)
        self.bh = self.target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self.activations = out.detach()

    def _bwd_hook(self, module, grad_in, grad_out):
        # grad_out es tupla; queremos la grad respecto a la salida del modulo
        self.gradients = grad_out[0].detach()

    def remove(self):
        self.fh.remove(); self.bh.remove()

    def compute(self, volume: torch.Tensor, target_logit_sign: int = 1) -> np.ndarray:
        """volume: (1, C, D, H, W). Devuelve CAM (D, H, W) en [0,1]."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(volume)
        # Para clasificacion binaria con 1 logit:
        if logits.ndim == 1:
            target = logits.squeeze() * target_logit_sign
        else:
            target = logits.view(-1)[0] * target_logit_sign
        target.backward()
        # activations: (1, C', d, h, w); gradients: same shape
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Hooks no capturaron nada.")
        w = self.gradients.mean(dim=(2, 3, 4), keepdim=True)  # (1, C', 1, 1, 1)
        cam = (w * self.activations).sum(dim=1, keepdim=True)  # (1, 1, d, h, w)
        cam = F.relu(cam)
        # Upsample al tamano del input
        target_shape = volume.shape[-3:]
        cam_up = F.interpolate(cam, size=target_shape, mode="trilinear", align_corners=False)
        cam_np = cam_up.squeeze().detach().cpu().numpy().astype(np.float32)
        m, M = cam_np.min(), cam_np.max()
        if M - m > 1e-8:
            cam_np = (cam_np - m) / (M - m)
        else:
            cam_np = np.zeros_like(cam_np)
        return cam_np


def render_overlay(t1: np.ndarray, cam: np.ndarray, title: str, out_path: Path,
                   score: float, label: int) -> None:
    """Tres planos (axial, coronal, sagital) con CAM superpuesto."""
    D, H, W = t1.shape
    cz, cy, cx = D // 2, H // 2, W // 2
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, sl, name in zip(axes[0], [t1[cz], t1[:, cy], t1[:, :, cx]],
                            ["axial", "coronal", "sagital"]):
        ax.imshow(np.rot90(sl), cmap="gray")
        ax.set_title(f"T1 {name}")
        ax.axis("off")
    for ax, sl_t1, sl_cam, name in zip(
        axes[1],
        [t1[cz], t1[:, cy], t1[:, :, cx]],
        [cam[cz], cam[:, cy], cam[:, :, cx]],
        ["axial+CAM", "coronal+CAM", "sagital+CAM"],
    ):
        ax.imshow(np.rot90(sl_t1), cmap="gray")
        ax.imshow(np.rot90(sl_cam), cmap="jet", alpha=0.45)
        ax.set_title(name)
        ax.axis("off")
    label_txt = "tumor" if label == 1 else "sano"
    pred_txt = "tumor" if score >= 0.5 else "sano"
    correct = "✓" if (score >= 0.5) == bool(label) else "✗"
    fig.suptitle(f"{title}\nlabel={label_txt} | score={score:.4f} -> pred={pred_txt} {correct}",
                 fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def parse_samples(spec: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in spec:
        if ":" in s:
            ds, n = s.split(":")
            out[ds.strip()] = int(n)
        else:
            out[s.strip()] = 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--tag", required=True, help="subcarpeta de salida: confound/lodo_A/lodo_B/ghent")
    ap.add_argument("--in-channels", type=int, default=2)
    ap.add_argument("--samples", nargs="+", default=["upenn:2", "ixi:2"],
                    help="dataset:N pares; ej. upenn:2 ixi:2 brats:1")
    ap.add_argument("--include-btc", action="store_true",
                    help="Incluye sujetos de Ghent (data/processed_btc) en el indice")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Modelo
    model = build_cnn3d(in_channels=args.in_channels, n_classes=1, base_channels=12, dropout=0.25).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in sd.items()})
    model.eval()

    cam_calc = GradCAM3D(model)

    # Indice
    pdirs = [PROCESSED] + ([PROCESSED_BTC] if args.include_btc else [])
    by_ds = list_npz_by_dataset(pdirs)
    print(f"datasets indexados: {sorted(by_ds.keys())}")

    targets = parse_samples(args.samples)
    rng = np.random.default_rng(0)
    out_root = OUT_BASE / args.tag

    for ds, n in targets.items():
        if ds not in by_ds:
            print(f"  AVISO: dataset '{ds}' no encontrado, salto")
            continue
        files = sorted(by_ds[ds])
        rng.shuffle(files)
        for i, path in enumerate(files[:n]):
            vol_np, t1_raw, lbl, ds_name, sid = load_volume(path, args.in_channels)
            vol_t = torch.from_numpy(vol_np[None, ...]).to(device)
            # Score sin grad
            with torch.no_grad():
                score = float(torch.sigmoid(model(vol_t)).cpu().item())
            # Grad-CAM activando el gradiente
            vol_t.requires_grad_(False)
            try:
                cam = cam_calc.compute(vol_t)
            except Exception as exc:
                print(f"  Error CAM en {sid}: {exc}")
                continue
            title = f"[{args.tag}] {ds_name} {sid}"
            out_path = out_root / f"{ds_name}_{sid}.png"
            render_overlay(t1_raw, cam, title, out_path, score=score, label=lbl)
            print(f"  guardado: {out_path}  (score={score:.4f})")

    cam_calc.remove()


if __name__ == "__main__":
    main()
