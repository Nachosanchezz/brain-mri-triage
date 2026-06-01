"""
embeddings_tsne.py  (E1)
------------------------
Extrae el embedding de 96-dim (salida de `model.features` tras
AdaptiveAvgPool3d, antes del clasificador) del checkpoint CONFUNDIDO para
cada volumen de data/processed, y los proyecta a 2D con PCA y t-SNE,
coloreados por dataset y por etiqueta.

Hipotesis: el espacio latente agrupa por DATASET (procedencia), no por
etiqueta clinica -> evidencia visual del confound de dominio.

Salida:
  docs/audit/figures/embeddings_pca.png    (2 paneles: por dataset / por label)
  docs/audit/figures/embeddings_tsne.png   (2 paneles: por dataset / por label)
  docs/audit/embeddings.npz                (vectores + meta, por si se re-grafica)
  docs/audit/embeddings_silhouette.json    (silhouette por dataset y por label)
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.cnn3d import build_cnn3d

PROCESSED = REPO_ROOT / "data" / "processed"
OUT_FIG = REPO_ROOT / "docs" / "audit" / "figures"
OUT_DIR = REPO_ROOT / "docs" / "audit"

DS_COLORS = {"brats": "#c62828", "upenn": "#ad1457", "ixi": "#1565c0", "nki_rockland": "#0277bd"}
LBL_COLORS = {0: "#1565c0", 1: "#c62828"}


def _scalar(v):
    if isinstance(v, np.ndarray):
        v = v.item()
    if isinstance(v, bytes):
        return v.decode("utf-8")
    return v


def center_crop_or_pad(volume, target_shape):
    result = np.zeros(target_shape, dtype=volume.dtype)
    src, dst = [], []
    for axis, t in enumerate(target_shape):
        s = volume.shape[axis]
        if s >= t:
            start = (s - t) // 2
            src.append(slice(start, start + t)); dst.append(slice(0, t))
        else:
            start = (t - s) // 2
            src.append(slice(0, s)); dst.append(slice(start, start + s))
    result[tuple(dst)] = volume[tuple(src)]
    return result


def list_files():
    files = sorted(list((PROCESSED / "positives").glob("*.npz"))
                   + list((PROCESSED / "negatives").glob("*.npz")))
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=REPO_ROOT / "outputs" / "checkpoints" / "20260527_152619" / "best.pt")
    ap.add_argument("--per-dataset", type=int, default=0, help="0 = todos")
    ap.add_argument("--crop", type=int, nargs=3, default=[128, 160, 128])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    crop = tuple(args.crop)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        import gc; gc.collect(); torch.cuda.empty_cache()
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Device: {device}")

    model = build_cnn3d(in_channels=2, n_classes=1, base_channels=12, dropout=0.25).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in sd.items()})
    model.eval()

    files = list_files()
    # submuestreo opcional por dataset
    if args.per_dataset > 0:
        rng = np.random.default_rng(args.seed)
        by_ds = {}
        for f in files:
            with np.load(f) as s:
                ds = str(_scalar(s["dataset"])) if "dataset" in s.files else "?"
            by_ds.setdefault(ds, []).append(f)
        files = []
        for ds, fs in by_ds.items():
            fs = sorted(fs); rng.shuffle(fs)
            files.extend(fs[:args.per_dataset])

    print(f"Extrayendo embeddings de {len(files)} volumenes...")
    embs, labels, datasets, sids = [], [], [], []
    with torch.inference_mode():
        for i, f in enumerate(files):
            with np.load(f) as s:
                t1 = center_crop_or_pad(s["t1"].astype(np.float32), crop)
                t2 = center_crop_or_pad(s["t2"].astype(np.float32), crop)
                lbl = int(s["label"]) if "label" in s.files else (1 if f.parent.name == "positives" else 0)
                ds = str(_scalar(s["dataset"])) if "dataset" in s.files else "?"
                sid = str(_scalar(s["subject_id"])) if "subject_id" in s.files else f.stem
            vol = torch.from_numpy(np.stack([t1, t2])[None]).to(device)
            feat = model.features(vol).flatten(1).squeeze(0).cpu().numpy()  # (96,)
            embs.append(feat); labels.append(lbl); datasets.append(ds); sids.append(sid)
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(files)}")

    X = np.array(embs, dtype=np.float32)
    y = np.array(labels)
    ds_arr = np.array(datasets)
    print(f"Embeddings shape: {X.shape}")

    np.savez_compressed(OUT_DIR / "embeddings.npz", X=X, labels=y, datasets=ds_arr, subject_ids=np.array(sids))

    # Silhouette: como de bien se agrupan por dataset vs por label
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)
    ds_codes = np.array([sorted(set(datasets)).index(d) for d in datasets])
    sil = {
        "silhouette_by_dataset": float(silhouette_score(Xs, ds_codes)),
        "silhouette_by_label": float(silhouette_score(Xs, y)),
        "interpretation": "valor alto = clusters compactos. Si by_dataset >> by_label, el espacio latente codifica procedencia.",
    }
    (OUT_DIR / "embeddings_silhouette.json").write_text(json.dumps(sil, indent=2), encoding="utf-8")
    print(f"Silhouette por dataset = {sil['silhouette_by_dataset']:.3f} | por label = {sil['silhouette_by_label']:.3f}")

    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=args.seed).fit(Xs)
    Z_pca = pca.transform(Xs)
    var = pca.explained_variance_ratio_

    # t-SNE
    from sklearn.manifold import TSNE
    perp = min(30, max(5, len(X) // 50))
    Z_tsne = TSNE(n_components=2, perplexity=perp, random_state=args.seed, init="pca").fit_transform(Xs)

    def two_panel(Z, title, fname, var=None):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        # por dataset
        for ds in sorted(set(datasets)):
            m = ds_arr == ds
            axes[0].scatter(Z[m, 0], Z[m, 1], s=10, alpha=0.6, c=DS_COLORS.get(ds, "#666"), label=ds)
        axes[0].set_title(f"{title} — coloreado por DATASET")
        axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
        # por label
        for lbl in (0, 1):
            m = y == lbl
            axes[1].scatter(Z[m, 0], Z[m, 1], s=10, alpha=0.6, c=LBL_COLORS[lbl],
                            label=("tumor" if lbl == 1 else "sano"))
        axes[1].set_title(f"{title} — coloreado por ETIQUETA")
        axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
        if var is not None:
            axes[0].set_xlabel(f"PC1 ({var[0]*100:.0f}%)"); axes[0].set_ylabel(f"PC2 ({var[1]*100:.0f}%)")
            axes[1].set_xlabel(f"PC1 ({var[0]*100:.0f}%)"); axes[1].set_ylabel(f"PC2 ({var[1]*100:.0f}%)")
        fig.suptitle(f"Embeddings de la CNN confundida (96-dim) — {title}", fontsize=12)
        fig.tight_layout()
        fig.savefig(OUT_FIG / fname, dpi=150)
        plt.close(fig)
        print(f"  guardado: {OUT_FIG / fname}")

    OUT_FIG.mkdir(parents=True, exist_ok=True)
    two_panel(Z_pca, "PCA", "embeddings_pca.png", var=var)
    two_panel(Z_tsne, "t-SNE", "embeddings_tsne.png")
    print("E1 done.")


if __name__ == "__main__":
    main()
