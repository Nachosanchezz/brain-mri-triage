"""
audit_splits.py
---------------
Audita data/splits.json: composicion por (split, dataset, label),
solapamiento por subject_id y cobertura de cada dataset.

Uso:
    python -m src.data.audit_splits
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_FILE = REPO_ROOT / "data" / "splits.json"


def _scalar(value):
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def read_meta(path: Path) -> tuple[str, str, int]:
    with np.load(path) as sample:
        ds = str(_scalar(sample["dataset"])) if "dataset" in sample.files else "unknown"
        sid = str(_scalar(sample["subject_id"])) if "subject_id" in sample.files else path.stem
        lbl = int(sample["label"]) if "label" in sample.files else (
            1 if path.parent.name == "positives" else 0
        )
    return ds, sid, lbl


def main() -> None:
    if not SPLITS_FILE.exists():
        print(f"ERROR: no existe {SPLITS_FILE}", file=sys.stderr)
        sys.exit(1)

    splits = json.loads(SPLITS_FILE.read_text(encoding="utf-8"))

    counts: dict[tuple[str, str, int], dict] = defaultdict(
        lambda: {"n": 0, "subjects": set()}
    )
    subjects_by_split: dict[str, set] = defaultdict(set)
    issues: list[str] = []

    for split in ("train", "val", "test"):
        for f in splits[split]:
            p = Path(f)
            if not p.exists():
                issues.append(f"MISSING {f}")
                continue
            ds, sid, lbl = read_meta(p)
            counts[(split, ds, lbl)]["n"] += 1
            counts[(split, ds, lbl)]["subjects"].add(sid)
            subjects_by_split[split].add(sid)

    header = f"{'split':<6} {'dataset':<14} {'label':<5} {'n_samples':<10} {'n_subjects'}"
    print(header)
    print("-" * len(header))
    for (split, ds, lbl), info in sorted(counts.items()):
        print(f"{split:<6} {ds:<14} {lbl:<5} {info['n']:<10} {len(info['subjects'])}")

    print("\n--- subject overlap ---")
    train_set = subjects_by_split["train"]
    val_set = subjects_by_split["val"]
    test_set = subjects_by_split["test"]
    print(f"train ∩ val : {len(train_set & val_set)}")
    print(f"train ∩ test: {len(train_set & test_set)}")
    print(f"val   ∩ test: {len(val_set & test_set)}")

    print("\n--- totals per split ---")
    for split in ("train", "val", "test"):
        total = sum(info["n"] for (s, _, _), info in counts.items() if s == split)
        pos = sum(info["n"] for (s, _, l), info in counts.items() if s == split and l == 1)
        neg = sum(info["n"] for (s, _, l), info in counts.items() if s == split and l == 0)
        print(f"  {split:5s}: total={total} pos={pos} neg={neg} subjects={len(subjects_by_split[split])}")

    if issues:
        print("\nISSUES:")
        for line in issues:
            print(" ", line)
    else:
        print("\nSin issues de archivos faltantes.")


if __name__ == "__main__":
    main()
