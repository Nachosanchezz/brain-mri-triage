"""
download_nki_rockland.py
------------------------
Descarga NKI Rockland desde aws_links.csv filtrando solo sujetos con T1w y T2w
en la misma sesion BIDS.

Salida esperada:
  data/raw/nki_rockland/sub-*/ses-*/anat/*_T1w.nii.gz
  data/raw/nki_rockland/sub-*/ses-*/anat/*_T2w.nii.gz
  data/raw/nki_rockland/nki_rockland_pairs.csv
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


try:
    from .base_preprocessing import REPO_ROOT
except ImportError:
    from base_preprocessing import REPO_ROOT


DEFAULT_OUT_DIR = REPO_ROOT / "data" / "raw" / "nki_rockland"
DEFAULT_AWS_LINKS = DEFAULT_OUT_DIR / "aws_links.csv"
S3_PREFIX = "s3://fcp-indi/data/Projects/RocklandSample/RawDataBIDSLatest/"
SESSION_PRIORITY = ["BAS1", "BAS2", "BAS3", "FLU1", "FLU2", "FLU3", "TRT"]


@dataclass(frozen=True)
class NkiPair:
    subject: str
    session: str
    gender: str
    age: str
    t1_path: str
    t2_path: str


def local_relative_path(s3_path: str) -> Path:
    if not s3_path.startswith(S3_PREFIX):
        raise ValueError(f"Ruta fuera del prefijo esperado: {s3_path}")
    return Path(s3_path.removeprefix(S3_PREFIX))


def sidecar_json_path(s3_path: str) -> str:
    return s3_path.removesuffix(".nii.gz") + ".json"


def read_aws_links(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_t1_t2_pairs(
    rows: list[dict[str, str]],
    sessions: set[str] | None = None,
    one_session_per_subject: bool = True,
) -> list[NkiPair]:
    grouped: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    metadata: dict[tuple[str, str], dict[str, str]] = {}

    for row in rows:
        subject = row.get("subject", "")
        session = row.get("session", "")
        filepath = row.get("filepath", "")
        if not subject.startswith("A") or not session:
            continue
        if sessions and session not in sessions:
            continue
        if "/anat/" not in filepath or not filepath.endswith(".nii.gz"):
            continue

        key = (subject, session)
        metadata[key] = {
            "gender": row.get("gender", ""),
            "age": row.get("age", ""),
        }
        if filepath.endswith("_T1w.nii.gz"):
            grouped[key]["t1"] = filepath
        elif filepath.endswith("_T2w.nii.gz"):
            grouped[key]["t2"] = filepath

    pairs = [
        NkiPair(
            subject=subject,
            session=session,
            gender=metadata[(subject, session)].get("gender", ""),
            age=metadata[(subject, session)].get("age", ""),
            t1_path=paths["t1"],
            t2_path=paths["t2"],
        )
        for (subject, session), paths in grouped.items()
        if "t1" in paths and "t2" in paths
    ]

    if not one_session_per_subject:
        return sorted(pairs, key=lambda pair: (pair.subject, session_rank(pair.session)))

    best_by_subject: dict[str, NkiPair] = {}
    for pair in sorted(pairs, key=lambda item: (item.subject, session_rank(item.session))):
        best_by_subject.setdefault(pair.subject, pair)
    return list(best_by_subject.values())


def session_rank(session: str) -> int:
    try:
        return SESSION_PRIORITY.index(session)
    except ValueError:
        return len(SESSION_PRIORITY)


def write_pairs_manifest(pairs: list[NkiPair], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["subject", "session", "gender", "age", "t1_path", "t2_path"],
        )
        writer.writeheader()
        for pair in pairs:
            writer.writerow(pair.__dict__)


def download_s3_paths(paths: list[str], out_dir: Path, dry_run: bool) -> tuple[int, int]:
    client = None
    if not dry_run:
        import boto3
        from botocore import UNSIGNED
        from botocore.client import Config

        client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    downloaded = 0
    skipped = 0

    for index, s3_path in enumerate(paths, start=1):
        rel_path = local_relative_path(s3_path)
        out_path = out_dir / rel_path
        if out_path.exists():
            skipped += 1
            print(f"[{index}/{len(paths)}] Ya existe: {out_path}")
            continue

        if dry_run:
            print(f"[{index}/{len(paths)}] Descargar: {out_path}")
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        bucket = "fcp-indi"
        key = s3_path.removeprefix("s3://fcp-indi/")
        print(f"[{index}/{len(paths)}] Descargando: {out_path}")
        with open(out_path, "wb") as f:
            assert client is not None
            client.download_fileobj(bucket, key, f)
        downloaded += 1

    return downloaded, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Descarga NKI Rockland T1w/T2w completos.")
    parser.add_argument("--aws-links", type=Path, default=DEFAULT_AWS_LINKS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sessions", nargs="*", default=None, help="Ejemplo: BAS1 BAS2 FLU1")
    parser.add_argument("--all-sessions", action="store_true", help="No limita a una sesion por sujeto.")
    parser.add_argument("--limit", type=int, default=None, help="Limite de sujetos/sesiones para pruebas.")
    parser.add_argument("--include-json", action="store_true", help="Descarga sidecars JSON de T1/T2.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.aws_links.exists():
        print(f"ERROR: no existe aws_links.csv: {args.aws_links}")
        sys.exit(1)

    if not args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    local_csv = args.out_dir / "aws_links.csv"
    if not args.dry_run and args.aws_links.resolve() != local_csv.resolve():
        shutil.copy2(args.aws_links, local_csv)

    rows = read_aws_links(args.aws_links)
    sessions = set(args.sessions) if args.sessions else None
    pairs = find_t1_t2_pairs(
        rows,
        sessions=sessions,
        one_session_per_subject=not args.all_sessions,
    )
    if args.limit is not None:
        pairs = pairs[: args.limit]

    manifest_path = args.out_dir / "nki_rockland_pairs.csv"
    if not args.dry_run:
        write_pairs_manifest(pairs, manifest_path)

    paths: list[str] = []
    for pair in pairs:
        paths.extend([pair.t1_path, pair.t2_path])
        if args.include_json:
            paths.extend([sidecar_json_path(pair.t1_path), sidecar_json_path(pair.t2_path)])

    print("=" * 60)
    print("NKI ROCKLAND - DESCARGA FILTRADA")
    print("=" * 60)
    print(f"aws_links     : {args.aws_links}")
    print(f"out_dir       : {args.out_dir}")
    print(f"pares T1/T2   : {len(pairs)}")
    print(f"archivos      : {len(paths)}")
    print(f"dry_run       : {args.dry_run}")
    if pairs[:5]:
        print("Primeros pares:")
        for pair in pairs[:5]:
            print(f"  {pair.subject} {pair.session} edad={pair.age} sexo={pair.gender}")

    downloaded, skipped = download_s3_paths(paths, args.out_dir, dry_run=args.dry_run)
    print("=" * 60)
    print(f"Descargados   : {downloaded}")
    print(f"Ya existentes : {skipped}")
    print(f"Manifest pares: {manifest_path}")


if __name__ == "__main__":
    main()
