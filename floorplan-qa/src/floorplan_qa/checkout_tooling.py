"""Materialize the FloorplanQA tooling without creating a nested Git repo."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY = "OldDeLorean/FloorplanQA"
REPOSITORY_URL = f"https://github.com/{REPOSITORY}"
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "fpqa-tooling"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Materialize {REPOSITORY} from a source archive, without nested "
            "Git metadata."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Branch, tag, or commit to materialize (default: main)",
    )
    return parser.parse_args()


def _archive_url(revision: str) -> str:
    encoded_revision = urllib.parse.quote(revision, safe="")
    return f"{REPOSITORY_URL}/archive/{encoded_revision}.tar.gz"


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.getmembers():
        member_path = (destination / member.name).resolve()
        if destination not in member_path.parents and member_path != destination:
            raise RuntimeError(f"Unsafe path in source archive: {member.name}")
        if member.issym() or member.islnk():
            raise RuntimeError(f"Links are not allowed in source archive: {member.name}")
    archive.extractall(destination)


def _replace_destination(source: Path, destination: Path) -> None:
    backup = destination.with_name(f".{destination.name}.backup")
    if backup.exists():
        shutil.rmtree(backup)

    if destination.exists():
        os.replace(destination, backup)

    try:
        os.replace(source, destination)
    except Exception:
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)


def main() -> None:
    args = parse_args()
    destination = args.output_dir.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    archive_url = _archive_url(args.revision)

    with tempfile.TemporaryDirectory(
        prefix=".fpqa-tooling-", dir=destination.parent
    ) as temporary_dir:
        temporary_path = Path(temporary_dir)
        archive_path = temporary_path / "source.tar.gz"
        extract_path = temporary_path / "extracted"
        extract_path.mkdir()

        print(f"Downloading {REPOSITORY}@{args.revision}...")
        urllib.request.urlretrieve(archive_url, archive_path)

        with tarfile.open(archive_path, mode="r:gz") as archive:
            _safe_extract(archive, extract_path)

        roots = [path for path in extract_path.iterdir() if path.is_dir()]
        if len(roots) != 1:
            raise RuntimeError("Expected exactly one root directory in source archive")

        source_root = roots[0]
        nested_git_dirs = list(source_root.rglob(".git"))
        if nested_git_dirs:
            raise RuntimeError("Source archive unexpectedly contains Git metadata")

        metadata = {
            "repository": REPOSITORY_URL,
            "revision": args.revision,
            "archive_url": archive_url,
            "materialized_at": datetime.now(timezone.utc).isoformat(),
        }
        (source_root / ".floorplanqa-source.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

        _replace_destination(source_root, destination)

    print(f"Materialized {REPOSITORY}@{args.revision} at {destination}")


if __name__ == "__main__":
    main()
