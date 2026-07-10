"""Download the FloorplanQA layouts dataset from Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

DATASET_ID = "OldDelorean/FloorplanQA-Layouts"
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "datasets" / "FloorplanQA-Layouts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Download {DATASET_ID} from Hugging Face Hub."
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
        help="Dataset revision to download (default: main)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded_path = snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=args.revision,
        local_dir=output_dir,
        token=os.environ.get("HF_TOKEN"),
    )

    print(f"Downloaded {DATASET_ID} to {downloaded_path}")


if __name__ == "__main__":
    main()
