#!/usr/bin/env python3
"""Download and extract an audio archive, then stage 1,000 files.

This script is intentionally Python-only (no shell syntax) to avoid
SyntaxError issues when run with `python`.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

DEFAULT_DATASET_URLS = (
    "https://github.com/Jakobovski/free-spoken-digit-dataset/"
    "archive/refs/tags/v1.0.9.zip",
)
SUPPORTED_AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download/extract an audio archive and stage up to 1,000 "
            "audio files in a flat directory for the Milvus loaders."
        )
    )
    parser.add_argument(
        "--dataset-url",
        default="",
        help=(
            "Direct URL to an audio archive (zip/tar/tar.gz). If empty, the "
            "script tries the Free Spoken Digit Dataset (FSDD) archive."
        ),
    )
    parser.add_argument(
        "--source-dir",
        default="",
        help=(
            "Optional local folder containing audio files. If set, the script "
            "skips downloading/extracting and stages audio from this folder."
        ),
    )
    parser.add_argument(
        "--download-dir",
        default="/home/ec2-user/dataset/downloads",
        help="Where to store the downloaded archive.",
    )
    parser.add_argument(
        "--extract-dir",
        default="/home/ec2-user/dataset/fsdd",
        help="Where to extract the archive contents.",
    )
    parser.add_argument(
        "--staging-dir",
        default="/home/ec2-user/dataset/fsdd_flat",
        help="Flat folder containing up to 1,000 audio files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="How many audio files to stage for embedding.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
        help="Network timeout (seconds) for downloading the archive.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of download retries on transient network errors.",
    )
    return parser.parse_args()


def download_archive(
    dataset_url: str, download_dir: Path, timeout_seconds: int, retries: int
) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    candidate_urls = [dataset_url] if dataset_url else list(DEFAULT_DATASET_URLS)
    last_error: Exception | None = None
    for url in candidate_urls:
        filename = url.rsplit("/", 1)[-1]
        archive_path = download_dir / filename
        if archive_path.exists():
            print(f"Archive already exists at {archive_path}")
            return archive_path
        print(f"Downloading {url} -> {archive_path}")
        for attempt in range(1, retries + 1):
            try:
                with urlopen(url, timeout=timeout_seconds) as response:
                    archive_path.write_bytes(response.read())
                return archive_path
            except (TimeoutError, URLError) as exc:
                last_error = exc
                if attempt < retries:
                    print(
                        "Download failed "
                        f"(attempt {attempt}/{retries}), retrying..."
                    )
                else:
                    print(f"Failed to download from {url}.")
    message = (
        "Failed to download the FSDD archive from known mirrors. "
        "If these hosts are blocked from your EC2, pass --dataset-url with a "
        "reachable mirror or use --source-dir to stage local audio."
    )
    raise RuntimeError(message) from last_error


def extract_archive(archive_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)
        return
    if archive_path.suffix in {".tar", ".gz", ".tgz", ".bz2", ".xz"}:
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(extract_dir)
        return
    raise ValueError(f"Unsupported archive type: {archive_path}")


def gather_audio_files(root_dir: Path) -> list[Path]:
    files = [
        path
        for path in root_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTS
    ]
    return sorted(files)


def stage_audio_files(files: list[Path], staging_dir: Path, limit: int) -> list[Path]:
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged_files = []
    for idx, src in enumerate(files[:limit]):
        dst = staging_dir / f"{idx:05d}{src.suffix.lower()}"
        if not dst.exists():
            shutil.copy2(src, dst)
        staged_files.append(dst)
    return staged_files


def main() -> int:
    args = parse_args()
    download_dir = Path(args.download_dir).expanduser()
    extract_dir = Path(args.extract_dir).expanduser()
    staging_dir = Path(args.staging_dir).expanduser()

    if args.source_dir:
        source_dir = Path(args.source_dir).expanduser()
        audio_files = gather_audio_files(source_dir)
    else:
        dataset_url = args.dataset_url
        archive_path = download_archive(
            dataset_url, download_dir, args.timeout_seconds, args.retries
        )
        extract_archive(archive_path, extract_dir)

        audio_files = gather_audio_files(extract_dir)
    if not audio_files:
        if args.source_dir:
            print(f"No audio files found under {source_dir}")
            return 1
        print(f"No audio files found under {extract_dir}")
        print(
            "Provide --dataset-url to an audio archive or --source-dir "
            "pointing at local audio."
        )
        return 1

    staged = stage_audio_files(audio_files, staging_dir, args.limit)
    if len(staged) < args.limit:
        print(
            f"Warning: only staged {len(staged)} files (requested {args.limit})."
        )

    print(f"Staged {len(staged)} files in {staging_dir}")
    print("Use this path in your Milvus ingestion scripts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
