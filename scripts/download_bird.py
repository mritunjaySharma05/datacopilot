"""Download the official BIRD dev set and print a summary.

Source: https://bird-bench.github.io/ (dev set link). Data goes to
data/benchmarks/bird_dev/, which is gitignored.

Usage:
    python scripts/download_bird.py            # download (if needed), unzip, summarize
    python scripts/download_bird.py --force    # re-download even if the zip exists
"""
import argparse
import json
import sys
import zipfile
from pathlib import Path

import requests

BIRD_DEV_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = ROOT / "data" / "benchmarks" / "bird_dev"
CHUNK_BYTES = 1 << 20  # 1 MiB
TIMEOUT_S = 60


def download(url: str, target: Path, force: bool = False) -> Path:
    """Stream `url` to `target` (via a .part file); skip if a valid zip is already there."""
    if target.exists() and not force and zipfile.is_zipfile(target):
        print(f"Already downloaded: {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    with requests.get(url, stream=True, timeout=TIMEOUT_S) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        with open(part, "wb") as f:
            for chunk in response.iter_content(CHUNK_BYTES):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\rDownloading: {done / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="")
    print()
    if total and done != total:
        raise IOError(f"Incomplete download: got {done} of {total} bytes")
    part.replace(target)
    return target


def extract_all(zip_path: Path, dest: Path) -> None:
    """Unzip `zip_path` into `dest`, then unzip any nested .zip files in place."""
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    while True:
        nested = [
            p for p in dest.rglob("*.zip")
            if "__MACOSX" not in p.parts and p != zip_path
        ]
        if not nested:
            return
        for inner in nested:
            with zipfile.ZipFile(inner) as zf:
                zf.extractall(inner.parent)
            inner.unlink()  # avoid re-extracting; the outer zip is kept


def find_questions_file(root: Path) -> Path:
    """Return the dev questions file (dev.json) under `root`."""
    matches = [p for p in root.rglob("dev.json") if "__MACOSX" not in p.parts]
    if not matches:
        raise FileNotFoundError(f"No dev.json found under {root}")
    return min(matches, key=lambda p: len(p.parts))


def count_questions(questions_file: Path) -> int:
    """Return the number of questions in a BIRD questions JSON file."""
    with open(questions_file, encoding="utf-8") as f:
        return len(json.load(f))


def find_databases(root: Path) -> list[str]:
    """Return sorted database names: folders that contain a <name>.sqlite file."""
    names = {
        p.stem for p in root.rglob("*.sqlite")
        if "__MACOSX" not in p.parts and p.parent.name == p.stem
    }
    return sorted(names)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    parser = argparse.ArgumentParser(description="Download and summarize the BIRD dev set.")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="Output folder")
    parser.add_argument("--force", action="store_true", help="Re-download the zip")
    args = parser.parse_args(argv)

    zip_path = args.dest / "dev.zip"
    try:
        download(BIRD_DEV_URL, zip_path, force=args.force)
    except (requests.RequestException, IOError) as err:
        print(f"Download failed: {err}", file=sys.stderr)
        return 1

    print(f"Extracting into {args.dest} ...")
    extract_all(zip_path, args.dest)

    questions_file = find_questions_file(args.dest)
    databases = find_databases(args.dest)
    print(f"\nQuestions file: {questions_file.relative_to(args.dest)}")
    print(f"Questions:      {count_questions(questions_file)}")
    print(f"Databases:      {len(databases)}")
    for name in databases:
        print(f"  - {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
