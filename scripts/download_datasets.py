"""Download the Module 3 datasets into data/raw/<name>/ (gitignored).

Datasets (sources and licences: docs/planning/data_sources.md):
    olist  Brazilian E-commerce by Olist (Kaggle, CC BY-NC-SA 4.0), 9 CSVs
    ipl    IPL ball-by-ball data from Cricsheet (ODC-BY 1.0), CSV "csv2" format
    crop   District-wise, season-wise crop production, data.gov.in (GODL-India)
           -> needs a free data.gov.in API key in .env as DATA_GOV_IN_API_KEY
    f1     F1 World Championship, Jolpica-F1 delayed CSV dump (CC BY-NC-SA 4.0)

Usage:
    python scripts/download_datasets.py              # all datasets
    python scripts/download_datasets.py olist f1     # only some
    python scripts/download_datasets.py --force      # re-download archives
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Callable

import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_bird import download, extract_all  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
TIMEOUT_S = 60

OLIST_URL = "https://www.kaggle.com/api/v1/datasets/download/olistbr/brazilian-ecommerce"
OLIST_FILES = [
    "olist_customers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
]
IPL_URL = "https://cricsheet.org/downloads/ipl_csv2.zip"
F1_URL = "https://api.jolpi.ca/data/dumps/download/delayed/?dump_type=csv"
CROP_RESOURCE_ID = "35be999b-0208-4354-b557-f6ca9a5355de"
CROP_API_URL = f"https://api.data.gov.in/resource/{CROP_RESOURCE_ID}"
CROP_PAGE_SIZE = 5000

OLIST_MANUAL_STEPS = """Olist could not be downloaded automatically. Manual steps:
  1. Sign in at https://www.kaggle.com (free account).
  2. Open https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce and click Download.
  3. Unzip archive.zip into data/raw/olist/ so the 9 olist_*.csv files sit directly in it.
  Or with the Kaggle CLI (pip install kaggle; API token in ~/.kaggle/kaggle.json):
     kaggle datasets download olistbr/brazilian-ecommerce -p data/raw/olist --unzip"""

CROP_MANUAL_STEPS = """Crop production needs a free data.gov.in API key. Steps:
  1. Register / sign in at https://data.gov.in (top right: Login > Sign up).
  2. Open My Account (profile menu) and generate an API key.
  3. Add it to .env:  DATA_GOV_IN_API_KEY=<your key>
  4. Re-run:  python scripts/download_datasets.py crop
  Alternative (no API): open
     https://data.gov.in/resource/district-wise-season-wise-crop-production-statistics-1997
  click Download > CSV, fill in the short usage form, and save the file as
     data/raw/crop_production/crop_production.csv"""


class ManualStepsRequired(Exception):
    """The dataset can't be fetched automatically; the message holds the steps."""


def dir_summary(folder: Path) -> tuple[int, int]:
    """Return (number of files, total bytes) under `folder`."""
    files = [p for p in folder.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def fetch_zip(url: str, dest: Path, name: str, force: bool = False) -> None:
    """Download a zip into `dest/_archive/<name>.zip` and extract it into `dest`."""
    zip_path = dest / "_archive" / f"{name}.zip"
    download(url, zip_path, force=force)
    extract_all(zip_path, dest)


def get_olist(dest: Path, force: bool = False) -> None:
    """Olist: Kaggle's public download endpoint; manual steps if it needs a login."""
    try:
        fetch_zip(OLIST_URL, dest, "olist", force)
    except requests.HTTPError as err:
        if err.response is not None and err.response.status_code in (401, 403):
            raise ManualStepsRequired(OLIST_MANUAL_STEPS) from err
        raise
    missing = [f for f in OLIST_FILES if not (dest / f).exists()]
    if missing:
        raise RuntimeError(f"Olist archive is missing expected files: {missing}")


def get_ipl(dest: Path, force: bool = False) -> None:
    """IPL: Cricsheet csv2 zip (one <match>.csv + <match>_info.csv per match)."""
    fetch_zip(IPL_URL, dest, "ipl_csv2", force)


def get_f1(dest: Path, force: bool = False) -> None:
    """F1: Jolpica-F1 delayed CSV dump (free tier, all seasons 1950 to ~2 weeks ago)."""
    fetch_zip(F1_URL, dest, "jolpica_f1_csv", force)


def fetch_crop_records(api_key: str, page_size: int = CROP_PAGE_SIZE) -> pd.DataFrame:
    """Page through the data.gov.in API and return all crop production records."""
    frames, offset = [], 0
    while True:
        response = requests.get(
            CROP_API_URL,
            params={"api-key": api_key, "format": "json", "limit": page_size, "offset": offset},
            timeout=TIMEOUT_S,
        )
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise RuntimeError(f"data.gov.in API error: {payload['error']}")
        records = payload.get("records", [])
        if not records:
            break
        frames.append(pd.DataFrame(records))
        offset += len(records)
        total = int(payload.get("total") or 0)
        print(f"\r  crop: {offset:,} / {total:,} records", end="")
        if total and offset >= total:
            break
    print()
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def get_crop(dest: Path, force: bool = False) -> None:
    """Crop production: data.gov.in API with the user's own key (from .env)."""
    target = dest / "crop_production.csv"
    if target.exists() and not force:
        print(f"Already downloaded: {target}")
        return
    api_key = os.getenv("DATA_GOV_IN_API_KEY", "").strip()
    if not api_key:
        raise ManualStepsRequired(CROP_MANUAL_STEPS)
    df = fetch_crop_records(api_key)
    if df.empty:
        raise RuntimeError("data.gov.in returned no records")
    dest.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)


DATASETS: dict[str, Callable[[Path, bool], None]] = {
    "olist": get_olist,
    "ipl": get_ipl,
    "crop": get_crop,
    "f1": get_f1,
}
FOLDERS = {"olist": "olist", "ipl": "ipl", "crop": "crop_production", "f1": "f1"}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns 0 if every requested dataset is present, else 1."""
    load_dotenv()
    parser = argparse.ArgumentParser(description="Download Module 3 datasets into data/raw/.")
    parser.add_argument("names", nargs="*", help=f"Datasets to fetch: {', '.join(DATASETS)} (default: all)")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--force", action="store_true", help="Re-download archives")
    args = parser.parse_args(argv)
    unknown = [n for n in args.names if n not in DATASETS]
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(unknown)}; choose from {', '.join(DATASETS)}")

    status: dict[str, str] = {}
    manual: list[str] = []
    for name in args.names or list(DATASETS):
        dest = args.raw_dir / FOLDERS[name]
        print(f"\n== {name} -> {dest}")
        try:
            DATASETS[name](dest, args.force)
        except ManualStepsRequired as steps:
            status[name] = "MANUAL STEPS NEEDED"
            manual.append(str(steps))
            continue
        except (requests.RequestException, OSError, RuntimeError) as err:
            status[name] = f"FAILED: {err}"
            continue
        files, size = dir_summary(dest)
        status[name] = f"ok, {files:,} files, {size / 1e6:,.1f} MB on disk"

    print("\nSummary:")
    for name, text in status.items():
        print(f"  {name:6} {text}")
    for steps in manual:
        print(f"\n{steps}")
    return 0 if all(s.startswith("ok") for s in status.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
