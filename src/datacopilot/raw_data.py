"""Load the Module 3 raw datasets from data/raw/ into DataFrames (no cleaning here)."""
import csv
from pathlib import Path

import pandas as pd

OLIST_PREFIX, OLIST_SUFFIX = "olist_", "_dataset"


def olist_table_name(path: Path) -> str:
    """'olist_order_items_dataset.csv' -> 'order_items'."""
    name = path.stem
    if name.startswith(OLIST_PREFIX):
        name = name[len(OLIST_PREFIX):]
    if name.endswith(OLIST_SUFFIX):
        name = name[: -len(OLIST_SUFFIX)]
    return name


def load_olist(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """All Olist CSVs keyed by short table name (orders, order_items, ...)."""
    return {olist_table_name(p): pd.read_csv(p) for p in sorted((raw_dir / "olist").glob("*.csv"))}


# Keys that appear once per match in a Cricsheet _info.csv, mapped to output columns.
IPL_SINGLE_KEYS = {
    "season": "season", "event": "event", "match_number": "match_number", "stage": "stage",
    "venue": "venue", "city": "city", "toss_winner": "toss_winner", "toss_decision": "toss_decision",
    "player_of_match": "player_of_match", "winner": "winner", "winner_runs": "winner_runs",
    "winner_wickets": "winner_wickets", "outcome": "outcome", "method": "method", "eliminator": "eliminator",
}


def parse_ipl_info(path: Path) -> dict:
    """Parse one Cricsheet `<match_id>_info.csv` (key-value lines) into a flat match record."""
    record: dict = {"match_id": int(path.stem.split("_")[0]), "team1": None, "team2": None, "date": None}
    teams: list[str] = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 3 or row[0] != "info":
                continue
            key, value = row[1], row[2]
            if key == "team":
                teams.append(value)
            elif key == "date" and record["date"] is None:
                record["date"] = value  # first day if a match spans several
            elif key in IPL_SINGLE_KEYS and IPL_SINGLE_KEYS[key] not in record:
                record[IPL_SINGLE_KEYS[key]] = value
    if teams:
        record["team1"] = teams[0]
        record["team2"] = teams[1] if len(teams) > 1 else None
    return record


def load_ipl(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """IPL deliveries (all_matches.csv, `ball` kept as text) and matches (from *_info.csv)."""
    folder = raw_dir / "ipl"
    deliveries = pd.read_csv(folder / "all_matches.csv", dtype={"ball": str}, low_memory=False)
    matches = pd.DataFrame([parse_ipl_info(p) for p in sorted(folder.glob("*_info.csv"))])
    return {"deliveries": deliveries, "matches": matches}


F1_PREFIX = "formula_one_"


def load_f1(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Jolpica-F1 CSV dump tables keyed by name without the 'formula_one_' prefix."""
    return {
        p.stem.removeprefix(F1_PREFIX): pd.read_csv(p, low_memory=False)
        for p in sorted((raw_dir / "f1").glob(f"{F1_PREFIX}*.csv"))
    }


def load_crop(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Crop production (single table)."""
    return {"crop_production": pd.read_csv(raw_dir / "crop_production" / "crop_production.csv")}


LOADERS = {"olist": load_olist, "ipl": load_ipl, "f1": load_f1, "crop_production": load_crop}
