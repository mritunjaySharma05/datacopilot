"""Clean the Module 3 raw datasets and write analysis-ready outputs to data/processed/.

Outputs:
    data/processed/olist.sqlite                          (9 tables)
    data/processed/ipl_deliveries.csv, ipl_matches.csv
    data/processed/f1_*.csv                              (seasons >= F1_MIN_SEASON)
    data/processed/crop_production.csv
Every step is logged with the rows it affected in docs/data_quality/cleaning_log.md.

Usage:
    python scripts/prepare_data.py                 # all datasets
    python scripts/prepare_data.py ipl crop_production
"""
import argparse
import re
import sqlite3
import sys
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datacopilot.dataprep import (  # noqa: E402
    CleaningLog, changed_rows, fold_text, normalize_whitespace, parse_duration_ms, tf_to_bool, to_snake_case,
)
from datacopilot.raw_data import LOADERS  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
LOG_PATH = ROOT / "docs" / "data_quality" / "cleaning_log.md"

F1_MIN_SEASON = 2014  # hybrid-engine era onwards
BRAZIL_LAT, BRAZIL_LNG = (-34.0, 6.0), (-74.0, -34.0)
STRAY_CITY_CHARS = r"[£´³º`]"

# Same franchise, two spellings in the source (2016 vs 2017 seasons).
IPL_TEAM_SPELLING = {"Rising Pune Supergiants": "Rising Pune Supergiant"}
# Official renames: original name stays in batting_team/bowling_team/team1/...,
# the *_franchise columns hold the current name so a franchise can be followed across years.
IPL_FRANCHISE = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
}


# ---------------------------------------------------------------- generic steps

def snake_case_columns(df: pd.DataFrame, log: CleaningLog, dataset: str, table: str) -> pd.DataFrame:
    """Rename columns to snake_case (fixing known typos); log the renames."""
    mapping = {c: to_snake_case(c) for c in df.columns}
    renamed = {a: b for a, b in mapping.items() if a != b}
    if renamed:
        log.record(dataset, table, "rename columns to snake_case", 0, len(df), len(df),
                   ", ".join(f"{a} -> {b}" for a, b in renamed.items()))
    return df.rename(columns=mapping)


def trim_text_columns(df: pd.DataFrame, log: CleaningLog, dataset: str, table: str,
                      columns: list[str] | None = None) -> pd.DataFrame:
    """Trim and collapse whitespace in text columns; log rows changed per column."""
    df = df.copy()
    for col in columns or [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]:
        cleaned = normalize_whitespace(df[col])
        n = changed_rows(df[col], cleaned)
        if n:
            log.record(dataset, table, f"trim whitespace in {col}", n, len(df), len(df))
        df[col] = cleaned
    return df


def drop_exact_duplicates(df: pd.DataFrame, log: CleaningLog, dataset: str, table: str,
                          when: str = "") -> pd.DataFrame:
    """Drop exact duplicate rows; always logged (0 is useful information too)."""
    out = df.drop_duplicates(ignore_index=True)
    log.record(dataset, table, f"drop exact duplicate rows{when}", len(df) - len(out), len(df), len(out))
    return out


def parse_datetimes(df: pd.DataFrame, columns: list[str], log: CleaningLog, dataset: str,
                    table: str) -> pd.DataFrame:
    """Convert text columns to datetimes; log values that could not be parsed."""
    df = df.copy()
    for col in columns:
        parsed = pd.to_datetime(df[col], errors="coerce")
        bad = int((parsed.isna() & df[col].notna()).sum())
        log.record(dataset, table, f"parse {col} as datetime", bad, len(df), len(df),
                   "unparseable values set to NULL" if bad else "all values parsed")
        df[col] = parsed
    return df


# ---------------------------------------------------------------- olist

def normalize_city(series: pd.Series) -> pd.Series:
    """Lowercase, strip accents and stray symbols, collapse spaces ('São  Paulo' -> 'sao paulo')."""
    def one(value: object) -> object:
        if not isinstance(value, str):
            return value
        text = re.sub(STRAY_CITY_CHARS, "", value)
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
        return " ".join(text.lower().split())
    return series.map(one)


def clean_olist(tables: dict[str, pd.DataFrame], log: CleaningLog) -> dict[str, pd.DataFrame]:
    """Clean all Olist tables (names, types, cities, duplicates, impossible coordinates)."""
    ds = "olist"
    out = {}
    for name, df in tables.items():
        df = snake_case_columns(df, log, ds, name)
        # Free-text review comments keep their line breaks; only short text fields are trimmed.
        text_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) and "comment" not in c]
        if text_cols:
            df = trim_text_columns(df, log, ds, name, text_cols)
        dt_cols = [c for c in df.columns if c.endswith(("_timestamp", "_date", "_at"))]
        if dt_cols:
            df = parse_datetimes(df, dt_cols, log, ds, name)
        for col in [c for c in df.columns if c.endswith("_city")]:
            cleaned = normalize_city(df[col])
            log.record(ds, name, f"normalize {col} (lowercase, no accents/stray symbols)",
                       changed_rows(df[col], cleaned), len(df), len(df))
            df[col] = cleaned
        for col in [c for c in df.columns if c.endswith("_state")]:
            df[col] = df[col].str.upper()
        out[name] = df

    geo = drop_exact_duplicates(out["geolocation"], log, ds, "geolocation")
    inside = geo.geolocation_lat.between(*BRAZIL_LAT) & geo.geolocation_lng.between(*BRAZIL_LNG)
    log.record(ds, "geolocation", "drop coordinates outside Brazil", int((~inside).sum()), len(geo),
               int(inside.sum()), f"lat {BRAZIL_LAT}, lng {BRAZIL_LNG}")
    out["geolocation"] = geo[inside].reset_index(drop=True)

    for name in [n for n in out if n != "geolocation"]:
        out[name] = drop_exact_duplicates(out[name], log, ds, name)

    tr = out["product_category_name_translation"].set_index("product_category_name")["product_category_name_english"]
    products = out["products"]
    products["product_category_name_english"] = products.product_category_name.map(tr)
    untranslated = products.product_category_name.notna() & products.product_category_name_english.isna()
    log.record(ds, "products", "add product_category_name_english from translation table",
               int(untranslated.sum()), len(products), len(products),
               "rows whose category has no translation keep NULL: "
               + ", ".join(sorted(products.product_category_name[untranslated].unique())))
    return out


def write_sqlite(tables: dict[str, pd.DataFrame], path: Path) -> None:
    """Write tables to a fresh SQLite file and index every *_id column for joins."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    with sqlite3.connect(path) as conn:
        for name, df in tables.items():
            df.to_sql(name, conn, index=False)
            for col in [c for c in df.columns if c.endswith("_id") or c.endswith("_zip_code_prefix")]:
                conn.execute(f'CREATE INDEX "ix_{name}_{col}" ON "{name}" ("{col}")')
    # Re-open read-only to confirm the file is valid.
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as conn:
        conn.execute("SELECT COUNT(*) FROM orders").fetchone()


# ---------------------------------------------------------------- ipl

def split_ball(ball: pd.Series) -> tuple[pd.Series, pd.Series]:
    """'2.10' -> (over 2, ball_in_over 10). Must be text: as float, 2.10 == 2.1."""
    parts = ball.astype(str).str.split(".", n=1, expand=True)
    return parts[0].astype(int), parts[1].astype(int)


def venue_mapping(*venue_columns: pd.Series) -> dict[str, str]:
    """Map raw venue names to one canonical name.

    Drops the locality suffix after the first comma (', Mumbai', ', Chepauk, Chennai'),
    groups names that are equal ignoring case/punctuation ('M.Chinnaswamy' vs
    'M Chinnaswamy'), and picks the most frequent spelling in each group. Grounds that
    were renamed (e.g. Feroz Shah Kotla) are NOT merged: that needs manual review.
    """
    raw = pd.concat(venue_columns).dropna().astype(str)
    base = raw.str.split(",").str[0].str.strip()
    key = base.map(fold_text)
    canonical = base.groupby(key).agg(lambda s: s.value_counts().index[0])
    return dict(zip(raw, key.map(canonical)))


def clean_ipl_team(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return (team with spelling fixed, current franchise name)."""
    team = series.replace(IPL_TEAM_SPELLING)
    return team, team.replace(IPL_FRANCHISE)


def clean_ipl(tables: dict[str, pd.DataFrame], log: CleaningLog) -> dict[str, pd.DataFrame]:
    """Clean IPL deliveries and matches (season, ball, team and venue names)."""
    ds = "ipl"
    d = snake_case_columns(tables["deliveries"], log, ds, "deliveries")
    d = trim_text_columns(d, log, ds, "deliveries")
    m = snake_case_columns(tables["matches"], log, ds, "matches")
    m = trim_text_columns(m, log, ds, "matches")
    venues = venue_mapping(d.venue, m.venue)

    d["over"], d["ball_in_over"] = split_ball(d["ball"])
    log.record(ds, "deliveries", "split ball (text) into over + ball_in_over, drop ball",
               int((d.ball_in_over >= 10).sum()), len(d), len(d),
               "rows with ball_in_over >= 10; any CSV reader would turn ball '2.10' into 2.1")
    d = d.drop(columns="ball")
    d["start_date"] = pd.to_datetime(d.start_date)
    season = d.start_date.dt.year
    log.record(ds, "deliveries", "season label -> calendar year of the match",
               changed_rows(d.season.astype(str), season.astype(str)), len(d), len(d),
               "2007/08 -> 2008, 2009/10 -> 2010, 2020/21 -> 2020")
    d["season"] = season
    for side in ("batting", "bowling"):
        before = d[f"{side}_team"]
        d[f"{side}_team"], d[f"{side}_franchise"] = clean_ipl_team(before)
        log.record(ds, "deliveries", f"fix {side}_team spelling variants", changed_rows(before, d[f"{side}_team"]),
                   len(d), len(d), "; ".join(f"{a} -> {b}" for a, b in IPL_TEAM_SPELLING.items()))
        log.record(ds, "deliveries", f"add {side}_franchise (current name after renames)",
                   changed_rows(d[f"{side}_team"], d[f"{side}_franchise"]), len(d), len(d),
                   "; ".join(f"{a} -> {b}" for a, b in IPL_FRANCHISE.items()))
    before = d.venue
    d["venue"] = before.map(venues)
    log.record(ds, "deliveries", "canonical venue names", changed_rows(before, d.venue), len(d), len(d),
               f"{before.nunique()} -> {d.venue.nunique()} distinct venues")
    d = drop_exact_duplicates(d, log, ds, "deliveries")

    m["date"] = pd.to_datetime(m.date)
    season = m.date.dt.year
    log.record(ds, "matches", "season label -> calendar year of the match",
               changed_rows(m.season.astype(str), season.astype(str)), len(m), len(m))
    m["season"] = season
    for col in ("team1", "team2", "toss_winner", "winner"):
        before = m[col]
        m[col], m[f"{col}_franchise"] = clean_ipl_team(before)
        log.record(ds, "matches", f"fix {col} spelling variants + add {col}_franchise",
                   changed_rows(before, m[col]), len(m), len(m))
    before = m.venue
    m["venue"] = before.map(venues)
    log.record(ds, "matches", "canonical venue names", changed_rows(before, m.venue), len(m), len(m))
    for col in ("winner_runs", "winner_wickets", "match_number"):
        if col in m:
            m[col] = pd.to_numeric(m[col], errors="coerce").astype("Int64")
    m = drop_exact_duplicates(m, log, ds, "matches")
    return {"ipl_deliveries": d, "ipl_matches": m.sort_values("date", ignore_index=True)}


# ---------------------------------------------------------------- f1

def bool_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert every 't'/'f' text column to boolean."""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) and df[col].notna().any() \
                and set(df[col].dropna().unique()) <= {"t", "f"}:
            df[col] = tf_to_bool(df[col])
    return df


def clean_f1(tables: dict[str, pd.DataFrame], log: CleaningLog,
             min_season: int = F1_MIN_SEASON) -> dict[str, pd.DataFrame]:
    """Denormalize the Jolpica tables into analysis tables for seasons >= `min_season`."""
    ds = "f1"
    t = {name: bool_columns(df) for name, df in tables.items()}
    n_bool = sum(1 for n in tables for c in tables[n].columns if str(t[n][c].dtype) == "boolean")
    log.record(ds, "(all)", "convert 't'/'f' text columns to booleans", 0, 0, 0, f"{n_bool} columns")

    season = t["season"][["id", "year"]].rename(columns={"id": "season_id", "year": "season"})
    circuit = t["circuit"][["id", "name", "locality", "country"]].rename(
        columns={"id": "circuit_id", "name": "circuit_name"})
    races = (t["round"].rename(columns={"id": "race_id", "name": "race_name", "number": "round"})
             .merge(season, on="season_id").merge(circuit, on="circuit_id"))
    all_races = len(races)
    races = races[races.season >= min_season]
    log.record(ds, "races", f"keep seasons >= {min_season}", all_races - len(races), all_races, len(races))
    races = races[["race_id", "season", "round", "race_name", "date", "circuit_name", "locality", "country",
                   "is_cancelled"]].sort_values(["season", "round"], ignore_index=True)
    races["date"] = pd.to_datetime(races.date)

    driver = t["driver"].assign(driver_name=lambda x: x.forename.str.strip() + " " + x.surname.str.strip())
    driver = driver[["id", "driver_name", "abbreviation", "nationality"]].rename(
        columns={"id": "driver_id", "abbreviation": "driver_code", "nationality": "driver_nationality"})
    team = t["team"][["id", "name"]].rename(columns={"id": "team_id", "name": "team_name"})
    team_driver = t["teamdriver"][["id", "driver_id", "team_id"]].rename(columns={"id": "team_driver_id"})
    round_entry = t["roundentry"][["id", "car_number", "team_driver_id"]].rename(columns={"id": "round_entry_id"})
    session = t["session"][["id", "round_id", "type", "number", "is_cancelled"]].rename(
        columns={"id": "session_id", "round_id": "race_id", "type": "session_type",
                 "number": "session_number", "is_cancelled": "session_cancelled"})
    race_cols = races[["race_id", "season", "round", "race_name"]]

    se = t["sessionentry"].rename(columns={"id": "session_entry_id"})
    entries = (se.merge(session, on="session_id").merge(race_cols, on="race_id")
               .merge(round_entry, on="round_entry_id", how="left")
               .merge(team_driver, on="team_driver_id", how="left")
               .merge(driver, on="driver_id", how="left").merge(team, on="team_id", how="left"))
    log.record(ds, "session_results", f"join entries to sessions/drivers/teams, keep seasons >= {min_season}",
               len(se) - len(entries), len(se), len(entries))
    entries["time_ms"] = parse_duration_ms(entries["time"])
    bad_time = int((entries.time_ms.isna() & entries["time"].notna()).sum())
    log.record(ds, "session_results", "parse time 'HH:MM:SS.fff' -> time_ms", bad_time, len(entries), len(entries),
               "unparseable values set to NULL" if bad_time else "all values parsed")
    entries = entries.rename(columns={"status": "status_code"})
    entries["status_code"] = entries.status_code.astype("Int64")
    for col in ("grid", "position", "laps_completed", "fastest_lap_rank", "car_number"):
        entries[col] = pd.to_numeric(entries[col], errors="coerce").astype("Int64")
    key_cols = ["session_entry_id", "season", "round", "race_name", "session_type", "session_number",
                "driver_name", "driver_code", "team_name"]
    results = entries[key_cols + ["car_number", "grid", "position", "points", "laps_completed",
                                  "is_classified", "status_code", "detail", "time_ms", "fastest_lap_rank",
                                  "session_cancelled"]]
    results = results.sort_values(["season", "round", "session_type", "position"], ignore_index=True)

    lap = t["lap"].rename(columns={"id": "lap_id", "number": "lap_number"})
    laps = lap.merge(entries[key_cols], on="session_entry_id")
    log.record(ds, "laps", f"keep laps of seasons >= {min_season}", len(lap) - len(laps), len(lap), len(laps))
    laps["time_ms"] = parse_duration_ms(laps["time"])
    laps["lap_number"] = laps.lap_number.astype("Int64")
    laps["position"] = laps.position.astype("Int64")
    no_number = int(laps.lap_number.isna().sum())
    log.record(ds, "laps", "keep laps without lap_number (fastest-lap-only records)", no_number, len(laps), len(laps),
               "kept; is_entry_fastest_lap is true for these")
    laps = laps[key_cols[1:] + ["lap_number", "position", "time_ms", "average_speed", "is_deleted",
                                "is_entry_fastest_lap"]]

    pit = t["pitstop"].rename(columns={"number": "stop_number"})
    pits = (pit.merge(entries[key_cols], on="session_entry_id")
            .merge(lap[["lap_id", "lap_number"]], on="lap_id", how="left"))
    log.record(ds, "pit_stops", f"keep pit stops of seasons >= {min_season}", len(pit) - len(pits), len(pit), len(pits))
    pits["duration_ms"] = parse_duration_ms(pits["duration"])
    pits["lap_number"] = pits.lap_number.astype("Int64")
    pits = pits[key_cols[1:] + ["stop_number", "lap_number", "duration_ms", "local_timestamp"]]

    def standings(name: str, who: pd.DataFrame, on: str) -> pd.DataFrame:
        raw = t[name]
        df = raw[raw.year >= min_season].merge(who, on=on, how="left").rename(columns={"year": "season"})
        # Latest standings = rows from the last session of the last round in the data for each
        # season. For completed seasons that is the final table; for the current season it is "so far".
        final_round = df.groupby("season").round_number.transform("max")
        last_session = df[df.round_number == final_round].groupby("season").session_number.max()
        df["is_latest"] = (df.round_number == final_round) & (df.session_number == df.season.map(last_session))
        log.record(ds, name.replace("championship", "_standings"), f"keep seasons >= {min_season}; flag latest standings",
                   len(raw) - len(df), len(raw), len(df), "standings after every session; is_latest = season end for completed seasons, current table for the ongoing one")
        return df

    drivers_st = standings("driverchampionship", driver[["driver_id", "driver_name", "driver_code"]], "driver_id")
    drivers_st = drivers_st[["season", "round_number", "session_number", "is_latest", "driver_name", "driver_code",
                             "points", "position", "win_count", "highest_finish", "is_eligible"]]
    teams_st = standings("teamchampionship", team, "team_id")
    teams_st = teams_st[["season", "round_number", "session_number", "is_latest", "team_name",
                         "points", "position", "win_count", "highest_finish", "is_eligible"]]

    out = {"f1_races": races, "f1_session_results": results.drop(columns="session_entry_id"),
           "f1_laps": laps, "f1_pit_stops": pits, "f1_driver_standings": drivers_st, "f1_team_standings": teams_st}
    return {name: drop_exact_duplicates(df, log, ds, name.removeprefix("f1_")) for name, df in out.items()}


# ---------------------------------------------------------------- crop

def clean_crop(tables: dict[str, pd.DataFrame], log: CleaningLog) -> dict[str, pd.DataFrame]:
    """Clean crop production (names, whitespace padding, duplicates)."""
    ds, name = "crop_production", "crop_production"
    df = snake_case_columns(tables[name], log, ds, name)
    df = trim_text_columns(df, log, ds, name, ["state_name", "district_name", "season", "crop"])
    df["crop_year"] = df.crop_year.astype(int)
    missing = int(df.production.isna().sum())
    log.record(ds, name, "keep missing production as NULL (not imputed)", missing, len(df), len(df),
               "filter with production IS NOT NULL when aggregating")
    df = drop_exact_duplicates(df, log, ds, name, " (after trimming)")
    return {name: df}


CLEANERS = {"olist": clean_olist, "ipl": clean_ipl, "f1": clean_f1, "crop_production": clean_crop}


def write_outputs(name: str, tables: dict[str, pd.DataFrame], out_dir: Path) -> list[Path]:
    """Olist -> one SQLite DB; everything else -> one CSV per table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if name == "olist":
        path = out_dir / "olist.sqlite"
        write_sqlite(tables, path)
        return [path]
    paths = []
    for table, df in tables.items():
        path = out_dir / f"{table}.csv"
        df.to_csv(path, index=False)
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Clean raw datasets into data/processed/.")
    parser.add_argument("names", nargs="*", help=f"Datasets: {', '.join(CLEANERS)} (default: all)")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--log-path", type=Path, default=LOG_PATH)
    args = parser.parse_args(argv)
    unknown = [n for n in args.names if n not in CLEANERS]
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(unknown)}")

    log = CleaningLog()
    for name in args.names or list(CLEANERS):
        print(f"\n== {name}")
        cleaned = CLEANERS[name](LOADERS[name](args.raw_dir), log)
        for path in write_outputs(name, cleaned, args.out_dir):
            print(f"  wrote {path}")
        for table, df in cleaned.items():
            print(f"  {table}: {len(df):,} rows x {df.shape[1]} columns")
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.write_text(log.to_markdown(), encoding="utf-8")
    print(f"\nCleaning log ({len(log.entries)} steps) -> {args.log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
