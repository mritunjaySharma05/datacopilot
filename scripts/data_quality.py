"""Profile the Module 3 raw datasets and write one markdown report per dataset.

Usage:
    python scripts/data_quality.py                 # all: olist, ipl, f1, crop_production
    python scripts/data_quality.py ipl crop_production
Reports: docs/data_quality/<name>.md
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from pandas.api.types import is_string_dtype as is_text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datacopilot.dataprep import Finding, fold_text, profile_markdown, profile_table, variant_groups  # noqa: E402
from datacopilot.raw_data import LOADERS  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "docs" / "data_quality"
# Pairs of venue names that may be one ground under an old and a new name. Only flagged
# for manual review when both appear; cleaning does not merge them.
POSSIBLE_VENUE_RENAMES = [
    ("Feroz Shah Kotla", "Arun Jaitley Stadium"),
    ("Sardar Patel Stadium", "Narendra Modi Stadium"),
    ("Punjab Cricket Association Stadium", "Punjab Cricket Association IS Bindra Stadium"),
    ("Sheikh Zayed Stadium", "Zayed Cricket Stadium"),
]
# Characters that never belong in a Brazilian city name: stray symbols seen in the
# Olist geolocation table (sa£o paulo, ´teresopolis) and URL/HTML escapes (d%26apos%3b).
STRAY_CITY_CHARS = r"[£´³º`%;&]"
# Rough bounding box of Brazil, to spot impossible coordinates.
BRAZIL_LAT, BRAZIL_LNG = (-34.0, 6.0), (-74.0, -34.0)
SOURCES = {
    "olist": "Kaggle olistbr/brazilian-ecommerce (data/raw/olist/)",
    "ipl": "Cricsheet ipl_csv2.zip (data/raw/ipl/: all_matches.csv + *_info.csv)",
    "f1": "Jolpica-F1 CSV dump (data/raw/f1/)",
    "crop_production": "Kaggle mirror of data.gov.in crop production (data/raw/crop_production/)",
}


def _examples(values, n: int = 3) -> str:
    return ", ".join(repr(v) for v in list(values)[:n])


def olist_checks(t: dict[str, pd.DataFrame]) -> list[Finding]:
    """Business-rule checks for Olist."""
    out = []
    items, pay, rev, orders = t["order_items"], t["order_payments"], t["order_reviews"], t["orders"]
    out.append(Finding("order_items: price <= 0", int((items.price <= 0).sum())))
    out.append(Finding("order_items: freight_value < 0", int((items.freight_value < 0).sum())))
    zero_pay = pay[pay.payment_value <= 0]
    out.append(Finding("order_payments: payment_value <= 0", len(zero_pay),
                       f"types: {zero_pay.payment_type.value_counts().to_dict()}"))
    out.append(Finding("order_payments: payment_installments == 0", int((pay.payment_installments == 0).sum())))
    out.append(Finding("order_payments: payment_type 'not_defined'", int((pay.payment_type == "not_defined").sum())))
    out.append(Finding("order_reviews: review_score outside 1-5", int((~rev.review_score.between(1, 5)).sum())))
    dup_reviews = rev.review_id.duplicated(keep=False)
    out.append(Finding("order_reviews: review_id not unique", int(dup_reviews.sum()),
                       "same review_id attached to several orders"))
    ts = {c: pd.to_datetime(orders[c], errors="coerce") for c in orders.columns if c.endswith(("_at", "_date", "timestamp"))}
    purchase = ts["order_purchase_timestamp"]
    out.append(Finding("orders: delivered to customer before purchase",
                       int((ts["order_delivered_customer_date"] < purchase).sum())))
    out.append(Finding("orders: handed to carrier before approval",
                       int((ts["order_delivered_carrier_date"] < ts["order_approved_at"]).sum())))
    delivered_no_date = (orders.order_status == "delivered") & orders.order_delivered_customer_date.isna()
    out.append(Finding("orders: status 'delivered' but no delivery date", int(delivered_no_date.sum())))
    out.append(Finding("orders: no order_items rows", int((~orders.order_id.isin(items.order_id)).sum())))
    cats = set(t["products"].product_category_name.dropna())
    missing_tr = sorted(cats - set(t["product_category_name_translation"].product_category_name))
    out.append(Finding("products: category without English translation", len(missing_tr), _examples(missing_tr)))
    zero_dims = (t["products"][["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]] == 0).any(axis=1)
    out.append(Finding("products: a weight/dimension of 0", int(zero_dims.sum())))
    geo = t["geolocation"]
    outside = ~geo.geolocation_lat.between(*BRAZIL_LAT) | ~geo.geolocation_lng.between(*BRAZIL_LNG)
    out.append(Finding("geolocation: coordinates outside Brazil", int(outside.sum())))
    # Accented letters (ã, é, ç) are fine; stray symbols and URL/HTML escapes are not.
    broken = geo.geolocation_city.str.contains(STRAY_CITY_CHARS, regex=True, na=False)
    out.append(Finding("geolocation: city containing stray symbols or escapes", int(broken.sum()),
                       _examples(geo.geolocation_city[broken].unique(), 6)))
    raw_n, folded_n = geo.geolocation_city.nunique(), geo.geolocation_city.map(fold_text).nunique()
    out.append(Finding("geolocation: city spellings that differ only by accents/case/punctuation",
                       raw_n - folded_n, f"{raw_n:,} raw names -> {folded_n:,} after folding"))
    return out


def ipl_ball_collisions(deliveries: pd.DataFrame) -> int:
    """Deliveries whose `ball` (over.ball) would clash with another if read as a float (2.10 vs 2.1)."""
    as_float = deliveries.ball.astype(float)
    return int(deliveries.assign(_f=as_float).duplicated(["match_id", "innings", "_f"]).sum())


def ipl_checks(t: dict[str, pd.DataFrame]) -> list[Finding]:
    """Checks for IPL deliveries and matches."""
    d, m = t["deliveries"], t["matches"]
    out = []
    odd_seasons = sorted(s for s in d.season.astype(str).unique() if not s.isdigit())
    out.append(Finding("season label is not a plain year", int(d.season.astype(str).isin(odd_seasons).sum()),
                       f"labels: {', '.join(odd_seasons)}"))
    out.append(Finding("ball numbers that collide if parsed as float (x.10 vs x.1)", ipl_ball_collisions(d),
                       "read `ball` as text and split into over / ball"))
    teams = pd.concat([d.batting_team, d.bowling_team])
    groups = variant_groups(teams, key=lambda s: fold_text(s).rstrip("s"))
    out.append(Finding("team name spelling variants", len(groups), "; ".join(" / ".join(g) for g in groups)))
    venue_key = lambda v: fold_text(str(v).split(",")[0])  # noqa: E731
    vgroups = variant_groups(d.venue, key=venue_key, limit=100)
    out.append(Finding("venue name variants (city suffix / punctuation)", len(vgroups),
                       "; ".join(" / ".join(g) for g in vgroups[:4])))
    venues = {venue_key(v) for v in d.venue.unique()}
    present = [f"{a} vs {b}" for a, b in POSSIBLE_VENUE_RENAMES
               if fold_text(a) in venues and fold_text(b) in venues]
    out.append(Finding("possible renamed grounds, both names present (not merged; verify manually)",
                       len(present), "; ".join(present)))
    for col in ["runs_off_bat", "extras", "wides", "noballs", "byes", "legbyes", "penalty"]:
        neg = int((d[col] < 0).sum())
        if neg:
            out.append(Finding(f"deliveries: negative {col}", neg))
    out.append(Finding("matches: no winner (no result / tie)", int(m.winner.isna().sum()),
                       f"outcome values: {m.get('outcome', pd.Series(dtype=str)).value_counts().to_dict()}"))
    no_deliveries = ~m.match_id.isin(d.match_id)
    out.append(Finding("matches without ball-by-ball rows", int(no_deliveries.sum())))
    out.append(Finding("deliveries without a match info file", int((~d.match_id.isin(m.match_id)).sum())))
    return out


def f1_checks(t: dict[str, pd.DataFrame]) -> list[Finding]:
    """Checks for the Jolpica-F1 dump."""
    out = []
    tf_cols = [f"{n}.{c}" for n, df in t.items() for c in df.columns
               if is_text(df[c]) and df[c].notna().any() and set(df[c].dropna().unique()) <= {"t", "f"}]
    out.append(Finding("boolean columns stored as 't'/'f' text", len(tf_cols), _examples(tf_cols, 6)))
    time_cols = [f"{n}.{c}" for n, df in t.items() for c in ("time", "duration")
                 if c in df.columns and is_text(df[c])]
    out.append(Finding("durations stored as 'HH:MM:SS.fff' text", len(time_cols), ", ".join(time_cols)))
    out.append(Finding("empty tables", sum(1 for df in t.values() if df.empty),
                       ", ".join(n for n, df in t.items() if df.empty)))
    se, re_, sess = t["sessionentry"], t["roundentry"], t["session"]
    out.append(Finding("sessionentry -> roundentry orphans", int((~se.round_entry_id.isin(re_.id)).sum())))
    out.append(Finding("sessionentry -> session orphans", int((~se.session_id.isin(sess.id)).sum())))
    out.append(Finding("sessions marked cancelled", int((sess.is_cancelled == "t").sum())))
    out.append(Finding("rounds marked cancelled", int((t["round"].is_cancelled == "t").sum())))
    race = se[se.session_id.isin(sess.id[sess.type == "R"])]
    out.append(Finding("race results with no finishing position", int(race.position.isna().sum())))
    out.append(Finding("integer-coded status column (meaning only in Jolpica source code)",
                       int(se.status.notna().sum()), f"codes: {sorted(se.status.dropna().astype(int).unique())}"))
    dc = t["driverchampionship"]
    out.append(Finding("standings stored after every session, not once per season", len(dc),
                       f"{dc.year.nunique()} seasons -> {len(dc):,} driver-standing rows"))
    return out


def crop_checks(t: dict[str, pd.DataFrame]) -> list[Finding]:
    """Checks for crop production."""
    c = t["crop_production"]
    out = []
    for col in ["State_Name", "District_Name", "Season", "Crop"]:
        s = c[col].astype(str)
        n = int((s != s.str.strip()).sum())
        if n:
            out.append(Finding(f"{col}: padded with spaces", n, _examples(s[s != s.str.strip()].unique())))
    out.append(Finding("Production missing", int(c.Production.isna().sum())))
    out.append(Finding("Production == 0", int((c.Production == 0).sum()), "may mean crop failure or missing data"))
    out.append(Finding("Area <= 0", int((c.Area <= 0).sum())))
    key = c.assign(**{k: c[k].astype(str).str.strip() for k in ["State_Name", "District_Name", "Season", "Crop"]})
    out.append(Finding("duplicate (state, district, year, season, crop) keys",
                       int(key.duplicated(["State_Name", "District_Name", "Crop_Year", "Season", "Crop"]).sum())))
    per_year = c.groupby("Crop_Year").size()
    thin = per_year[per_year < per_year.median() * 0.25]
    out.append(Finding("years with < 25% of the median row count (thin coverage)", len(thin),
                       ", ".join(f"{y}: {n:,} rows" for y, n in thin.items())))
    out.append(Finding("no unit columns (area / production units not stated in the file)", 1,
                       "official series: hectares and tonnes; check per crop"))
    return out


CHECKS = {"olist": olist_checks, "ipl": ipl_checks, "f1": f1_checks, "crop_production": crop_checks}


def run_dataset(name: str, raw_dir: Path, report_dir: Path) -> tuple[Path, list[Finding]]:
    """Profile one dataset and write its markdown report; return (path, findings)."""
    tables = LOADERS[name](raw_dir)
    profiles = {table: profile_table(df) for table, df in tables.items()}
    findings = CHECKS[name](tables)
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{name}.md"
    path.write_text(profile_markdown(name, profiles, findings, SOURCES[name]), encoding="utf-8")
    return path, findings


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Profile raw datasets into markdown reports.")
    parser.add_argument("names", nargs="*", help=f"Datasets: {', '.join(LOADERS)} (default: all)")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args(argv)
    unknown = [n for n in args.names if n not in LOADERS]
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(unknown)}")

    for name in args.names or list(LOADERS):
        path, findings = run_dataset(name, args.raw_dir, args.report_dir)
        print(f"\n== {name}: report -> {path}")
        for f in findings:
            if f.count:
                print(f"  {f.count:>9,}  {f.check}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
