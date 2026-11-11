"""Tests for scripts/prepare_data.py on a tiny fake data/raw tree."""
import importlib.util
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from datacopilot.dataprep import CleaningLog
from datacopilot.raw_data import LOADERS
from fake_raw import build_all

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_data.py"
spec = importlib.util.spec_from_file_location("prepare_data", SCRIPT)
pdx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pdx)


@pytest.fixture
def raw(tmp_path):
    return build_all(tmp_path / "raw")


def steps(log, dataset, table=None):
    return {e["step"]: e for e in log.entries if e["dataset"] == dataset and (table is None or e["table"] == table)}


# --- small helpers ---

def test_split_ball_keeps_ten():
    over, ball = pdx.split_ball(pd.Series(["0.1", "2.10", "19.6"]))
    assert over.tolist() == [0, 2, 19] and ball.tolist() == [1, 10, 6]


def test_venue_mapping_merges_suffix_and_punctuation_variants():
    s = pd.Series(["M Chinnaswamy Stadium"] * 3 + ["M.Chinnaswamy Stadium", "M Chinnaswamy Stadium, Bengaluru",
                                                    "Feroz Shah Kotla", "Arun Jaitley Stadium, Delhi"])
    m = pdx.venue_mapping(s)
    assert {m["M.Chinnaswamy Stadium"], m["M Chinnaswamy Stadium, Bengaluru"]} == {"M Chinnaswamy Stadium"}
    assert m["Feroz Shah Kotla"] == "Feroz Shah Kotla"  # renames are not merged
    assert m["Arun Jaitley Stadium, Delhi"] == "Arun Jaitley Stadium"


def test_clean_ipl_team():
    team, franchise = pdx.clean_ipl_team(pd.Series(["Rising Pune Supergiants", "Delhi Daredevils", "Mumbai Indians"]))
    assert team.tolist() == ["Rising Pune Supergiant", "Delhi Daredevils", "Mumbai Indians"]
    assert franchise.tolist() == ["Rising Pune Supergiant", "Delhi Capitals", "Mumbai Indians"]


def test_normalize_city():
    s = pd.Series(["São  Paulo", "sa£o paulo", "´teresopolis", None])
    out = pdx.normalize_city(s)
    assert out.tolist()[:3] == ["sao paulo", "sao paulo", "teresopolis"]
    assert pd.isna(out[3])


# --- per-dataset cleaning ---

def test_clean_crop(raw):
    log = CleaningLog()
    df = pdx.clean_crop(LOADERS["crop_production"](raw), log)["crop_production"]
    assert list(df.columns) == ["state_name", "district_name", "crop_year", "season", "crop", "area", "production"]
    assert set(df.season) == {"Kharif", "Whole Year"} and set(df.state_name) == {"Telangana", "Kerala"}
    assert len(df) == 3  # Telangana rows became exact duplicates after trimming
    s = steps(log, "crop_production")
    assert s["trim whitespace in season"]["rows_affected"] == 3
    assert s["drop exact duplicate rows (after trimming)"]["rows_affected"] == 1
    assert s["keep missing production as NULL (not imputed)"]["rows_affected"] == 1


def test_clean_ipl(raw):
    log = CleaningLog()
    out = pdx.clean_ipl(LOADERS["ipl"](raw), log)
    d, m = out["ipl_deliveries"], out["ipl_matches"]
    assert "ball" not in d.columns
    assert d.loc[d.ball_in_over == 10, "over"].unique().tolist() == [2]
    assert sorted(d.season.unique()) == [2008, 2016, 2017]
    assert d.venue.unique().tolist() == ["M Chinnaswamy Stadium"]
    assert "Rising Pune Supergiants" not in set(d.batting_team)
    assert set(d.batting_franchise) >= {"Royal Challengers Bengaluru", "Rising Pune Supergiant"}
    assert d.striker.unique().tolist() == ["A"]
    assert m.season.tolist() == [2008, 2016, 2017]
    assert m.team2_franchise.tolist()[:2] == ["Punjab Kings", "Delhi Capitals"]
    s = steps(log, "ipl", "deliveries")
    assert s["split ball (text) into over + ball_in_over, drop ball"]["rows_affected"] == 3
    assert s["season label -> calendar year of the match"]["rows_affected"] == 4


def test_clean_olist_and_sqlite(raw, tmp_path):
    log = CleaningLog()
    t = pdx.clean_olist(LOADERS["olist"](raw), log)
    assert "product_name_length" in t["products"].columns
    english = t["products"].product_category_name_english
    assert english[0] == "health_beauty" and pd.isna(english[1])  # pc_gamer has no translation
    geo = t["geolocation"]
    assert len(geo) == 2  # 3 sao paulo rows -> 1 after normalization + dedupe; madrid dropped
    assert set(geo.geolocation_city) == {"sao paulo", "rio de janeiro"}
    assert t["customers"].customer_city.tolist() == ["sao paulo", "rio de janeiro"]
    assert t["customers"].customer_state.tolist() == ["SP", "RJ"]
    assert t["order_reviews"].review_comment_message[0] == "great\nproduct"  # free text untouched
    assert pd.api.types.is_datetime64_any_dtype(t["orders"].order_purchase_timestamp)
    s = steps(log, "olist", "geolocation")
    assert s["drop coordinates outside Brazil"]["rows_affected"] == 1

    path = tmp_path / "olist.sqlite"
    pdx.write_sqlite(t, path)
    con = sqlite3.connect(path)
    assert con.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 3
    indexes = {r[1] for r in con.execute("SELECT * FROM sqlite_master WHERE type = 'index'")}
    assert "ix_order_items_order_id" in indexes


def test_clean_f1_filters_seasons_and_joins(raw):
    log = CleaningLog()
    out = pdx.clean_f1(LOADERS["f1"](raw), log, min_season=2014)
    races, res = out["f1_races"], out["f1_session_results"]
    assert races.season.tolist() == [2014]
    assert res[["season", "driver_name", "team_name", "position", "time_ms"]].values.tolist() == [
        [2014, "Nico Rosberg", "Mercedes", 1, 5578710]]
    assert out["f1_laps"].time_ms.tolist() == [94200, 93900]
    assert out["f1_pit_stops"][["lap_number", "duration_ms"]].values.tolist() == [[2, 22500]]
    st = out["f1_driver_standings"]
    assert st.loc[st.is_latest, "points"].tolist() == [25]
    assert out["f1_laps"].is_entry_fastest_lap.dtype == "boolean"
    assert steps(log, "f1", "races")["keep seasons >= 2014"]["rows_affected"] == 1


# --- CLI ---

def test_main_writes_outputs_and_log(raw, tmp_path):
    out_dir, log_path = tmp_path / "processed", tmp_path / "cleaning_log.md"
    assert pdx.main(["--raw-dir", str(raw), "--out-dir", str(out_dir), "--log-path", str(log_path)]) == 0
    names = sorted(p.name for p in out_dir.iterdir())
    assert "olist.sqlite" in names and "crop_production.csv" in names
    assert {"ipl_deliveries.csv", "ipl_matches.csv", "f1_races.csv", "f1_session_results.csv"} <= set(names)
    log = log_path.read_text(encoding="utf-8")
    for dataset in ("## olist", "## ipl", "## f1", "## crop_production"):
        assert dataset in log
    # ball must not come back as a float column from the CSV
    assert "ball" not in pd.read_csv(out_dir / "ipl_deliveries.csv").columns
