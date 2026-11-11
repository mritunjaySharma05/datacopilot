"""Tests for scripts/data_quality.py on a tiny fake data/raw tree."""
import importlib.util
from pathlib import Path

import pytest

from datacopilot.raw_data import LOADERS
from fake_raw import build_all

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "data_quality.py"
spec = importlib.util.spec_from_file_location("data_quality", SCRIPT)
dq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dq)


@pytest.fixture
def raw(tmp_path):
    return build_all(tmp_path / "raw")


def counts(findings):
    return {f.check: f.count for f in findings}


def test_olist_checks(raw):
    c = counts(dq.olist_checks(LOADERS["olist"](raw)))
    assert c["order_items: price <= 0"] == 1
    assert c["order_payments: payment_value <= 0"] == 1
    assert c["order_payments: payment_installments == 0"] == 1
    assert c["order_reviews: review_id not unique"] == 2
    assert c["orders: handed to carrier before approval"] == 1
    assert c["orders: status 'delivered' but no delivery date"] == 1
    assert c["orders: no order_items rows"] == 1
    assert c["products: category without English translation"] == 1
    assert c["products: a weight/dimension of 0"] == 1
    assert c["geolocation: coordinates outside Brazil"] == 1
    assert c["geolocation: city containing stray symbols or escapes"] == 1


def test_ipl_checks(raw):
    c = counts(dq.ipl_checks(LOADERS["ipl"](raw)))
    assert c["season label is not a plain year"] == 4  # one match's deliveries
    assert c["ball numbers that collide if parsed as float (x.10 vs x.1)"] == 3
    assert c["team name spelling variants"] == 1
    assert c["venue name variants (city suffix / punctuation)"] == 1
    assert c["matches without ball-by-ball rows"] == 0


def test_ipl_ball_collisions_only_when_float(raw):
    d = LOADERS["ipl"](raw)["deliveries"]
    assert dq.ipl_ball_collisions(d[d.ball != "2.10"]) == 0


def test_f1_checks(raw):
    c = counts(dq.f1_checks(LOADERS["f1"](raw)))
    assert c["boolean columns stored as 't'/'f' text"] >= 5
    assert c["durations stored as 'HH:MM:SS.fff' text"] == 3  # sessionentry, lap, pitstop
    assert c["empty tables"] == 1
    assert c["sessionentry -> roundentry orphans"] == 0


def test_crop_checks(raw):
    c = counts(dq.crop_checks(LOADERS["crop_production"](raw)))
    assert c["State_Name: padded with spaces"] == 1
    assert c["Season: padded with spaces"] == 3
    assert c["Crop: padded with spaces"] == 2
    assert c["Production missing"] == 1
    assert c["Production == 0"] == 1
    assert c["duplicate (state, district, year, season, crop) keys"] == 1


def test_main_writes_one_report_per_dataset(raw, tmp_path, capsys):
    reports = tmp_path / "reports"
    assert dq.main(["--raw-dir", str(raw), "--report-dir", str(reports)]) == 0
    assert sorted(p.name for p in reports.glob("*.md")) == ["crop_production.md", "f1.md", "ipl.md", "olist.md"]
    text = (reports / "crop_production.md").read_text(encoding="utf-8")
    assert "# Data quality: crop_production" in text
    assert "Season: padded with spaces" in text
    assert "not snake_case" in text
    assert "olist" in capsys.readouterr().out


def test_main_rejects_unknown_dataset():
    with pytest.raises(SystemExit):
        dq.main(["nope"])
