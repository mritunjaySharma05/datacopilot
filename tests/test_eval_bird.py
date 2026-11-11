"""Tests for scripts/eval_bird.py using a tiny fake BIRD layout (no LLM, no network)."""
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from conftest import build_tiny_sqlite

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_bird.py"
spec = importlib.util.spec_from_file_location("eval_bird", SCRIPT)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

QUESTIONS = [
    {
        "question_id": 0, "db_id": "shop", "difficulty": "simple",
        "question": "How many items cost more than 5?", "evidence": "more than 5 refers to price > 5",
        "SQL": "SELECT COUNT(*) FROM items WHERE price > 5",
    },
    {
        "question_id": 1, "db_id": "shop", "difficulty": "moderate",
        "question": "Total quantity ordered per item name?", "evidence": "",
        "SQL": "SELECT T1.`Item Name`, SUM(T2.qty) FROM items AS T1 "
               "JOIN orders AS T2 ON T1.id = T2.item_id GROUP BY T1.`Item Name`",
    },
]


@pytest.fixture
def bird_dir(tmp_path):
    root = tmp_path / "bird"
    (root / "dev_x" / "dev_databases" / "shop").mkdir(parents=True)
    (root / "dev_x" / "dev.json").write_text(json.dumps(QUESTIONS))
    build_tiny_sqlite(root / "dev_x" / "dev_databases" / "shop" / "shop.sqlite")
    return root


# --- helpers ---

def test_find_dev_json_and_db(bird_dir):
    assert ev.find_dev_json(bird_dir).name == "dev.json"
    assert ev.find_db(bird_dir, "shop").name == "shop.sqlite"


def test_find_db_missing(bird_dir):
    with pytest.raises(FileNotFoundError):
        ev.find_db(bird_dir, "nope")


def test_load_questions_takes_first_n(bird_dir):
    assert [q["question_id"] for q in ev.load_questions(bird_dir, 1)] == [0]


def test_results_match_ignores_order_and_duplicates():
    assert ev.results_match([(1, "a"), (2, "b")], [(2, "b"), (1, "a"), (1, "a")])
    assert not ev.results_match([(1, "a")], [(1, "b")])
    assert ev.results_match([(3,)], [(3.0,)])  # int vs float with equal value


def test_to_rows_maps_nan_back_to_none():
    df = pd.DataFrame([(1, None), (2, 2.5)], columns=["a", "b"])  # None becomes NaN
    assert ev.to_rows(df) == [(1, None), (2, 2.5)]


def test_output_path_names_mode_and_timestamp(tmp_path):
    now = datetime(2026, 11, 11, 9, 30, 0)
    assert ev.output_path(tmp_path, True, now).name == "bird_validated_20261111_093000.csv"
    assert ev.output_path(tmp_path, False, now).name == "bird_baseline_20261111_093000.csv"


# --- evaluate_question ---

def test_evaluate_question_correct(bird_dir, fake_llm):
    gen = fake_llm("SELECT COUNT(*) AS n FROM items WHERE price > 5")
    record = ev.evaluate_question(QUESTIONS[0], bird_dir, generate=gen)
    assert record["correct"] is True
    assert record["error"] == ""
    assert gen.calls[0]["evidence"] == "more than 5 refers to price > 5"
    assert gen.calls[0]["dialect"] == "sqlite"


def test_evaluate_question_wrong_result(bird_dir, fake_llm):
    gen = fake_llm("SELECT COUNT(*) FROM items")
    record = ev.evaluate_question(QUESTIONS[0], bird_dir, generate=gen)
    assert record["correct"] is False
    assert record["pred_sql"] == "SELECT COUNT(*) FROM items"


def test_evaluate_question_empty_evidence_sent_as_none(bird_dir, fake_llm):
    gen = fake_llm(
        'SELECT i."Item Name", SUM(o.qty) FROM items i JOIN orders o ON i.id = o.item_id GROUP BY 1'
    )
    record = ev.evaluate_question(QUESTIONS[1], bird_dir, generate=gen)
    assert record["correct"] is True
    assert gen.calls[0]["evidence"] is None


def test_evaluate_question_validated_retries(bird_dir, fake_llm):
    gen = fake_llm("SELECT COUNT(*) FROM item_list", "SELECT COUNT(*) FROM items WHERE price > 5")
    record = ev.evaluate_question(QUESTIONS[0], bird_dir, use_validator=True, generate=gen)
    assert record["correct"] is True
    assert "Unknown tables: item_list" in gen.calls[1]["feedback"]


def test_evaluate_question_baseline_no_retry(bird_dir, fake_llm):
    gen = fake_llm("SELECT COUNT(*) FROM item_list", "SELECT COUNT(*) FROM items WHERE price > 5")
    record = ev.evaluate_question(QUESTIONS[0], bird_dir, use_validator=False, generate=gen)
    assert record["correct"] is False
    assert "no such table" in record["error"]
    assert len(gen.calls) == 1


def test_evaluate_question_gold_failure_is_recorded(bird_dir, fake_llm):
    broken = dict(QUESTIONS[0], SQL="SELECT nope FROM items")
    record = ev.evaluate_question(broken, bird_dir, generate=fake_llm("SELECT 1"))
    assert record["correct"] is False
    assert record["error"].startswith("Gold SQL failed")


# --- main ---

def test_main_writes_csv_and_prints_accuracy(bird_dir, tmp_path, monkeypatch, capsys):
    replies = iter(["SELECT COUNT(*) FROM items WHERE price > 5", "SELECT 1"])
    monkeypatch.setattr(ev, "generate_sql", lambda q, s, **kw: next(replies))
    out_dir = tmp_path / "results"
    assert ev.main(["--n", "2", "--no-validator", "--bird-dir", str(bird_dir), "--out-dir", str(out_dir)]) == 0

    out = capsys.readouterr().out
    assert "Execution accuracy (baseline): 1/2 = 50.0%" in out
    files = list(out_dir.glob("bird_baseline_*.csv"))
    assert len(files) == 1
    df = pd.read_csv(files[0])
    assert df["correct"].tolist() == [True, False]
    assert {"question_id", "gold_sql", "pred_sql", "error", "seconds"} <= set(df.columns)


def test_main_missing_bird_dir(tmp_path, capsys):
    assert ev.main(["--bird-dir", str(tmp_path / "none")]) == 1
    assert "download_bird.py" in capsys.readouterr().err
