"""Tests for scripts/eval_questions.py with a fake SQL generator (no LLM, no network)."""
import importlib.util
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from datacopilot.questions import SET_FILES, write_jsonl

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_questions.py"
spec = importlib.util.spec_from_file_location("eval_questions", SCRIPT)
eq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eq)

SCHEMA = {"crop_production": [{"name": "crop", "type": "VARCHAR", "samples": ["Rice"]},
                              {"name": "production", "type": "DOUBLE", "samples": [1.0]}]}
GOLD_SQL = "SELECT crop, SUM(production) AS total FROM crop_production GROUP BY crop ORDER BY total DESC"


@pytest.fixture
def conn():
    c = duckdb.connect()
    c.execute("CREATE TABLE crop_production (crop VARCHAR, production DOUBLE)")
    c.execute("INSERT INTO crop_production VALUES ('Rice', 10), ('Rice', 5), ('Wheat', 7)")
    yield c
    c.close()


def row(gold_answer=None):
    return {"id": "crop-001", "question_set": "csv", "dataset": "crop", "difficulty": "medium",
            "question": "Total production per crop?", "gold_sql": GOLD_SQL,
            "gold_answer": gold_answer or [["Rice", 15], ["Wheat", 7]]}


def fake(*sqls):
    """Generator returning the given SQL strings in turn."""
    calls = iter(sqls)
    return lambda question, schema, **kw: next(calls)


def test_evaluate_question_correct(conn):
    rec = eq.evaluate_question(row(), conn, SCHEMA, generate=fake(GOLD_SQL))
    assert rec["status"] == "correct" and rec["strict"] and rec["relaxed"]
    assert rec["pred_sql"] == GOLD_SQL and rec["seconds"] >= 0


def test_evaluate_question_relaxed_only_for_extra_column_and_rounding(conn):
    gold = [[7.33]]  # AVG(10, 5, 7) rounded to 2 places in the gold answer
    r = {**row(gold), "gold_sql": "SELECT ROUND(AVG(production), 2) FROM crop_production"}
    rec = eq.evaluate_question(r, conn, SCHEMA, generate=fake(
        "SELECT COUNT(*) AS n, AVG(production) AS avg FROM crop_production"))
    assert rec["status"] == "wrong_result" and not rec["strict"] and rec["relaxed"]


def test_evaluate_question_order_matters_with_order_by(conn):
    rec = eq.evaluate_question(row(), conn, SCHEMA, generate=fake(
        "SELECT crop, SUM(production) FROM crop_production GROUP BY crop ORDER BY crop DESC"))
    assert rec["status"] == "wrong_result" and not rec["strict"] and not rec["relaxed"]


def test_validated_mode_retries_after_validation_error(conn):
    rec = eq.evaluate_question(row(), conn, SCHEMA, generate=fake(
        "SELECT crop, SUM(yield) FROM crop_production GROUP BY crop", GOLD_SQL))
    assert rec["status"] == "correct"


def test_baseline_mode_reports_execution_failure_without_retry(conn):
    rec = eq.evaluate_question(row(), conn, SCHEMA, use_validator=False, generate=fake(
        "SELECT crop, SUM(yield) FROM crop_production GROUP BY crop", GOLD_SQL))
    assert rec["status"] == "execution_failed" and not rec["strict"]
    assert "yield" in rec["pred_sql"]


def test_validated_mode_reports_validation_failure_after_retry(conn):
    bad = "SELECT nope FROM crop_production"
    rec = eq.evaluate_question(row(), conn, SCHEMA, generate=fake(bad, bad))
    assert rec["status"] == "validation_failed" and "after one retry" in rec["error"]


def test_status_of():
    assert eq.status_of("Validation failed: unknown column") == "validation_failed"
    assert eq.status_of("Execution failed: boom") == "execution_failed"


def test_result_rows_normalizes_like_gold():
    df = pd.DataFrame({"a": [1.0, 2.123456], "b": ["x", None]})
    assert eq.result_rows(df) == [[1, "x"], [2.1235, None]]


def test_load_questions_filters_split_and_adds_heldout(tmp_path):
    base = {"dataset": "crop", "gold_sql": "SELECT 1", "gold_answer": [[1]], "unanswerable_type": None,
            "difficulty": "easy"}
    write_jsonl(tmp_path / SET_FILES["csv"], [
        {**base, "id": "crop-001", "question": "a?", "answerable": True, "split": "train"},
        {**base, "id": "crop-002", "question": "b?", "answerable": True, "split": "test"}])
    write_jsonl(tmp_path / "human_heldout.jsonl", [
        {**base, "id": "human-001", "question": "c?", "answerable": True, "split": "heldout", "language": "en"},
        {**base, "id": "human-002", "question": "d?", "answerable": False, "split": "heldout", "language": "en",
         "gold_sql": None, "gold_answer": None, "unanswerable_type": "subjective"}])
    rows = eq.load_questions(["csv", "heldout"], "test", tmp_path)
    assert [(r["id"], r["question_set"]) for r in rows] == [("crop-002", "csv"), ("human-001", "heldout")]
    assert len(eq.load_questions(["csv"], "all", tmp_path)) == 2


def test_accuracy_table_and_report():
    recs = pd.DataFrame([
        {"mode": "validated", "question_set": "csv", "dataset": "crop", "difficulty": "easy",
         "strict": True, "relaxed": True, "status": "correct"},
        {"mode": "validated", "question_set": "hinglish", "dataset": "f1", "difficulty": "hard",
         "strict": False, "relaxed": True, "status": "wrong_result"},
        {"mode": "baseline", "question_set": "csv", "dataset": "crop", "difficulty": "easy",
         "strict": False, "relaxed": False, "status": "execution_failed"},
    ])
    t = eq.accuracy_table(recs, "question_set")
    overall = t[(t["mode"] == "validated") & (t.question_set == "all")].iloc[0]
    assert overall.n == 2 and overall.strict == 0.5 and overall.relaxed == 1.0
    text = eq.report(recs, "Test")
    assert text.startswith("# Test")
    assert "| **all** | 2 | 50.0% | 100.0% |" in text
    assert "## Outcomes" in text and "execution_failed" in text


def test_main_writes_csv_per_mode_and_report(tmp_path, monkeypatch):
    qdir, out = tmp_path / "q", tmp_path / "out"
    write_jsonl(qdir / SET_FILES["csv"], [{
        "id": "crop-001", "dataset": "crop", "question": "Total production per crop?", "gold_sql": GOLD_SQL,
        "gold_answer": [["Rice", 15], ["Wheat", 7]], "answerable": True, "unanswerable_type": None,
        "difficulty": "medium", "split": "test"}])
    db = duckdb.connect()
    db.execute("CREATE TABLE crop_production AS SELECT * FROM (VALUES ('Rice', 15.0), ('Wheat', 7.0)) t(crop, production)")
    monkeypatch.setattr(eq.Datasets, "__call__", lambda self, ds: (db, SCHEMA))
    monkeypatch.setattr(eq, "generate_sql", lambda question, schema, **kw: GOLD_SQL)
    report_path = tmp_path / "report.md"
    assert eq.main(["--sets", "csv", "--questions-dir", str(qdir), "--out-dir", str(out),
                    "--report", str(report_path)]) == 0
    files = sorted(p.name.split("_")[1] for p in out.glob("questions_*.csv"))
    assert files == ["baseline", "validated"]
    assert "| **all** | 1 | 100.0% | 100.0% | 100.0% | 100.0% |" in report_path.read_text()
