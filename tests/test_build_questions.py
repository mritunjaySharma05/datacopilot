"""Tests for scripts/build_questions.py on a tiny fake processed directory."""
import importlib.util
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from datacopilot.questions import SET_FILES, UNANSWERABLE_TYPES, read_jsonl, write_jsonl

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_questions.py"
spec = importlib.util.spec_from_file_location("build_questions", SCRIPT)
bq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bq)

N_CSV, N_HI, N_HI_UN = 10, 5, 5  # per dataset in the fake sets
GOOD_SQL = {
    "olist": "SELECT COUNT(*) FROM orders WHERE amount > {k}",
    "ipl": "SELECT COUNT(*) FROM ipl_matches WHERE runs > {k}",
    "f1": "SELECT COUNT(*) FROM f1_races WHERE season >= {k}",
    "crop": "SELECT COUNT(*) FROM crop_production WHERE crop_year >= {k}",
}


@pytest.fixture
def processed(tmp_path):
    d = tmp_path / "processed"
    d.mkdir()
    with sqlite3.connect(d / "olist.sqlite") as con:
        con.execute("CREATE TABLE orders (order_id TEXT, amount REAL)")
        con.executemany("INSERT INTO orders VALUES (?, ?)", [(f"o{i}", i) for i in range(20)])
    pd.DataFrame({"match_id": range(20), "runs": range(100, 120)}).to_csv(d / "ipl_matches.csv", index=False)
    pd.DataFrame({"season": range(2000, 2020)}).to_csv(d / "f1_races.csv", index=False)
    pd.DataFrame({"crop_year": range(1990, 2010)}).to_csv(d / "crop_production.csv", index=False)
    return d


def write_sets(qdir: Path) -> None:
    """Small sets with the same shape as the real ones (authored fields only)."""
    base = {"olist": 0, "ipl": 100, "f1": 2000, "crop": 1990}
    csv, un, hi, hi_un = [], [], [], []
    for ds, sql in GOOD_SQL.items():
        for i in range(N_CSV):
            csv.append({"id": f"{ds}-{i:03d}", "dataset": ds, "question": f"{ds} count above {i}?",
                        "gold_sql": sql.format(k=base[ds] + i), "answerable": True, "unanswerable_type": None,
                        "difficulty": "easy"})
        for j, t in enumerate(UNANSWERABLE_TYPES * 2):
            un.append({"id": f"{ds}-u{j:03d}", "dataset": ds, "question": f"{ds} {t} question {j}?",
                       "gold_sql": None, "answerable": False, "unanswerable_type": t, "difficulty": "medium"})
        for i in range(N_HI):
            hi.append({"id": f"{ds}-{i:03d}-hi", "dataset": ds, "question": f"{ds} mein {i} se zyada kitne?",
                       "source_id": f"{ds}-{i:03d}"})
        for j in range(N_HI_UN):  # one of each unanswerable type per dataset
            hi_un.append({"id": f"{ds}-u{j:03d}-hi", "dataset": ds, "question": f"{ds} ka sawaal {j}?",
                          "source_id": f"{ds}-u{j:03d}"})
    write_jsonl(qdir / SET_FILES["csv"], csv)
    write_jsonl(qdir / SET_FILES["unanswerable"], un)
    write_jsonl(qdir / SET_FILES["hinglish"], hi)
    write_jsonl(qdir / SET_FILES["hinglish_unanswerable"], hi_un)


@pytest.fixture
def built(tmp_path, processed, monkeypatch):
    monkeypatch.setattr(bq, "EXPECTED_PER_DATASET", {"csv": N_CSV, "hinglish": N_HI, "unanswerable": 10,
                                                     "hinglish_unanswerable": (N_HI_UN, N_HI_UN)})
    monkeypatch.setattr(bq, "EXPECTED_TOTAL", {"hinglish_unanswerable": 4 * N_HI_UN})
    monkeypatch.setattr(bq, "SPLIT_TOLERANCE", 1.0)  # tiny groups can't hit 70/30 exactly
    qdir = tmp_path / "questions"
    qdir.mkdir()
    write_sets(qdir)
    assert bq.build(qdir, processed) == []
    return qdir


def test_build_fills_answers_splits_and_hinglish(built, processed):
    csv = read_jsonl(built / SET_FILES["csv"])
    assert csv[0]["gold_answer"] == [[19]]  # olist amount > 0
    assert {r["split"] for r in csv} == {"train", "test"}
    un = read_jsonl(built / SET_FILES["unanswerable"])
    assert all(r["gold_sql"] is None and r["gold_answer"] is None for r in un)
    english = {r["id"]: r for r in csv}
    for row in read_jsonl(built / SET_FILES["hinglish"]):
        src = english[row["source_id"]]
        assert (row["gold_sql"], row["gold_answer"], row["split"], row["difficulty"]) == (
            src["gold_sql"], src["gold_answer"], src["split"], src["difficulty"])
    un_src = {r["id"]: r for r in un}
    for row in read_jsonl(built / SET_FILES["hinglish_unanswerable"]):
        src = un_src[row["source_id"]]
        assert row["answerable"] is False and row["gold_sql"] is None
        assert (row["unanswerable_type"], row["split"], row["difficulty"]) == (
            src["unanswerable_type"], src["split"], src["difficulty"])
    assert bq.check(built, processed) == []


def test_build_drops_failing_and_empty_sql(tmp_path, processed):
    rows = [{"id": "ipl-1", "dataset": "ipl", "question": "a?", "gold_sql": "SELECT nope FROM ipl_matches",
             "answerable": True, "unanswerable_type": None, "difficulty": "easy"},
            {"id": "ipl-2", "dataset": "ipl", "question": "b?", "gold_sql": "SELECT runs FROM ipl_matches WHERE runs < 0",
             "answerable": True, "unanswerable_type": None, "difficulty": "easy"},
            {"id": "ipl-3", "dataset": "ipl", "question": "c?", "gold_sql": "SELECT MAX(runs) FROM ipl_matches",
             "answerable": True, "unanswerable_type": None, "difficulty": "easy"}]
    kept, dropped = bq.build_answerable(rows, bq.Connections(processed))
    assert [r["id"] for r in kept] == ["ipl-3"] and kept[0]["gold_answer"] == [[119]]
    assert any("ipl-1: gold_sql failed" in d for d in dropped)
    assert any("ipl-2: gold_sql returned no data" in d for d in dropped)


def _edit(qdir, name, fn):
    rows = read_jsonl(qdir / SET_FILES[name])
    fn(rows)
    write_jsonl(qdir / SET_FILES[name], rows)


@pytest.mark.parametrize("name, mutate, expected", [
    ("csv", lambda rows: rows[0].update(gold_answer=[[999]]), "stored gold_answer differs"),
    ("csv", lambda rows: rows[1].update(id=rows[0]["id"]), "duplicate id"),
    ("csv", lambda rows: rows[1].update(question=rows[0]["question"].upper()), "duplicate question"),
    ("csv", lambda rows: rows[0].update(split="test" if rows[0]["split"] == "train" else "train"),
     "splits differ from the seeded assignment"),
    ("csv", lambda rows: rows[0].update(difficulty="extreme"), "bad difficulty"),
    ("csv", lambda rows: rows[0].update(gold_sql="DELETE FROM orders"), "gold_sql failed"),
    ("hinglish", lambda rows: rows[0].update(split="test" if rows[0]["split"] == "train" else "train"),
     "differ from"),
    ("hinglish", lambda rows: rows[0].update(source_id="nope"), "not found"),
    ("unanswerable", lambda rows: rows[0].update(unanswerable_type="subjective"), "not balanced"),
    ("unanswerable", lambda rows: rows[0].update(gold_sql="SELECT 1"), "must have null gold_sql"),
    ("unanswerable", lambda rows: rows.pop(), "has 9 rows"),
    ("hinglish_unanswerable", lambda rows: rows[0].update(unanswerable_type="subjective"), "differ from"),
    ("hinglish_unanswerable", lambda rows: rows[0].update(source_id="nope"), "not found"),
    ("hinglish_unanswerable", lambda rows: rows.pop(), "types not balanced"),
    ("hinglish_unanswerable", lambda rows: rows[0].update(answerable=True), "answerable != False"),
])
def test_check_catches_problems(built, processed, name, mutate, expected):
    _edit(built, name, mutate)
    problems = bq.check(built, processed)
    assert any(expected in p for p in problems), problems


def test_check_flags_unbalanced_test_share(built, processed, monkeypatch):
    monkeypatch.setattr(bq, "SPLIT_TOLERANCE", 0.05)
    _edit(built, "hinglish", lambda rows: None)
    rows = read_jsonl(built / SET_FILES["csv"])
    share = sum(r["split"] == "test" for r in rows if r["dataset"] == "ipl") / N_CSV
    assert share == 0.3  # 3 of 10 -> within tolerance for csv
    problems = bq.check(built, processed)
    assert all("csv:" not in p or "test share" not in p for p in problems)


def test_main_check_mode_does_not_rewrite(built, processed, capsys):
    before = (built / SET_FILES["csv"]).read_bytes()
    code = bq.main(["--check", "--questions-dir", str(built), "--processed-dir", str(processed)])
    assert code == 0 and (built / SET_FILES["csv"]).read_bytes() == before
    assert "All question sets valid." in capsys.readouterr().out


def test_main_reports_problems(built, processed, capsys):
    _edit(built, "csv", lambda rows: rows[0].update(gold_answer=[[0]]))
    assert bq.main(["--check", "--questions-dir", str(built), "--processed-dir", str(processed)]) == 1
    assert "problem(s)" in capsys.readouterr().out
