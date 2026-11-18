import datetime as dt
import decimal
import json

import duckdb
import pytest

from datacopilot import questions as qs


def test_normalize_value():
    assert qs.normalize_value(3.14159265) == 3.1416
    assert qs.normalize_value(5.0) == 5 and isinstance(qs.normalize_value(5.0), int)
    assert qs.normalize_value(decimal.Decimal("2.50")) == 2.5
    assert qs.normalize_value(float("nan")) is None
    assert qs.normalize_value(dt.date(2024, 4, 15)) == "2024-04-15"
    assert qs.normalize_value(True) is True
    assert qs.normalize_value("x") == "x"


def test_run_gold_rejects_writes():
    conn = duckdb.connect()
    conn.execute("CREATE TABLE t AS SELECT 1 AS a")
    with pytest.raises(ValueError):
        qs.run_gold(conn, "DELETE FROM t", "ipl")
    assert qs.run_gold(conn, "SELECT a, 2.5 FROM t", "ipl") == [[1, 2.5]]


def test_answers_match_tolerance_and_order():
    assert qs.answers_match([[1, 2.00001]], [[1, 2.00002]], ordered=True)
    assert not qs.answers_match([[1, 2.0]], [[1, 2.1]], ordered=True)
    assert qs.answers_match([["a", 1], ["b", 2]], [["b", 2], ["a", 1]], ordered=False)
    assert not qs.answers_match([["a", 1], ["b", 2]], [["b", 2], ["a", 1]], ordered=True)
    assert not qs.answers_match([[1]], [[1], [2]], ordered=False)


def test_is_ordered_looks_at_outer_query():
    assert qs.is_ordered("SELECT a FROM t ORDER BY a")
    assert not qs.is_ordered("SELECT COUNT(*) FROM (SELECT a FROM t ORDER BY a LIMIT 3)")


def _rows(n, dataset="ipl", answerable=True, utype=None):
    return [{"id": f"{dataset}-{i}", "dataset": dataset, "answerable": answerable, "unanswerable_type": utype}
            for i in range(n)]


def test_assign_splits_is_seeded_and_stratified():
    rows = _rows(50, "ipl") + _rows(50, "f1")
    first, second = qs.assign_splits(rows), qs.assign_splits(list(reversed(rows)))
    assert first == second  # input order does not matter
    for ds in ("ipl", "f1"):
        assert sum(first[r["id"]] == "test" for r in rows if r["dataset"] == ds) == 15
    assert qs.assign_splits(rows, seed=7) != first  # the seed matters


def test_assign_splits_stable_for_existing_ids_within_group():
    rows = _rows(20)
    assert qs.assign_splits(rows) == qs.assign_splits([dict(r) for r in rows])


def test_jsonl_roundtrip_keeps_field_order(tmp_path):
    row = {"split": "train", "id": "x", "dataset": "ipl", "question": "q?", "gold_sql": None, "gold_answer": None,
           "answerable": False, "unanswerable_type": "subjective", "difficulty": "easy", "source_id": "y"}
    path = tmp_path / "a.jsonl"
    qs.write_jsonl(path, [row])
    assert list(json.loads(path.read_text(encoding="utf-8")).keys())[:2] == ["id", "dataset"]
    assert qs.read_jsonl(path) == [row]


def test_train_examples_only_returns_train(tmp_path):
    rows = [{"id": "a", "dataset": "ipl", "split": "train"}, {"id": "b", "dataset": "ipl", "split": "test"},
            {"id": "c", "dataset": "f1", "split": "train"}]
    for name in qs.SET_FILES:
        qs.write_jsonl(tmp_path / qs.SET_FILES[name], [dict(r, id=f"{name}-{r['id']}") for r in rows])
    pool = qs.train_examples(questions_dir=tmp_path)
    assert {r["split"] for r in pool} == {"train"} and len(pool) == 2 * len(qs.SET_FILES)
    assert sorted(r["id"] for r in qs.train_examples("f1", questions_dir=tmp_path)) == sorted(
        f"{name}-c" for name in qs.SET_FILES)


def test_real_question_files_never_leak_test_into_training_pool():
    """The committed sets: the training pool holds no test ids, including via Hinglish."""
    if not (qs.QUESTIONS_DIR / qs.SET_FILES["csv"]).exists():
        pytest.skip("question sets not built")
    test_ids = {r["id"] for name in qs.SET_FILES for r in qs.load_question_set(name, "test")}
    pool = qs.train_examples()
    assert not test_ids & {r["id"] for r in pool}
    assert not test_ids & {r.get("source_id") for r in pool}


def test_load_heldout_rejects_other_splits(tmp_path):
    rows = [{"id": "human-001", "split": "heldout"}, {"id": "human-002", "split": "train"}]
    qs.write_jsonl(tmp_path / qs.HELDOUT_FILE, rows[:1])
    assert [r["id"] for r in qs.load_heldout(tmp_path)] == ["human-001"]
    qs.write_jsonl(tmp_path / qs.HELDOUT_FILE, rows)
    with pytest.raises(AssertionError, match="human-002"):
        qs.load_heldout(tmp_path)


def test_real_heldout_is_never_in_training_pool():
    heldout_ids = {r["id"] for r in qs.load_heldout()}
    assert len(heldout_ids) == 30
    assert not heldout_ids & {r["id"] for r in qs.train_examples()}


@pytest.mark.parametrize("expected, actual, ordered, ok", [
    ([[4.09]], [[4.0864]], False, True),                      # unrounded AVG, gold rounded to 2 places
    ([["SP", 41746]], [[41746, "SP", 3]], False, True),       # extra column, different column order
    ([[4]], [[4.09]], False, False),                          # integer gold is not rounded towards
    ([["a", 1], ["b", 2]], [["b", 2], ["a", 1]], True, False),  # order matters with ORDER BY
    ([["a", 1], ["b", 2]], [["b", 2], ["a", 1]], False, True),
    ([["a", 1]], [["a", 1], ["b", 2]], False, False),         # row count must match
    ([["a", 1]], [["a"]], False, False),                      # gold column missing
    ([], [], False, True),
])
def test_answers_match_relaxed(expected, actual, ordered, ok):
    assert qs.answers_match_relaxed(expected, actual, ordered) is ok


def test_answers_match_relaxed_keeps_rows_aligned():
    # Each column matches as a multiset, but the rows pair values differently.
    assert not qs.answers_match_relaxed([["a", 1], ["b", 2]], [["a", 2], ["b", 1]], ordered=False)


@pytest.mark.parametrize("sql, ranking", [
    ("SELECT a, COUNT(*) AS n FROM t GROUP BY a ORDER BY n DESC LIMIT 5", True),
    ("SELECT a, COUNT(*) AS n FROM t GROUP BY a ORDER BY n DESC LIMIT 1", False),  # one row: order irrelevant
    ("SELECT a, COUNT(*) AS n FROM t GROUP BY a ORDER BY a", False),              # presentation order only
    ("SELECT a FROM t LIMIT 5", False),
    ("SELECT a FROM (SELECT a FROM t ORDER BY a LIMIT 5)", False),                 # inner ORDER BY ignored
    ("not sql at all (", False),
])
def test_is_ranking(sql, ranking):
    assert qs.is_ranking(sql) is ranking


@pytest.mark.parametrize("sql, keys", [
    ("SELECT a, COUNT(*) AS n FROM t GROUP BY a", [0]),
    ("SELECT ROUND(AVG(x), 2) FROM t", []),
    ("SELECT a, b, total FROM (SELECT a, b, SUM(x) AS total FROM t GROUP BY a, b)", [0, 1]),
    ("WITH s AS (SELECT a, SUM(x) AS p1, SUM(y) AS p2 FROM t GROUP BY a) SELECT a, ROUND(p2 / p1, 1) AS g FROM s", [0]),
    ("SELECT name, points FROM standings", [0, 1]),                               # stored values are keys
])
def test_key_columns(sql, keys):
    assert qs.key_columns(sql) == keys


@pytest.mark.parametrize("expected, actual, keys, ok", [
    ([["Mumbai Indians", 160]], [["Mumbai Indians"]], [0], True),               # count left out
    ([["Mumbai Indians", 160]], [["Chennai Super Kings"]], [0], False),         # wrong key
    ([["Mumbai Indians", 160]], [["Mumbai Indians", 999]], [0], False),         # wrong count present
    ([["Mumbai Indians", 160]], [["Mumbai Indians"]], None, False),             # no key info
    ([["Mumbai Indians", 160]], [["Mumbai Indians"]], [0, 1], False),           # all columns are keys
    ([[2002, -18604468]], [[2002, 71453399, -18604468]], [0], True),           # rule 1: extra column is fine
    ([[2002, -18604468]], [[2002, -18604468.33]], [0], False),                # integer gold is not rounded towards
])
def test_answers_match_relaxed_key_columns(expected, actual, keys, ok):
    assert qs.answers_match_relaxed(expected, actual, ordered=False, keys=keys) is ok
