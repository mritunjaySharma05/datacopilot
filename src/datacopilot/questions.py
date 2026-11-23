"""Evaluation question sets: execute gold SQL, normalize answers, assign splits, load safely.

Question rows (JSONL) have: id, dataset, question, gold_sql, gold_answer, answerable,
unanswerable_type, difficulty, split. Rows of both Hinglish sets also have source_id.

Test questions must never be used for prompt examples or training: use
`train_examples()`, which only ever returns train-split rows. The human held-out set
(HELDOUT_FILE, split "heldout") is evaluation-only and is never part of SET_FILES.
"""
import datetime as dt
import decimal
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import duckdb
import sqlglot
from sqlglot import exp

from datacopilot.pipeline.executor import connect_sqlite, load_csv
from datacopilot.pipeline.validator import is_read_only

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS_DIR = ROOT / "data" / "questions"
PROCESSED_DIR = ROOT / "data" / "processed"

DATASETS = ("olist", "ipl", "f1", "crop")
SET_FILES = {
    "csv": "csv_questions.jsonl",
    "hinglish": "hinglish_questions.jsonl",
    "unanswerable": "unanswerable_questions.jsonl",
    "hinglish_unanswerable": "hinglish_unanswerable.jsonl",
}
# Human-reviewed held-out questions: evaluation only, never training or prompt examples.
HELDOUT_FILE = "human_heldout.jsonl"
HELDOUT_SPLIT = "heldout"
LANGUAGES = ("en", "hinglish")
DIFFICULTIES = ("easy", "medium", "hard")
UNANSWERABLE_TYPES = ("missing_column", "out_of_range", "external_knowledge", "subjective", "write_request")
FIELDS = ("id", "dataset", "question", "gold_sql", "gold_answer", "answerable", "unanswerable_type",
          "difficulty", "split")
SPLIT_SEED = 42
TEST_SHARE = 0.30
MAX_ANSWER_ROWS = 25
FLOAT_DIGITS = 4
# SQL dialect each dataset's gold SQL is written in.
DIALECT = {"olist": "sqlite", "ipl": "duckdb", "f1": "duckdb", "crop": "duckdb"}
# Processed CSVs loaded into DuckDB for each non-Olist dataset (table = file stem).
CSV_GLOB = {"ipl": "ipl_*.csv", "f1": "f1_*.csv", "crop": "crop_production.csv"}


def connect(dataset: str, processed_dir: Path = PROCESSED_DIR, check_same_thread: bool = True):
    """Read-only connection with the dataset's processed tables.

    `check_same_thread=False` (SQLite only) is for servers that share one connection across
    threads behind a lock; DuckDB connections have no such check.
    """
    if dataset == "olist":
        return connect_sqlite(processed_dir / "olist.sqlite", timeout_s=60, check_same_thread=check_same_thread)
    conn = duckdb.connect()
    paths = sorted(processed_dir.glob(CSV_GLOB[dataset]))
    if not paths:
        raise FileNotFoundError(f"No processed files for {dataset} in {processed_dir}")
    for path in paths:
        load_csv(conn, path)
    return conn


def normalize_value(value: object) -> object:
    """Make one result value JSON-friendly and stable (floats rounded, dates ISO, NaN -> None)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, (float, decimal.Decimal)):
        f = float(value)
        if math.isnan(f):
            return None
        r = round(f, FLOAT_DIGITS)
        return int(r) if r.is_integer() and abs(r) < 2**53 else r
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if hasattr(value, "item"):  # numpy scalar
        return normalize_value(value.item())
    return str(value)


def run_gold(conn, sql: str, dataset: str) -> list[list]:
    """Execute read-only gold SQL and return normalized rows (list of lists)."""
    if not is_read_only(sql, DIALECT[dataset]):
        raise ValueError("gold_sql must be a single read-only SELECT")
    return [[normalize_value(v) for v in row] for row in conn.execute(sql).fetchall()]


def _values_match(a: object, b: object) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=10 ** -FLOAT_DIGITS)
    return a == b


def _row_key(row: list) -> str:
    return json.dumps([round(v, 2) if isinstance(v, float) else v for v in row], default=str)


def answers_match(expected: list[list], actual: list[list], ordered: bool) -> bool:
    """Compare two results; floats with a small tolerance, rows in order only if `ordered`."""
    if len(expected) != len(actual):
        return False
    if not ordered:
        expected, actual = sorted(expected, key=_row_key), sorted(actual, key=_row_key)
    return all(len(e) == len(a) and all(_values_match(x, y) for x, y in zip(e, a))
               for e, a in zip(expected, actual))


def is_ordered(sql: str) -> bool:
    """True if the outermost query has ORDER BY (so row order is part of the answer)."""
    return "ORDER BY" in sql.upper().rsplit(")", 1)[-1]


def assign_splits(rows: list[dict], seed: int = SPLIT_SEED, test_share: float = TEST_SHARE) -> dict[str, str]:
    """Seeded stratified split: within each (dataset, answerable, unanswerable_type) group,
    round(test_share * n) rows go to test. Same ids + seed -> same split, always."""
    groups: dict[tuple, list[str]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["answerable"], row["unanswerable_type"])].append(row["id"])
    splits = {}
    for key, ids in sorted(groups.items(), key=lambda kv: str(kv[0])):
        ids = sorted(ids)
        random.Random(f"{seed}|{key}").shuffle(ids)
        n_test = round(test_share * len(ids))
        for i, qid in enumerate(ids):
            splits[qid] = "test" if i < n_test else "train"
    return splits


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file (blank lines ignored)."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write rows as JSONL with a stable key order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            ordered = {k: row[k] for k in FIELDS if k in row}
            ordered.update({k: v for k, v in row.items() if k not in ordered})
            f.write(json.dumps(ordered, ensure_ascii=False) + "\n")


def load_question_set(name: str, split: str | None = None, questions_dir: Path = QUESTIONS_DIR) -> list[dict]:
    """Load one set ('csv', 'hinglish', 'unanswerable'), optionally filtered by split."""
    rows = read_jsonl(questions_dir / SET_FILES[name])
    return [r for r in rows if split is None or r["split"] == split]


def train_examples(dataset: str | None = None, questions_dir: Path = QUESTIONS_DIR) -> list[dict]:
    """Rows that may be used as prompt examples or training data: train split ONLY."""
    rows = [r for name in SET_FILES for r in load_question_set(name, "train", questions_dir)]
    leaked = [r["id"] for r in rows if r["split"] != "train"]
    assert not leaked, f"test rows in training pool: {leaked}"
    return [r for r in rows if dataset is None or r["dataset"] == dataset]


def load_heldout(questions_dir: Path = QUESTIONS_DIR) -> list[dict]:
    """The human held-out questions (evaluation only; every row has split "heldout")."""
    rows = read_jsonl(questions_dir / HELDOUT_FILE)
    wrong = [r["id"] for r in rows if r["split"] != HELDOUT_SPLIT]
    assert not wrong, f"held-out rows with another split: {wrong}"
    return rows


def _decimals(column: list) -> int | None:
    """Most decimal places among the float values of a gold column (None if it has no floats)."""
    places = [len(repr(v).split(".")[1]) for v in column
              if isinstance(v, float) and "." in repr(v) and "e" not in repr(v)]
    return max(places) if places else None


def _round_column(column: list, decimals: int | None) -> list:
    if decimals is None:
        return column
    return [round(float(v), decimals) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
            for v in column]


def _same_values(a: list, b: list) -> bool:
    """Same multiset of values (floats with tolerance)."""
    a, b = sorted(a, key=lambda v: _row_key([v])), sorted(b, key=lambda v: _row_key([v]))
    return len(a) == len(b) and all(_values_match(x, y) for x, y in zip(a, b))


def _outer_select(sql: str, dialect: str | None = None):
    """Outermost SELECT of a query (None if it cannot be parsed or is not a SELECT)."""
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except sqlglot.errors.ParseError:
        return None
    return tree if isinstance(tree, exp.Select) else None


def is_ranking(sql: str, dialect: str | None = None) -> bool:
    """True if the outer query has ORDER BY and LIMIT > 1 (a ranking / top-N question).

    Relaxed matching only checks row order for these; strict matching uses is_ordered.
    """
    select = _outer_select(sql, dialect)
    if select is None or select.args.get("order") is None or select.args.get("limit") is None:
        return False
    try:
        return int(select.args["limit"].expression.name) > 1
    except (AttributeError, ValueError):
        return False


def key_columns(sql: str, dialect: str | None = None) -> list[int]:
    """Positions of the outer SELECT columns that are not aggregates.

    A column is an aggregate if its expression contains an aggregate function (COUNT, SUM,
    AVG, ...) or refers to an alias given to an aggregate anywhere in the query (e.g. a
    COUNT(*) AS sixes computed in a subquery, or growth computed from such aliases). Unparseable SQL: every column is a key.
    """
    select = _outer_select(sql, dialect)
    if select is None:
        return []
    agg_aliases = {a.alias.lower() for a in select.find_all(exp.Alias) if a.this.find(exp.AggFunc)}
    keys = []
    for i, col in enumerate(select.expressions):
        inner = col.this if isinstance(col, exp.Alias) else col
        if inner.find(exp.AggFunc):
            continue
        if any(c.name.lower() in agg_aliases for c in inner.find_all(exp.Column)):
            continue
        keys.append(i)
    return keys


def _match_columns(gold_cols: list[list], pred_cols: list[list]) -> list[list] | None:
    """For each gold column, the (rounded) predicted column with the same values; None if any is missing."""
    chosen, used = [], set()
    for col in gold_cols:
        d = _decimals(col)
        match = next((k for k, p in enumerate(pred_cols)
                      if k not in used and _same_values(col, _round_column(p, d))), None)
        if match is None:
            return None
        used.add(match)
        chosen.append(_round_column(pred_cols[match], d))
    return chosen


def answers_match_relaxed(expected: list[list], actual: list[list], ordered: bool,
                          keys: list[int] | None = None) -> bool:
    """Lenient match (secondary metric; strict answers_match stays primary).

    1. Every gold column appears among the predicted columns: extra predicted columns are
       allowed, column order may differ, predicted numbers are rounded to the gold column's
       precision (so AVG without ROUND still matches).
    2. Otherwise, if `keys` (gold column positions that are not aggregates, see key_columns)
       is a non-empty proper subset: the prediction has exactly the key columns, i.e. it
       names the right rows but leaves out the count / aggregate. A prediction that has an
       extra column (e.g. a wrong count) is not accepted by this rule.
    Rows must match one-to-one, in order only if `ordered` (callers pass is_ranking(gold_sql)).
    """
    if len(expected) != len(actual):
        return False
    if not expected:
        return True
    gold_cols = [list(c) for c in zip(*expected)]
    pred_cols = [list(c) for c in zip(*actual)]
    chosen = _match_columns(gold_cols, pred_cols)
    if chosen is not None:
        return answers_match(expected, [list(r) for r in zip(*chosen)], ordered)
    if not keys or len(keys) >= len(gold_cols) or len(pred_cols) != len(keys):
        return False
    key_gold = [gold_cols[i] for i in keys]
    chosen = _match_columns(key_gold, pred_cols)
    if chosen is None:
        return False
    return answers_match([list(r) for r in zip(*key_gold)], [list(r) for r in zip(*chosen)], ordered)
