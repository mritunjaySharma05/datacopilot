"""Evaluation question sets: execute gold SQL, normalize answers, assign splits, load safely.

Question rows (JSONL) have: id, dataset, question, gold_sql, gold_answer, answerable,
unanswerable_type, difficulty, split. Hinglish rows also have source_id (the English id).

Test questions must never be used for prompt examples or training: use
`train_examples()`, which only ever returns train-split rows.
"""
import datetime as dt
import decimal
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import duckdb

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
}
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


def connect(dataset: str, processed_dir: Path = PROCESSED_DIR):
    """Read-only connection with the dataset's processed tables."""
    if dataset == "olist":
        return connect_sqlite(processed_dir / "olist.sqlite", timeout_s=60)
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
