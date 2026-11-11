"""Evaluate the pipeline on the BIRD dev set (execution accuracy).

A question counts as correct when our SQL returns the same set of rows as the gold
SQL (same rule as BIRD's official evaluation: row order and duplicates are ignored).

Usage:
    python scripts/eval_bird.py                  # first 10 questions, validated pipeline
    python scripts/eval_bird.py --n 50
    python scripts/eval_bird.py --no-validator   # baseline: one attempt, no schema check
"""
import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datacopilot.pipeline.executor import connect_sqlite  # noqa: E402
from datacopilot.pipeline.profiler import get_sqlite_schema  # noqa: E402
from datacopilot.pipeline.runner import DB_ERRORS, SQLAttemptError, answer  # noqa: E402
from datacopilot.pipeline.sql_generator import generate_sql  # noqa: E402

DEFAULT_BIRD_DIR = ROOT / "data" / "benchmarks" / "bird_dev"
DEFAULT_OUT_DIR = ROOT / "data" / "results"
DEFAULT_N = 10
QUERY_TIMEOUT_S = 30  # BIRD's official evaluation also uses a per-query timeout
DIALECT = "sqlite"


def find_dev_json(bird_dir: Path) -> Path:
    """Return the dev.json questions file under `bird_dir` (shallowest match)."""
    matches = [p for p in bird_dir.rglob("dev.json") if "__MACOSX" not in p.parts]
    if not matches:
        raise FileNotFoundError(f"No dev.json under {bird_dir}. Run scripts/download_bird.py first.")
    return min(matches, key=lambda p: len(p.parts))


def find_db(bird_dir: Path, db_id: str) -> Path:
    """Return the path of <db_id>/<db_id>.sqlite under `bird_dir`."""
    for path in bird_dir.rglob(f"{db_id}.sqlite"):
        if "__MACOSX" not in path.parts and path.parent.name == db_id:
            return path
    raise FileNotFoundError(f"Database {db_id} not found under {bird_dir}")


def load_questions(bird_dir: Path, n: int) -> list[dict]:
    """Load the first `n` BIRD dev questions."""
    with open(find_dev_json(bird_dir), encoding="utf-8") as f:
        return json.load(f)[:n]


def _normalize(value: object) -> object:
    """Treat NaN (from pandas) as NULL so it compares equal to SQLite's None."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def to_rows(df: pd.DataFrame) -> list[tuple]:
    """DataFrame -> list of row tuples with NaN mapped back to None."""
    return [tuple(_normalize(v) for v in row) for row in df.itertuples(index=False, name=None)]


def results_match(predicted: list[tuple], gold: list[tuple]) -> bool:
    """BIRD execution accuracy: same set of rows (order and duplicates ignored)."""
    return set(predicted) == set(gold)


def evaluate_question(
    item: dict,
    bird_dir: Path,
    use_validator: bool = True,
    generate: Callable[..., str] | None = None,
    timeout_s: float = QUERY_TIMEOUT_S,
) -> dict:
    """Run one BIRD question through the pipeline and score it against the gold SQL."""
    record = {
        "question_id": item["question_id"],
        "db_id": item["db_id"],
        "difficulty": item.get("difficulty", ""),
        "question": item["question"],
        "evidence": item.get("evidence", ""),
        "gold_sql": item["SQL"],
        "pred_sql": "",
        "correct": False,
        "error": "",
        "seconds": 0.0,
    }
    start = time.monotonic()
    conn = connect_sqlite(find_db(bird_dir, item["db_id"]), timeout_s=timeout_s)
    try:
        try:
            gold = conn.execute(item["SQL"]).fetchall()
        except DB_ERRORS as err:
            record["error"] = f"Gold SQL failed: {err}"
            return record
        try:
            sql, result = answer(
                conn,
                item["question"],
                get_sqlite_schema(conn),
                generate or generate_sql,
                evidence=item.get("evidence") or None,
                dialect=DIALECT,
                use_validator=use_validator,
                limit=None,  # compare full result sets, not the first 1000 rows
            )
        except SQLAttemptError as err:
            record["pred_sql"] = err.sql
            record["error"] = str(err)
            return record
        record["pred_sql"] = sql
        record["correct"] = results_match(to_rows(result), gold)
        return record
    finally:
        conn.close()
        record["seconds"] = round(time.monotonic() - start, 2)


def output_path(out_dir: Path, use_validator: bool, now: datetime | None = None) -> Path:
    """data/results/bird_<mode>_<timestamp>.csv, mode = validated | baseline."""
    mode = "validated" if use_validator else "baseline"
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return out_dir / f"bird_{mode}_{stamp}.csv"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    parser = argparse.ArgumentParser(description="Execution accuracy on the BIRD dev set.")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Number of questions (from the start)")
    parser.add_argument("--no-validator", action="store_true", help="Baseline: no schema check, no retry")
    parser.add_argument("--bird-dir", type=Path, default=DEFAULT_BIRD_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--timeout", type=float, default=QUERY_TIMEOUT_S, help="Per-query timeout (s)")
    args = parser.parse_args(argv)
    use_validator = not args.no_validator

    try:
        questions = load_questions(args.bird_dir, args.n)
    except FileNotFoundError as err:
        print(err, file=sys.stderr)
        return 1

    mode = "validated" if use_validator else "baseline"
    print(f"Mode: {mode} | questions: {len(questions)}")
    records = []
    for i, item in enumerate(questions, 1):
        try:
            record = evaluate_question(item, args.bird_dir, use_validator, timeout_s=args.timeout)
        except requests.ConnectionError:
            print("Could not reach Ollama. Is it running? (check OLLAMA_HOST in .env)", file=sys.stderr)
            return 1
        records.append(record)
        mark = "OK  " if record["correct"] else "FAIL"
        note = f" ({record['error'][:80]})" if record["error"] else ""
        print(f"[{i}/{len(questions)}] {mark} q{record['question_id']} {record['db_id']}{note}")

    out = output_path(args.out_dir, use_validator)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out, index=False)

    correct = sum(r["correct"] for r in records)
    accuracy = correct / len(records) if records else 0.0
    print(f"\nExecution accuracy ({mode}): {correct}/{len(records)} = {accuracy:.1%}")
    print(f"Saved per-question results to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
