"""Ask a question about a CSV file from the command line.

Usage:
    python scripts/ask.py data/raw/sales.csv "Which city had the highest revenue?"
"""
import argparse
import sys
from pathlib import Path
from typing import Callable

import duckdb
import pandas as pd
import requests

# Allow running from a fresh checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datacopilot.pipeline.executor import load_csv, run_query  # noqa: E402
from datacopilot.pipeline.profiler import get_schema  # noqa: E402
from datacopilot.pipeline.sql_generator import generate_sql  # noqa: E402
from datacopilot.pipeline.validator import validate  # noqa: E402


def get_valid_sql(
    question: str, schema: dict, generate: Callable[..., str] | None = None
) -> str:
    """Generate SQL and validate it, retrying once with the validator's error message.

    `generate` defaults to the Ollama-backed generate_sql. Raises ValueError if the
    second attempt is still invalid.
    """
    generate = generate or generate_sql
    sql = generate(question, schema)
    problems = validate(sql, schema)
    if problems:
        sql = generate(question, schema, feedback=" ".join(problems))
        problems = validate(sql, schema)
    if problems:
        raise ValueError(f"Generated SQL failed validation: {' '.join(problems)}\n{sql}")
    return sql


def ask(
    csv_path: str, question: str, generate: Callable[..., str] | None = None
) -> tuple[str, pd.DataFrame]:
    """Load `csv_path` into an in-memory DuckDB, answer `question`, return (sql, result)."""
    conn = duckdb.connect()
    try:
        load_csv(conn, csv_path)
        schema = get_schema(conn)
        sql = get_valid_sql(question, schema, generate)
        return sql, run_query(conn, sql)
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    parser = argparse.ArgumentParser(description="Ask a question about a CSV file.")
    parser.add_argument("csv_path", help="Path to a CSV file")
    parser.add_argument("question", help="Question in English or Hinglish")
    args = parser.parse_args(argv)

    if not Path(args.csv_path).is_file():
        print(f"File not found: {args.csv_path}", file=sys.stderr)
        return 1
    try:
        sql, result = ask(args.csv_path, args.question)
    except requests.ConnectionError:
        print("Could not reach Ollama. Is it running? (check OLLAMA_HOST in .env)", file=sys.stderr)
        return 1
    except (ValueError, duckdb.Error) as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    print("SQL:")
    print(sql)
    print("\nResult:")
    print(result.to_string(index=False) if not result.empty else "(no rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
