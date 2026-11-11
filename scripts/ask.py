"""Ask a question about a CSV file from the command line.

Usage:
    python scripts/ask.py data/sample/sales.csv "Which city had the highest revenue?"
    python scripts/ask.py data/sample/sales.csv "har product ki total quantity kitni biki?"
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

from datacopilot.pipeline.executor import load_csv  # noqa: E402
from datacopilot.pipeline.profiler import get_schema  # noqa: E402
from datacopilot.pipeline.runner import SQLAttemptError, answer  # noqa: E402
from datacopilot.pipeline.sql_generator import generate_sql  # noqa: E402


def ask(
    csv_path: str, question: str, generate: Callable[..., str] | None = None
) -> tuple[str, pd.DataFrame]:
    """Load `csv_path` into an in-memory DuckDB, answer `question`, return (sql, result)."""
    conn = duckdb.connect()
    try:
        load_csv(conn, csv_path)
        return answer(conn, question, get_schema(conn), generate or generate_sql)
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
    except SQLAttemptError as err:
        print(f"Error: {err}\n{err.sql}", file=sys.stderr)
        return 1
    except duckdb.Error as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    print("SQL:")
    print(sql)
    print("\nResult:")
    print(result.to_string(index=False) if not result.empty else "(no rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
