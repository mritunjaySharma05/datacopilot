"""Runner: generate SQL, validate it, execute it, and retry once with the error."""
import sqlite3
from typing import Callable

import duckdb
import pandas as pd

from datacopilot.pipeline.executor import MAX_ROWS, Connection, run_query
from datacopilot.pipeline.sql_generator import generate_sql
from datacopilot.pipeline.validator import validate

DB_ERRORS = (duckdb.Error, sqlite3.Error)


class SQLAttemptError(Exception):
    """Generated SQL was rejected by the validator or failed when executed."""

    def __init__(self, message: str, sql: str = "") -> None:
        super().__init__(message)
        self.sql = sql


def try_sql(
    conn: Connection,
    sql: str,
    schema: dict,
    dialect: str = "duckdb",
    limit: int | None = MAX_ROWS,
) -> pd.DataFrame:
    """Validate then execute `sql`; raise SQLAttemptError with a message the LLM can act on."""
    problems = validate(sql, schema, dialect)
    if problems:
        raise SQLAttemptError(f"Validation failed: {' '.join(problems)}", sql)
    try:
        return run_query(conn, sql, limit, dialect)
    except DB_ERRORS as err:
        raise SQLAttemptError(f"Execution failed: {err}", sql) from err


def run_unvalidated(
    conn: Connection, sql: str, dialect: str = "duckdb", limit: int | None = MAX_ROWS
) -> pd.DataFrame:
    """Baseline: execute without the schema check (the read-only guard in run_query stays)."""
    try:
        return run_query(conn, sql, limit, dialect)
    except (ValueError, *DB_ERRORS) as err:
        raise SQLAttemptError(f"Execution failed: {err}", sql) from err


def answer(
    conn: Connection,
    question: str,
    schema: dict,
    generate: Callable[..., str] | None = None,
    *,
    evidence: str | None = None,
    dialect: str = "duckdb",
    use_validator: bool = True,
    limit: int | None = MAX_ROWS,
) -> tuple[str, pd.DataFrame]:
    """Answer `question` with SQL; return (sql, result).

    With the validator: generate, validate, execute; on a validation or execution error,
    retry once with that error message. Without it (baseline): one attempt, no schema
    check, no retry. `generate` defaults to the Ollama-backed generate_sql.
    Raises SQLAttemptError (with `.sql` set) if no attempt succeeds.
    """
    generate = generate or generate_sql

    def ask_llm(feedback: str | None = None) -> str:
        return generate(question, schema, feedback=feedback, evidence=evidence, dialect=dialect)

    sql = ask_llm()
    if not use_validator:
        return sql, run_unvalidated(conn, sql, dialect, limit)

    try:
        return sql, try_sql(conn, sql, schema, dialect, limit)
    except SQLAttemptError as err:
        feedback = str(err)

    sql = ask_llm(feedback)
    try:
        return sql, try_sql(conn, sql, schema, dialect, limit)
    except SQLAttemptError as err:
        raise SQLAttemptError(f"{err} (after one retry)", sql) from err
