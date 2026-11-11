import sqlite3

import pytest

from datacopilot.pipeline.executor import connect_sqlite
from datacopilot.pipeline.profiler import get_schema, get_sqlite_schema
from datacopilot.pipeline.runner import SQLAttemptError, answer, run_unvalidated, try_sql


# --- try_sql ---

def test_try_sql_returns_result(conn):
    df = try_sql(conn, "SELECT COUNT(*) AS n FROM sales", get_schema(conn))
    assert df.iloc[0]["n"] == 4


def test_try_sql_reports_validation_error(conn):
    with pytest.raises(SQLAttemptError, match="Validation failed: Unknown columns: price.") as info:
        try_sql(conn, "SELECT price FROM sales", get_schema(conn))
    assert info.value.sql == "SELECT price FROM sales"


def test_try_sql_reports_execution_error(conn):
    # Schema-valid but DuckDB can't SUM a VARCHAR column.
    with pytest.raises(SQLAttemptError, match="Execution failed:"):
        try_sql(conn, "SELECT SUM(city) FROM sales", get_schema(conn))


def test_try_sql_sqlite_backticks(tiny_sqlite):
    con = connect_sqlite(tiny_sqlite)
    df = try_sql(con, "SELECT `Item Name` FROM items WHERE price > 5", get_sqlite_schema(con), "sqlite")
    assert df["Item Name"].tolist() == ["Pen"]


def test_try_sql_reports_sqlite_execution_error(tiny_sqlite):
    con = connect_sqlite(tiny_sqlite)
    with pytest.raises(SQLAttemptError, match="Execution failed:"):
        try_sql(con, "SELECT price FROM items GROUP BY", get_sqlite_schema(con), "sqlite")


# --- run_unvalidated (baseline) ---

def test_run_unvalidated_skips_schema_check_but_still_executes(tiny_sqlite):
    con = connect_sqlite(tiny_sqlite)
    with pytest.raises(SQLAttemptError, match="Execution failed: no such column"):
        run_unvalidated(con, "SELECT colour FROM items", "sqlite")


def test_run_unvalidated_keeps_read_only_guard(tiny_sqlite):
    con = connect_sqlite(tiny_sqlite)
    with pytest.raises(SQLAttemptError, match="read-only"):
        run_unvalidated(con, "DELETE FROM items", "sqlite")


# --- answer (generate -> validate -> execute, one retry) ---

def test_answer_first_try(conn, fake_llm):
    gen = fake_llm("SELECT city FROM sales")
    sql, df = answer(conn, "q", get_schema(conn), gen)
    assert sql == "SELECT city FROM sales"
    assert len(df) == 4
    assert [c["feedback"] for c in gen.calls] == [None]


def test_answer_retries_with_validation_error(conn, fake_llm):
    gen = fake_llm("SELECT price FROM sales", "SELECT revenue FROM sales")
    sql, _ = answer(conn, "q", get_schema(conn), gen)
    assert sql == "SELECT revenue FROM sales"
    assert gen.calls[1]["feedback"] == "Validation failed: Unknown columns: price."


def test_answer_retries_with_execution_error(conn, fake_llm):
    gen = fake_llm("SELECT SUM(city) FROM sales", "SELECT SUM(revenue) AS total FROM sales")
    sql, df = answer(conn, "q", get_schema(conn), gen)
    assert sql == "SELECT SUM(revenue) AS total FROM sales"
    assert df.iloc[0]["total"] == 188000
    assert gen.calls[1]["feedback"].startswith("Execution failed:")


def test_answer_gives_up_after_one_retry(conn, fake_llm):
    gen = fake_llm("DROP TABLE sales", "SELECT SUM(city) FROM sales", "SELECT 1")
    with pytest.raises(SQLAttemptError, match="after one retry") as info:
        answer(conn, "q", get_schema(conn), gen)
    assert info.value.sql == "SELECT SUM(city) FROM sales"
    assert len(gen.calls) == 2
    assert conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0] == 4


def test_answer_passes_evidence_and_dialect(tiny_sqlite, fake_llm):
    con = connect_sqlite(tiny_sqlite)
    gen = fake_llm("SELECT COUNT(*) FROM items")
    answer(con, "q", get_sqlite_schema(con), gen, evidence="cheap = price < 5", dialect="sqlite")
    assert gen.calls[0] == {"feedback": None, "evidence": "cheap = price < 5", "dialect": "sqlite"}


def test_answer_without_validator_is_single_attempt(tiny_sqlite, fake_llm):
    con = connect_sqlite(tiny_sqlite)
    gen = fake_llm("SELECT colour FROM items", "SELECT 1")
    with pytest.raises(SQLAttemptError, match="no such column"):
        answer(con, "q", get_sqlite_schema(con), gen, dialect="sqlite", use_validator=False)
    assert len(gen.calls) == 1


def test_answer_without_validator_returns_result(tiny_sqlite, fake_llm):
    con = connect_sqlite(tiny_sqlite)
    gen = fake_llm("SELECT COUNT(*) AS n FROM items")
    _, df = answer(con, "q", {}, gen, dialect="sqlite", use_validator=False)
    assert df.iloc[0]["n"] == 3


def test_answer_limit_none_returns_all_rows(fake_llm):
    con = sqlite3.connect(":memory:")
    gen = fake_llm("WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r WHERE x < 1500) SELECT x FROM r")
    _, df = answer(con, "q", {}, gen, dialect="sqlite", use_validator=False, limit=None)
    assert len(df) == 1500
