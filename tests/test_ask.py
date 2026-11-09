"""Tests for scripts/ask.py with the LLM call replaced by a fake."""
import importlib.util
from pathlib import Path

import pytest

from datacopilot.pipeline.profiler import get_schema

ROOT = Path(__file__).resolve().parents[1]
ASK_PATH = ROOT / "scripts" / "ask.py"
SAMPLE_CSV = ROOT / "data" / "sample" / "sales.csv"
spec = importlib.util.spec_from_file_location("ask", ASK_PATH)
ask_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ask_module)


def fake_generator(*replies):
    """Return a fake generate_sql that yields `replies` in order and records calls."""
    calls = []

    def generate(question, schema, feedback=None):
        calls.append(feedback)
        return replies[len(calls) - 1]

    generate.calls = calls
    return generate


# --- try_sql ---

def test_try_sql_returns_result(conn):
    df = ask_module.try_sql(conn, "SELECT COUNT(*) AS n FROM sales", get_schema(conn))
    assert df.iloc[0]["n"] == 4


def test_try_sql_reports_validation_error(conn):
    with pytest.raises(ask_module.SQLAttemptError, match="Validation failed: Unknown columns: price."):
        ask_module.try_sql(conn, "SELECT price FROM sales", get_schema(conn))


def test_try_sql_reports_execution_error(conn):
    # Schema-valid but DuckDB can't SUM a VARCHAR column.
    with pytest.raises(ask_module.SQLAttemptError, match="Execution failed:"):
        ask_module.try_sql(conn, "SELECT SUM(city) FROM sales", get_schema(conn))


# --- answer (generate -> validate -> execute, one retry) ---

def test_answer_first_try(conn):
    gen = fake_generator("SELECT city FROM sales")
    sql, df = ask_module.answer(conn, "q", get_schema(conn), gen)
    assert sql == "SELECT city FROM sales"
    assert len(df) == 4
    assert gen.calls == [None]


def test_answer_retries_with_validation_error(conn):
    gen = fake_generator("SELECT price FROM sales", "SELECT revenue FROM sales")
    sql, _ = ask_module.answer(conn, "q", get_schema(conn), gen)
    assert sql == "SELECT revenue FROM sales"
    assert gen.calls[1] == "Validation failed: Unknown columns: price."


def test_answer_retries_with_execution_error(conn):
    gen = fake_generator("SELECT SUM(city) FROM sales", "SELECT SUM(revenue) AS total FROM sales")
    sql, df = ask_module.answer(conn, "q", get_schema(conn), gen)
    assert sql == "SELECT SUM(revenue) AS total FROM sales"
    assert df.iloc[0]["total"] == 188000
    assert gen.calls[1].startswith("Execution failed:")
    assert "sum" in gen.calls[1].lower()


def test_answer_gives_up_after_one_retry(conn):
    gen = fake_generator("DROP TABLE sales", "SELECT SUM(city) FROM sales", "SELECT 1")
    with pytest.raises(ask_module.SQLAttemptError, match="after one retry"):
        ask_module.answer(conn, "q", get_schema(conn), gen)
    assert len(gen.calls) == 2
    assert conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0] == 4


def test_ask_end_to_end(sales_csv):
    gen = fake_generator("SELECT city, SUM(revenue) AS total FROM sales GROUP BY city ORDER BY total DESC")
    sql, result = ask_module.ask(str(sales_csv), "top city?", gen)
    assert sql.startswith("SELECT city")
    assert result.iloc[0]["city"] == "Delhi"


def test_main_prints_sql_and_result(sales_csv, monkeypatch, capsys):
    monkeypatch.setattr(
        ask_module, "generate_sql", lambda q, s, feedback=None: "SELECT COUNT(*) AS n FROM sales"
    )
    assert ask_module.main([str(sales_csv), "how many rows?"]) == 0
    out = capsys.readouterr().out
    assert "SELECT COUNT(*) AS n FROM sales" in out
    assert "4" in out


def test_main_reports_validation_failure(sales_csv, monkeypatch, capsys):
    monkeypatch.setattr(ask_module, "generate_sql", lambda q, s, feedback=None: "DROP TABLE sales")
    assert ask_module.main([str(sales_csv), "delete it"]) == 1
    err = capsys.readouterr().err
    assert "Validation failed" in err
    assert "after one retry" in err


def test_main_ollama_unreachable(sales_csv, monkeypatch, capsys):
    def down(q, s, feedback=None):
        raise ask_module.requests.ConnectionError()

    monkeypatch.setattr(ask_module, "generate_sql", down)
    assert ask_module.main([str(sales_csv), "q"]) == 1
    assert "Could not reach Ollama" in capsys.readouterr().err


def test_main_on_bundled_sample(monkeypatch, capsys):
    monkeypatch.setattr(
        ask_module, "generate_sql", lambda q, s, feedback=None: "SELECT COUNT(*) AS n FROM sales"
    )
    assert ask_module.main([str(SAMPLE_CSV), "how many rows?"]) == 0
    assert "50" in capsys.readouterr().out


def test_main_missing_file(capsys):
    assert ask_module.main(["nope.csv", "q"]) == 1
    assert "File not found" in capsys.readouterr().err
