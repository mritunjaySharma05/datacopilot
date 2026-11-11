"""Tests for scripts/ask.py with the LLM call replaced by a fake."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASK_PATH = ROOT / "scripts" / "ask.py"
SAMPLE_CSV = ROOT / "data" / "sample" / "sales.csv"
spec = importlib.util.spec_from_file_location("ask", ASK_PATH)
ask_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ask_module)


def test_ask_end_to_end(sales_csv, fake_llm):
    gen = fake_llm("SELECT city, SUM(revenue) AS total FROM sales GROUP BY city ORDER BY total DESC")
    sql, result = ask_module.ask(str(sales_csv), "top city?", gen)
    assert sql.startswith("SELECT city")
    assert result.iloc[0]["city"] == "Delhi"
    assert gen.calls[0]["dialect"] == "duckdb"


def test_main_prints_sql_and_result(sales_csv, monkeypatch, capsys):
    monkeypatch.setattr(ask_module, "generate_sql", lambda q, s, **kw: "SELECT COUNT(*) AS n FROM sales")
    assert ask_module.main([str(sales_csv), "how many rows?"]) == 0
    out = capsys.readouterr().out
    assert "SELECT COUNT(*) AS n FROM sales" in out
    assert "4" in out


def test_main_reports_validation_failure(sales_csv, monkeypatch, capsys):
    monkeypatch.setattr(ask_module, "generate_sql", lambda q, s, **kw: "DROP TABLE sales")
    assert ask_module.main([str(sales_csv), "delete it"]) == 1
    err = capsys.readouterr().err
    assert "Validation failed" in err
    assert "after one retry" in err


def test_main_ollama_unreachable(sales_csv, monkeypatch, capsys):
    def down(q, s, **kw):
        raise ask_module.requests.ConnectionError()

    monkeypatch.setattr(ask_module, "generate_sql", down)
    assert ask_module.main([str(sales_csv), "q"]) == 1
    assert "Could not reach Ollama" in capsys.readouterr().err


def test_main_on_bundled_sample(monkeypatch, capsys):
    monkeypatch.setattr(ask_module, "generate_sql", lambda q, s, **kw: "SELECT COUNT(*) AS n FROM sales")
    assert ask_module.main([str(SAMPLE_CSV), "how many rows?"]) == 0
    assert "50" in capsys.readouterr().out


def test_main_missing_file(capsys):
    assert ask_module.main(["nope.csv", "q"]) == 1
    assert "File not found" in capsys.readouterr().err
