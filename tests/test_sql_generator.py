from unittest.mock import MagicMock

import pytest

from datacopilot.pipeline import sql_generator
from datacopilot.pipeline.sql_generator import build_prompt, extract_sql, generate_sql


# --- extract_sql ---

def test_extract_sql_from_sql_code_block():
    text = "Here you go:\n```sql\nSELECT city FROM sales;\n```\nThis lists cities."
    assert extract_sql(text) == "SELECT city FROM sales"


def test_extract_sql_from_plain_code_block():
    assert extract_sql("```\nSELECT 1\n```") == "SELECT 1"


def test_extract_sql_from_prose_without_block():
    text = "Sure! SELECT city, SUM(revenue) FROM sales GROUP BY city; This sums revenue."
    assert extract_sql(text) == "SELECT city, SUM(revenue) FROM sales GROUP BY city"


def test_extract_sql_stops_at_blank_line():
    text = "SELECT *\nFROM sales\n\nThis query returns every row."
    assert extract_sql(text) == "SELECT *\nFROM sales"


def test_extract_sql_keeps_with_clause():
    text = "```sql\nWITH t AS (SELECT 1 AS x) SELECT x FROM t\n```"
    assert extract_sql(text).startswith("WITH t AS")


def test_extract_sql_returns_empty_when_no_sql():
    assert extract_sql("I cannot answer that.") == ""


# --- build_prompt ---

def test_build_prompt_contains_schema_and_question(schema):
    prompt = build_prompt("Total revenue?", schema)
    assert "Table sales:" in prompt
    assert "Question: Total revenue?" in prompt
    assert "rejected" not in prompt


def test_build_prompt_includes_feedback(schema):
    prompt = build_prompt("Total revenue?", schema, feedback="Unknown columns: price.")
    assert "rejected: Unknown columns: price." in prompt


# --- _call_ollama (HTTP mocked) ---

def test_call_ollama_uses_env_and_temperature_zero(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama.test:1234/")
    monkeypatch.setenv("OLLAMA_MODEL", "test-model")
    response = MagicMock()
    response.json.return_value = {"response": "SELECT 1"}
    post = MagicMock(return_value=response)
    monkeypatch.setattr(sql_generator.requests, "post", post)

    assert sql_generator._call_ollama("hello") == "SELECT 1"

    url = post.call_args.args[0]
    body = post.call_args.kwargs["json"]
    assert url == "http://ollama.test:1234/api/generate"
    assert body["model"] == "test-model"
    assert body["options"]["temperature"] == 0
    assert body["stream"] is False


# --- generate_sql (Ollama mocked) ---

def test_generate_sql_returns_extracted_sql(monkeypatch, schema):
    monkeypatch.setattr(
        sql_generator, "_call_ollama",
        lambda prompt: "```sql\nSELECT SUM(revenue) FROM sales\n```\nExplanation...",
    )
    assert generate_sql("Total revenue?", schema) == "SELECT SUM(revenue) FROM sales"


def test_generate_sql_passes_feedback_into_prompt(monkeypatch, schema):
    seen = {}

    def fake(prompt):
        seen["prompt"] = prompt
        return "SELECT 1"

    monkeypatch.setattr(sql_generator, "_call_ollama", fake)
    generate_sql("q", schema, feedback="Unknown tables: orders.")
    assert "Unknown tables: orders." in seen["prompt"]


def test_generate_sql_propagates_http_errors(monkeypatch, schema):
    def boom(prompt):
        raise sql_generator.requests.ConnectionError("down")

    monkeypatch.setattr(sql_generator, "_call_ollama", boom)
    with pytest.raises(sql_generator.requests.ConnectionError):
        generate_sql("q", schema)


def test_build_prompt_default_dialect_is_duckdb(schema):
    assert build_prompt("q", schema).startswith("You write DuckDB SQL")


def test_build_prompt_sqlite_dialect_and_evidence(schema):
    prompt = build_prompt("q", schema, evidence="rate = free / enrollment", dialect="sqlite")
    assert prompt.startswith("You write SQLite SQL")
    assert "Hint: rate = free / enrollment" in prompt


def test_build_prompt_without_evidence_has_no_hint(schema):
    assert "Hint:" not in build_prompt("q", schema)


def test_call_ollama_sets_context_window(monkeypatch):
    monkeypatch.setenv("OLLAMA_NUM_CTX", "4096")
    response = MagicMock()
    response.json.return_value = {"response": "SELECT 1"}
    post = MagicMock(return_value=response)
    monkeypatch.setattr(sql_generator.requests, "post", post)
    sql_generator._call_ollama("hello")
    assert post.call_args.kwargs["json"]["options"]["num_ctx"] == 4096


def test_generate_sql_passes_evidence_and_dialect(monkeypatch, schema):
    seen = {}
    monkeypatch.setattr(sql_generator, "_call_ollama", lambda p: seen.setdefault("p", p) and "SELECT 1")
    generate_sql("q", schema, evidence="hint here", dialect="sqlite")
    assert "SQLite" in seen["p"] and "Hint: hint here" in seen["p"]
