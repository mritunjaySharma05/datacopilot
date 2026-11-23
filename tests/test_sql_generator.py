import pytest
import requests

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


# --- generate_sql (fake provider; HTTP providers are tested in test_llm.py) ---

class FakeProvider:
    """Records prompts and returns canned replies."""

    name, model = "fake", "fake-model"

    def __init__(self, reply="SELECT 1", error=None):
        self.reply, self.error, self.prompts = reply, error, []

    def complete(self, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.reply


def test_generate_sql_returns_extracted_sql(schema):
    fake = FakeProvider("```sql\nSELECT SUM(revenue) FROM sales\n```\nExplanation...")
    assert generate_sql("Total revenue?", schema, provider=fake) == "SELECT SUM(revenue) FROM sales"


def test_generate_sql_sends_the_shared_prompt_with_feedback(schema):
    fake = FakeProvider()
    generate_sql("q", schema, feedback="Unknown tables: orders.", dialect="sqlite", provider=fake)
    assert fake.prompts == [build_prompt("q", schema, "Unknown tables: orders.", None, "sqlite")]


def test_generate_sql_propagates_provider_errors(schema):
    fake = FakeProvider(error=requests.ConnectionError("down"))
    with pytest.raises(requests.ConnectionError):
        generate_sql("q", schema, provider=fake)


def test_generate_sql_defaults_to_env_provider(monkeypatch, schema):
    fake = FakeProvider("SELECT 2")
    monkeypatch.setattr(sql_generator, "get_provider", lambda: fake)
    assert generate_sql("q", schema) == "SELECT 2" and len(fake.prompts) == 1


def test_build_prompt_default_dialect_is_duckdb(schema):
    assert build_prompt("q", schema).startswith("You write DuckDB SQL")


def test_build_prompt_sqlite_dialect_and_evidence(schema):
    prompt = build_prompt("q", schema, evidence="rate = free / enrollment", dialect="sqlite")
    assert prompt.startswith("You write SQLite SQL")
    assert "Hint: rate = free / enrollment" in prompt


def test_build_prompt_without_evidence_has_no_hint(schema):
    assert "Hint:" not in build_prompt("q", schema)


def test_generate_sql_passes_evidence_and_dialect(schema):
    fake = FakeProvider()
    generate_sql("q", schema, evidence="hint here", dialect="sqlite", provider=fake)
    assert "SQLite" in fake.prompts[0] and "Hint: hint here" in fake.prompts[0]
