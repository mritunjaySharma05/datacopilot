"""SQL generator: ask a local Ollama model to turn a question into SQL (DuckDB or SQLite)."""
import os
import re

import requests
from dotenv import load_dotenv

from datacopilot.pipeline.profiler import format_schema

load_dotenv()

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5-coder:7b"
REQUEST_TIMEOUT_S = 120
# Ollama's default context window is small; wide schemas would be silently truncated.
DEFAULT_NUM_CTX = 8192
DIALECT_NAMES = {"duckdb": "DuckDB", "sqlite": "SQLite"}

PROMPT_TEMPLATE = """You write {dialect} SQL for data analysis.
Use only the tables and columns in this schema:

{schema}

Rules:
- Write exactly one read-only SELECT query (WITH ... SELECT is fine).
- Use only the columns listed above.
- Quote column names that contain spaces or special characters.
- Return only the SQL inside a ```sql code block, with no explanation.
{feedback}{hint}
Question: {question}
"""

_CODE_BLOCK = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_SQL_START = re.compile(r"\b(SELECT|WITH)\b", re.IGNORECASE)


def build_prompt(
    question: str,
    schema: dict,
    feedback: str | None = None,
    evidence: str | None = None,
    dialect: str = "duckdb",
) -> str:
    """Build the LLM prompt.

    `feedback` explains why the previous SQL was rejected; `evidence` is a domain hint
    (e.g. BIRD's evidence field); `dialect` is "duckdb" or "sqlite".
    """
    feedback_text = ""
    if feedback:
        feedback_text = f"\nYour previous query was rejected: {feedback}\nFix it.\n"
    hint_text = f"\nHint: {evidence}\n" if evidence else ""
    return PROMPT_TEMPLATE.format(
        dialect=DIALECT_NAMES.get(dialect, dialect),
        schema=format_schema(schema),
        feedback=feedback_text,
        hint=hint_text,
        question=question,
    )


def extract_sql(text: str) -> str:
    """Pull the SQL out of an LLM reply (code block or SQL mixed with prose)."""
    block = _CODE_BLOCK.search(text)
    if block:
        sql = block.group(1)
    else:
        start = _SQL_START.search(text)
        if not start:
            return ""
        sql = text[start.start():]
        # Stop at the first semicolon or blank line (explanation usually follows).
        sql = re.split(r";|\n\s*\n", sql, maxsplit=1)[0]
    return sql.strip().rstrip(";").strip()


def _call_ollama(prompt: str) -> str:
    """Send `prompt` to Ollama with temperature 0 and return the raw reply text."""
    num_ctx = int(os.getenv("OLLAMA_NUM_CTX", DEFAULT_NUM_CTX))
    host = os.getenv("OLLAMA_HOST", DEFAULT_HOST).rstrip("/")
    model = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
    response = requests.post(
        f"{host}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "num_ctx": num_ctx},
        },
        timeout=REQUEST_TIMEOUT_S,
    )
    response.raise_for_status()
    return response.json()["response"]


def generate_sql(
    question: str,
    schema: dict,
    feedback: str | None = None,
    evidence: str | None = None,
    dialect: str = "duckdb",
) -> str:
    """Return one SQL query answering `question` over `schema`."""
    prompt = build_prompt(question, schema, feedback, evidence, dialect)
    return extract_sql(_call_ollama(prompt))
