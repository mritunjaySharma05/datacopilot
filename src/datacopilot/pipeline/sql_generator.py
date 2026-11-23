"""SQL generator: ask the configured LLM (Ollama or Groq, see llm.py) to turn a question into SQL."""
import re

from dotenv import load_dotenv

from datacopilot.pipeline.llm import LLMProvider, get_provider
from datacopilot.pipeline.profiler import format_schema

load_dotenv()

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


def generate_sql(
    question: str,
    schema: dict,
    feedback: str | None = None,
    evidence: str | None = None,
    dialect: str = "duckdb",
    provider: LLMProvider | None = None,
) -> str:
    """Return one SQL query answering `question` over `schema`.

    `provider` defaults to the one selected by LLM_PROVIDER (llm.get_provider()).
    """
    prompt = build_prompt(question, schema, feedback, evidence, dialect)
    return extract_sql((provider or get_provider()).complete(prompt))
