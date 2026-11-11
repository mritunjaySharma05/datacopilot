"""Profiler: read the schema of a DuckDB or SQLite database so the LLM knows what exists."""
import sqlite3

import duckdb

SAMPLES_PER_COLUMN = 3
MAX_SAMPLE_CHARS = 40


def get_schema(conn: duckdb.DuckDBPyConnection) -> dict[str, list[dict]]:
    """Return {table: [{"name", "type", "samples"}, ...]} for every table in `main`.

    `samples` holds up to SAMPLES_PER_COLUMN distinct non-null values per column.
    """
    rows = conn.execute(
        """
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'main'
        ORDER BY table_name, ordinal_position
        """
    ).fetchall()

    schema: dict[str, list[dict]] = {}
    for table, column, dtype in rows:
        samples = conn.execute(
            f'SELECT DISTINCT "{column}" FROM "{table}" '
            f'WHERE "{column}" IS NOT NULL LIMIT {SAMPLES_PER_COLUMN}'
        ).fetchall()
        schema.setdefault(table, []).append(
            {"name": column, "type": dtype, "samples": [r[0] for r in samples]}
        )
    return schema


def _quote(name: str) -> str:
    """Quote an SQL identifier, escaping embedded double quotes."""
    return '"' + name.replace('"', '""') + '"'


def get_sqlite_schema(conn: sqlite3.Connection) -> dict[str, list[dict]]:
    """Same shape as get_schema, for a sqlite3 connection (internal sqlite_* tables skipped)."""
    tables = [
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    ]
    schema: dict[str, list[dict]] = {}
    for table in tables:
        schema[table] = []
        # PRAGMA table_info rows: (cid, name, type, notnull, default, pk)
        for _, column, dtype, *_ in conn.execute(f"PRAGMA table_info({_quote(table)})").fetchall():
            samples = conn.execute(
                f"SELECT DISTINCT {_quote(column)} FROM {_quote(table)} "
                f"WHERE {_quote(column)} IS NOT NULL LIMIT {SAMPLES_PER_COLUMN}"
            ).fetchall()
            schema[table].append(
                {"name": column, "type": dtype or "ANY", "samples": [r[0] for r in samples]}
            )
    return schema


def _short(value: object) -> str:
    """Sample value as text, cut to MAX_SAMPLE_CHARS so prompts stay small."""
    text = str(value)
    return text if len(text) <= MAX_SAMPLE_CHARS else text[: MAX_SAMPLE_CHARS - 3] + "..."


def format_schema(schema: dict[str, list[dict]]) -> str:
    """Render the schema as compact text for an LLM prompt."""
    lines = []
    for table, columns in schema.items():
        lines.append(f"Table {table}:")
        for col in columns:
            samples = ", ".join(repr(_short(s)) for s in col["samples"])
            lines.append(f"  - {col['name']} ({col['type']}), e.g. {samples}")
    return "\n".join(lines)
