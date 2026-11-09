"""Profiler: read the schema of a DuckDB database so the LLM knows what exists."""
import duckdb

SAMPLES_PER_COLUMN = 3


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


def format_schema(schema: dict[str, list[dict]]) -> str:
    """Render the schema as compact text for an LLM prompt."""
    lines = []
    for table, columns in schema.items():
        lines.append(f"Table {table}:")
        for col in columns:
            samples = ", ".join(repr(str(s)) for s in col["samples"])
            lines.append(f"  - {col['name']} ({col['type']}), e.g. {samples}")
    return "\n".join(lines)
