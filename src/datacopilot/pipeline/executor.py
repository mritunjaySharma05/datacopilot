"""Executor: load files into DuckDB and run validated read-only SQL."""
import re
from pathlib import Path

import duckdb
import pandas as pd

from datacopilot.pipeline.validator import is_read_only

MAX_ROWS = 1000


def table_name_for(path: str | Path) -> str:
    """Turn a file name into a safe SQL table name, e.g. 'Sales 2024.csv' -> 'sales_2024'."""
    name = re.sub(r"\W+", "_", Path(path).stem.lower()).strip("_")
    if not name or name[0].isdigit():
        name = f"t_{name}"
    return name


def load_csv(conn: duckdb.DuckDBPyConnection, path: str | Path) -> str:
    """Load a CSV into a DuckDB table named after the file; return the table name."""
    table = table_name_for(path)
    conn.execute(
        f'CREATE OR REPLACE TABLE "{table}" AS SELECT * FROM read_csv_auto(?)',
        [str(path)],
    )
    return table


def run_query(
    conn: duckdb.DuckDBPyConnection, sql: str, limit: int = MAX_ROWS
) -> pd.DataFrame:
    """Run a read-only query and return at most `limit` rows as a DataFrame.

    Raises ValueError if `sql` is not a single read-only SELECT (defense in depth:
    callers should already have validated it).
    """
    if not is_read_only(sql):
        raise ValueError("Refusing to run SQL that is not a read-only SELECT.")
    cursor = conn.execute(sql)
    columns = [d[0] for d in cursor.description]
    return pd.DataFrame(cursor.fetchmany(limit), columns=columns)
