"""Executor: load files into DuckDB, open SQLite read-only, and run validated SQL."""
import re
import sqlite3
import time
from pathlib import Path

import duckdb
import pandas as pd

from datacopilot.pipeline.validator import is_read_only

MAX_ROWS = 1000
PROGRESS_STEPS = 10_000  # SQLite VM steps between timeout checks

Connection = duckdb.DuckDBPyConnection | sqlite3.Connection


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


class _TimeoutConnection(sqlite3.Connection):
    """sqlite3 connection that interrupts any single statement running past `timeout_s`."""

    timeout_s: float | None = None

    def execute(self, sql: str, parameters=(), /):  # type: ignore[override]
        if not self.timeout_s:
            return super().execute(sql, parameters)
        deadline = time.monotonic() + self.timeout_s
        self.set_progress_handler(lambda: int(time.monotonic() > deadline), PROGRESS_STEPS)
        try:
            # Fetching also runs the VM, so callers should fetch before the next execute;
            # the handler stays armed until the next execute replaces it.
            return super().execute(sql, parameters)
        except sqlite3.OperationalError as err:
            if "interrupted" in str(err):
                raise sqlite3.OperationalError(
                    f"Query timed out after {self.timeout_s:g}s"
                ) from err
            raise


def connect_sqlite(path: str | Path, timeout_s: float | None = None) -> sqlite3.Connection:
    """Open a SQLite file read-only (writes fail at the database level).

    If `timeout_s` is set, each statement is interrupted after that many seconds.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"SQLite database not found: {path}")
    conn = sqlite3.connect(
        f"{path.resolve().as_uri()}?mode=ro", uri=True, factory=_TimeoutConnection
    )
    conn.timeout_s = timeout_s
    return conn


def run_query(
    conn: Connection, sql: str, limit: int | None = MAX_ROWS, dialect: str | None = None
) -> pd.DataFrame:
    """Run a read-only query and return at most `limit` rows (all rows if None).

    Works with DuckDB and sqlite3 connections. Raises ValueError if `sql` is not a
    single read-only query (defense in depth: callers should already have validated it).
    """
    if not is_read_only(sql, dialect):
        raise ValueError("Refusing to run SQL that is not a single parseable read-only SELECT.")
    cursor = conn.execute(sql)
    columns = [d[0] for d in cursor.description]
    rows = cursor.fetchall() if limit is None else cursor.fetchmany(limit)
    return pd.DataFrame(rows, columns=columns)
