import sqlite3
from pathlib import Path

import duckdb
import pytest

from datacopilot.pipeline.executor import connect_sqlite, load_csv, run_query, table_name_for


def test_table_name_for_sanitizes():
    assert table_name_for("data/raw/Sales 2024.csv") == "sales_2024"
    assert table_name_for("2024-sales.csv") == "t_2024_sales"


def test_load_csv_creates_table(sales_csv):
    con = duckdb.connect()
    assert load_csv(con, sales_csv) == "sales"
    assert con.execute("SELECT COUNT(*) FROM sales").fetchone()[0] == 4


def test_bundled_sample_loads():
    sample = Path(__file__).resolve().parents[1] / "data" / "sample" / "sales.csv"
    con = duckdb.connect()
    assert load_csv(con, sample) == "sales"
    assert con.execute("SELECT COUNT(*) FROM sales").fetchone()[0] == 50
    columns = [r[0] for r in con.execute("DESCRIBE sales").fetchall()]
    assert columns == ["date", "city", "product", "quantity", "revenue"]


def test_run_query_returns_dataframe(conn):
    df = run_query(conn, "SELECT city, SUM(revenue) AS total FROM sales GROUP BY city ORDER BY total DESC")
    assert list(df.columns) == ["city", "total"]
    assert df.iloc[0]["city"] == "Delhi"
    assert df.iloc[0]["total"] == 128000


def test_run_query_caps_rows(conn):
    df = run_query(conn, "SELECT * FROM range(5000)")
    assert len(df) == 1000


def test_run_query_custom_limit(conn):
    assert len(run_query(conn, "SELECT * FROM sales", limit=2)) == 2


def test_run_query_runs_union_all(conn):
    df = run_query(conn, "SELECT city AS name FROM sales UNION ALL SELECT product FROM sales")
    assert len(df) == 8


def test_run_query_runs_intersect_and_except(conn):
    both = run_query(conn, "SELECT city FROM sales WHERE product = 'Phone' INTERSECT SELECT city FROM sales WHERE product = 'Laptop'")
    assert both["city"].tolist() == ["Delhi"]
    only_phone = run_query(conn, "SELECT city FROM sales WHERE product = 'Phone' EXCEPT SELECT city FROM sales WHERE product = 'Laptop'")
    assert only_phone["city"].tolist() == ["Mumbai"]


def test_run_query_refuses_writes(conn):
    with pytest.raises(ValueError):
        run_query(conn, "DELETE FROM sales")
    assert conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0] == 4


# --- SQLite (read-only) ---

def test_connect_sqlite_reads(tiny_sqlite):
    con = connect_sqlite(tiny_sqlite)
    df = run_query(con, "SELECT `Item Name` FROM items ORDER BY id", dialect="sqlite")
    assert df["Item Name"].tolist() == ["Pen", "Pencil", "Eraser"]


def test_connect_sqlite_is_read_only_at_db_level(tiny_sqlite):
    con = connect_sqlite(tiny_sqlite)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        con.execute("DELETE FROM items")
    assert con.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 3


def test_connect_sqlite_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        connect_sqlite(tmp_path / "nope.sqlite")


def test_connect_sqlite_timeout_interrupts_long_query(tiny_sqlite):
    con = connect_sqlite(tiny_sqlite, timeout_s=0.2)
    endless = "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r) SELECT MAX(x) FROM r"
    with pytest.raises(sqlite3.OperationalError, match="timed out"):
        con.execute(endless).fetchall()
    # The connection stays usable for the next statement.
    assert con.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 3


def test_run_query_limit_none_returns_everything(conn):
    assert len(run_query(conn, "SELECT * FROM range(2500)", limit=None)) == 2500
