import sqlite3

import duckdb
import pytest

SALES_CSV = """date,city,product,quantity,revenue
2024-01-05,Delhi,Laptop,2,110000
2024-01-06,Mumbai,Phone,3,54000
2024-02-10,Delhi,Phone,1,18000
2024-02-11,Pune,Keyboard,4,6000
"""


@pytest.fixture
def sales_csv(tmp_path):
    path = tmp_path / "sales.csv"
    path.write_text(SALES_CSV)
    return path


@pytest.fixture
def conn(sales_csv):
    con = duckdb.connect()
    con.execute(f"CREATE TABLE sales AS SELECT * FROM read_csv_auto('{sales_csv.as_posix()}')")
    yield con
    con.close()


def build_tiny_sqlite(path):
    """Write a tiny SQLite DB (BIRD-style: a column name with a space) to `path`."""
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE items (id INTEGER PRIMARY KEY, "Item Name" TEXT, price REAL);
        INSERT INTO items VALUES (1, 'Pen', 10.0), (2, 'Pencil', 2.5), (3, 'Eraser', NULL);
        CREATE TABLE orders (order_id INTEGER, item_id INTEGER, qty INTEGER);
        INSERT INTO orders VALUES (100, 1, 3), (101, 2, 5), (102, 1, 1);
        """
    )
    con.commit()
    con.close()
    return path


@pytest.fixture
def tiny_sqlite(tmp_path):
    return build_tiny_sqlite(tmp_path / "shop.sqlite")


@pytest.fixture
def fake_llm():
    """Factory for a fake generate_sql that returns `replies` in order and records calls.

    Each recorded call is the dict of keyword arguments (feedback, evidence, dialect).
    """

    def make(*replies):
        calls = []

        def generate(question, schema, **kwargs):
            calls.append(kwargs)
            return replies[len(calls) - 1]

        generate.calls = calls
        return generate

    return make


@pytest.fixture
def schema():
    cols = ["date", "city", "product", "quantity", "revenue"]
    return {"sales": [{"name": c, "type": "VARCHAR", "samples": []} for c in cols]}
