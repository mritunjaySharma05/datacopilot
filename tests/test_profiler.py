from datacopilot.pipeline.profiler import format_schema, get_schema


def test_get_schema_lists_tables_and_columns_in_order(conn):
    schema = get_schema(conn)
    assert list(schema) == ["sales"]
    names = [c["name"] for c in schema["sales"]]
    assert names == ["date", "city", "product", "quantity", "revenue"]


def test_get_schema_reports_types(conn):
    types = {c["name"]: c["type"] for c in get_schema(conn)["sales"]}
    assert types["date"] == "DATE"
    assert types["city"] == "VARCHAR"
    assert types["quantity"] == "BIGINT"


def test_get_schema_has_at_most_three_distinct_samples(conn):
    city = next(c for c in get_schema(conn)["sales"] if c["name"] == "city")
    assert len(city["samples"]) == 3
    assert len(set(city["samples"])) == 3
    assert set(city["samples"]) <= {"Delhi", "Mumbai", "Pune"}


def test_get_schema_skips_nulls(conn):
    conn.execute("CREATE TABLE notes AS SELECT NULL::VARCHAR AS note UNION ALL SELECT 'hi'")
    note = get_schema(conn)["notes"][0]
    assert note["samples"] == ["hi"]


def test_get_schema_empty_database():
    import duckdb

    assert get_schema(duckdb.connect()) == {}


def test_format_schema_mentions_table_columns_and_samples():
    schema = {"sales": [{"name": "city", "type": "VARCHAR", "samples": ["Delhi"]}]}
    text = format_schema(schema)
    assert "Table sales:" in text
    assert "city (VARCHAR)" in text
    assert "'Delhi'" in text


def test_get_sqlite_schema(tiny_sqlite):
    from datacopilot.pipeline.executor import connect_sqlite
    from datacopilot.pipeline.profiler import get_sqlite_schema

    schema = get_sqlite_schema(connect_sqlite(tiny_sqlite))
    assert list(schema) == ["items", "orders"]
    items = {c["name"]: c for c in schema["items"]}
    assert list(items) == ["id", "Item Name", "price"]
    assert items["price"]["type"] == "REAL"
    assert items["price"]["samples"] == [10.0, 2.5]  # NULL skipped


def test_get_sqlite_schema_skips_internal_tables(tmp_path):
    import sqlite3

    from datacopilot.pipeline.profiler import get_sqlite_schema

    con = sqlite3.connect(tmp_path / "a.sqlite")
    con.execute("CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT, v TEXT)")
    con.execute("INSERT INTO t (v) VALUES ('x')")
    assert list(get_sqlite_schema(con)) == ["t"]  # sqlite_sequence is hidden


def test_format_schema_truncates_long_samples():
    schema = {"t": [{"name": "notes", "type": "TEXT", "samples": ["x" * 500]}]}
    text = format_schema(schema)
    assert "x" * 500 not in text
    assert "..." in text
