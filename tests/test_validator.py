from datacopilot.pipeline.validator import check_schema, is_read_only, validate


# --- is_read_only ---

def test_select_is_allowed():
    assert is_read_only("SELECT city, SUM(revenue) FROM sales GROUP BY city")


def test_delete_is_blocked():
    assert not is_read_only("DELETE FROM sales")


def test_drop_is_blocked():
    assert not is_read_only("DROP TABLE sales")


def test_multiple_statements_blocked():
    assert not is_read_only("SELECT * FROM sales; DROP TABLE sales")


def test_cte_select_is_allowed():
    assert is_read_only("WITH t AS (SELECT city FROM sales) SELECT city FROM t")


def test_union_is_allowed():
    assert is_read_only("SELECT city FROM sales UNION SELECT product FROM sales")


def test_union_all_is_allowed():
    assert is_read_only("SELECT city FROM sales UNION ALL SELECT product FROM sales")


def test_chained_parenthesized_union_with_order_is_allowed():
    sql = (
        "(SELECT city AS name FROM sales) UNION ALL (SELECT product FROM sales) "
        "UNION SELECT 'total' ORDER BY name LIMIT 5"
    )
    assert is_read_only(sql)


def test_union_followed_by_write_is_blocked():
    assert not is_read_only("SELECT city FROM sales UNION SELECT city FROM sales; DELETE FROM sales")


def test_union_with_write_in_cte_is_blocked():
    sql = (
        "WITH d AS (DELETE FROM sales RETURNING city) "
        "SELECT city FROM d UNION SELECT city FROM sales"
    )
    assert not is_read_only(sql)


def test_select_into_is_blocked():
    assert not is_read_only("SELECT * INTO backup FROM sales")
    assert not is_read_only("SELECT city FROM sales UNION SELECT city INTO backup FROM sales")


def test_intersect_is_allowed():
    assert is_read_only("SELECT city FROM sales INTERSECT SELECT city FROM sales")


def test_except_is_allowed():
    assert is_read_only("SELECT city FROM sales EXCEPT SELECT city FROM sales")


def test_mixed_set_operations_with_parentheses_are_allowed():
    sql = (
        "(SELECT city FROM sales UNION SELECT product FROM sales) "
        "EXCEPT (SELECT city FROM sales WHERE revenue < 1000) ORDER BY 1"
    )
    assert is_read_only(sql)


def test_intersect_followed_by_write_is_blocked():
    assert not is_read_only("SELECT city FROM sales INTERSECT SELECT city FROM sales; DROP TABLE sales")


def test_except_with_write_in_cte_is_blocked():
    sql = (
        "WITH d AS (DELETE FROM sales RETURNING city) "
        "SELECT city FROM sales EXCEPT SELECT city FROM d"
    )
    assert not is_read_only(sql)


def test_select_into_in_intersect_branch_is_blocked():
    assert not is_read_only("SELECT city FROM sales INTERSECT SELECT city INTO backup FROM sales")


def test_check_schema_covers_except_branches(schema):
    sql = "SELECT city FROM sales EXCEPT SELECT price FROM orders"
    assert check_schema(sql, schema) == {"tables": ["orders"], "columns": ["price"]}


def test_check_schema_covers_union_branches(schema):
    sql = "SELECT city FROM sales UNION SELECT price FROM orders"
    assert check_schema(sql, schema) == {"tables": ["orders"], "columns": ["price"]}


# --- check_schema ---

def test_check_schema_clean_query(schema):
    sql = "SELECT city, SUM(revenue) AS total FROM sales GROUP BY city ORDER BY total DESC"
    assert check_schema(sql, schema) == {"tables": [], "columns": []}


def test_check_schema_unknown_table(schema):
    assert check_schema("SELECT city FROM orders", schema)["tables"] == ["orders"]


def test_check_schema_unknown_column(schema):
    assert check_schema("SELECT price FROM sales", schema)["columns"] == ["price"]


def test_check_schema_is_case_insensitive(schema):
    assert check_schema("SELECT CITY FROM Sales", schema) == {"tables": [], "columns": []}


def test_check_schema_handles_table_alias(schema):
    assert check_schema("SELECT s.city FROM sales AS s", schema)["columns"] == []
    assert check_schema("SELECT s.price FROM sales AS s", schema)["columns"] == ["s.price"]


def test_check_schema_ignores_cte_names(schema):
    sql = "WITH top AS (SELECT city, revenue FROM sales) SELECT city FROM top"
    assert check_schema(sql, schema) == {"tables": [], "columns": []}


def test_check_schema_star_is_fine(schema):
    assert check_schema("SELECT * FROM sales", schema) == {"tables": [], "columns": []}


# --- validate ---

def test_validate_ok(schema):
    assert validate("SELECT city FROM sales", schema) == []


def test_validate_empty_sql(schema):
    assert validate("", schema) == ["No SQL was produced."]


def test_validate_rejects_writes(schema):
    assert validate("DELETE FROM sales", schema) == [
        "Only a single read-only SELECT query is allowed."
    ]


def test_validate_reports_unknown_names(schema):
    problems = validate("SELECT price FROM orders", schema)
    assert "Unknown tables: orders." in problems
    assert "Unknown columns: price." in problems
