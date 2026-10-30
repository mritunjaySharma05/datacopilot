from datacopilot.pipeline.validator import is_read_only


def test_select_is_allowed():
    assert is_read_only("SELECT city, SUM(revenue) FROM sales GROUP BY city")


def test_delete_is_blocked():
    assert not is_read_only("DELETE FROM sales")


def test_drop_is_blocked():
    assert not is_read_only("DROP TABLE sales")


def test_multiple_statements_blocked():
    assert not is_read_only("SELECT * FROM sales; DROP TABLE sales")
