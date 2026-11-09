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


@pytest.fixture
def schema():
    cols = ["date", "city", "product", "quantity", "revenue"]
    return {"sales": [{"name": c, "type": "VARCHAR", "samples": []} for c in cols]}
