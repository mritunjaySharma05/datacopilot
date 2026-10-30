"""SQL validator: allow only safe, read-only SELECT queries."""
import sqlglot
from sqlglot import exp

FORBIDDEN = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter)


def is_read_only(sql: str) -> bool:
    """Return True if every statement in `sql` is a read-only SELECT."""
    try:
        statements = sqlglot.parse(sql)
    except sqlglot.errors.ParseError:
        return False
    for stmt in statements:
        if stmt is None or not isinstance(stmt, exp.Select):
            return False
        if any(stmt.find(node) for node in FORBIDDEN):
            return False
    return True
