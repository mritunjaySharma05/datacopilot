"""SQL validator: allow only safe, read-only SELECT queries over known tables/columns."""
import sqlglot
from sqlglot import exp

# exp.Into covers SELECT ... INTO new_table, which creates a table in PostgreSQL.
FORBIDDEN = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Into)
DIALECT = "duckdb"


def _is_select_query(node: exp.Expression) -> bool:
    """True for a SELECT, or a UNION / UNION ALL whose every branch is one."""
    if isinstance(node, exp.Subquery):  # parenthesized branch: (SELECT ...)
        return _is_select_query(node.this)
    if isinstance(node, exp.Select):
        return True
    if isinstance(node, exp.Union):
        return _is_select_query(node.left) and _is_select_query(node.right)
    return False


def is_read_only(sql: str) -> bool:
    """Return True if `sql` is a single read-only SELECT (UNION / UNION ALL allowed)."""
    try:
        statements = sqlglot.parse(sql)
    except sqlglot.errors.ParseError:
        return False
    for stmt in statements:
        if stmt is None or not _is_select_query(stmt):
            return False
        if any(stmt.find(node) for node in FORBIDDEN):
            return False
    return True


def check_schema(sql: str, schema: dict[str, list[dict]]) -> dict[str, list[str]]:
    """Return {"tables": [...], "columns": [...]} that `sql` uses but `schema` lacks.

    Names are compared case-insensitively (DuckDB identifiers are). Names that the
    query defines itself (CTEs, table aliases, SELECT aliases) are not reported.
    """
    tree = sqlglot.parse_one(sql, read=DIALECT)
    known = {t.lower(): {c["name"].lower() for c in cols} for t, cols in schema.items()}
    all_columns = set().union(*known.values()) if known else set()

    ctes = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    select_aliases = {a.alias.lower() for a in tree.find_all(exp.Alias)}

    unknown_tables: list[str] = []
    alias_to_table: dict[str, str] = {}
    for table in tree.find_all(exp.Table):
        name = table.name.lower()
        if name in ctes:
            continue
        if name not in known and table.name not in unknown_tables:
            unknown_tables.append(table.name)
        alias_to_table[table.alias_or_name.lower()] = name

    unknown_columns: list[str] = []
    for column in tree.find_all(exp.Column):
        name = column.name.lower()
        if not name or name == "*" or name in select_aliases:
            continue
        qualifier = column.table.lower()
        table = alias_to_table.get(qualifier) if qualifier else None
        if table in known:
            ok = name in known[table]
        elif qualifier:
            ok = True  # qualified by a CTE/subquery we don't track, or an unknown table
        else:
            ok = name in all_columns
        if not ok and column.sql() not in unknown_columns:
            unknown_columns.append(column.sql())

    return {"tables": unknown_tables, "columns": unknown_columns}


def validate(sql: str, schema: dict[str, list[dict]]) -> list[str]:
    """Return a list of human-readable problems with `sql`; empty means it is safe to run."""
    if not sql.strip():
        return ["No SQL was produced."]
    if not is_read_only(sql):
        return ["Only a single read-only SELECT query is allowed."]
    unknown = check_schema(sql, schema)
    problems = []
    if unknown["tables"]:
        problems.append(f"Unknown tables: {', '.join(unknown['tables'])}.")
    if unknown["columns"]:
        problems.append(f"Unknown columns: {', '.join(unknown['columns'])}.")
    return problems
