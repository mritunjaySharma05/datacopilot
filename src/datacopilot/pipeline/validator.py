"""SQL validator: allow only safe, read-only SELECT queries over known tables/columns."""
import sqlglot
from sqlglot import exp

# exp.Into covers SELECT ... INTO new_table, which creates a table in PostgreSQL.
FORBIDDEN = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Into)
DIALECT = "duckdb"


def _is_select_query(node: exp.Expression) -> bool:
    """True for a SELECT, or a UNION / INTERSECT / EXCEPT whose every branch is one."""
    if isinstance(node, exp.Subquery):  # parenthesized branch: (SELECT ...)
        return _is_select_query(node.this)
    if isinstance(node, exp.Select):
        return True
    if isinstance(node, exp.SetOperation):  # Union, Intersect, Except
        return _is_select_query(node.left) and _is_select_query(node.right)
    return False


def is_read_only(sql: str, dialect: str | None = None) -> bool:
    """Return True if `sql` is a single read-only SELECT (UNION / INTERSECT / EXCEPT allowed).

    `dialect` is the sqlglot dialect to parse with (e.g. "sqlite" for backtick identifiers).
    """
    try:
        statements = sqlglot.parse(sql, read=dialect)
    except sqlglot.errors.ParseError:
        return False
    for stmt in statements:
        if stmt is None or not _is_select_query(stmt):
            return False
        if any(stmt.find(node) for node in FORBIDDEN):
            return False
    return True


def check_schema(
    sql: str, schema: dict[str, list[dict]], dialect: str = DIALECT
) -> dict[str, list[str]]:
    """Return {"tables": [...], "columns": [...]} that `sql` uses but `schema` lacks.

    Names are compared case-insensitively (DuckDB and SQLite identifiers are). Names that the
    query defines itself (CTEs, table aliases, SELECT aliases) are not reported.
    """
    tree = sqlglot.parse_one(sql, read=dialect)
    known = {t.lower(): {c["name"].lower() for c in cols} for t, cols in schema.items()}
    all_columns = set().union(*known.values()) if known else set()

    ctes = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    select_aliases = {a.alias.lower() for a in tree.find_all(exp.Alias)}

    unknown_tables: list[str] = []
    alias_to_table: dict[str, str] = {}
    used_tables: set[str] = set()
    for table in tree.find_all(exp.Table):
        name = table.name.lower()
        if name in ctes:
            continue
        if name in known:
            used_tables.add(name)
        elif table.name not in unknown_tables:
            unknown_tables.append(table.name)
        alias_to_table[table.alias_or_name.lower()] = name

    # Unqualified columns must exist in a table the query actually uses. (SQLite would
    # otherwise turn an unknown "quoted" name into a string literal and return wrong data.)
    in_scope = set().union(*(known[t] for t in used_tables)) if used_tables else all_columns

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
            ok = name in in_scope
        shown = column.sql(dialect=dialect)
        if not ok and shown not in unknown_columns:
            unknown_columns.append(shown)

    return {"tables": unknown_tables, "columns": unknown_columns}


def find_syntax_error(sql: str, dialect: str | None = DIALECT) -> str | None:
    """Return a short message locating the first parse error, or None if `sql` parses."""
    try:
        sqlglot.parse(sql, read=dialect)
    except sqlglot.errors.ParseError as err:
        if not err.errors:
            return f"SQL syntax error: {str(err).splitlines()[0]}"
        first = err.errors[0]
        near = (first.get("highlight") or "").strip()
        where = f" near '{near}'" if near else ""
        return f"SQL syntax error{where} (line {first.get('line')}, column {first.get('col')})."
    return None


def validate(sql: str, schema: dict[str, list[dict]], dialect: str = DIALECT) -> list[str]:
    """Return a list of human-readable problems with `sql`; empty means it is safe to run."""
    if not sql.strip():
        return ["No SQL was produced."]
    syntax_error = find_syntax_error(sql, dialect)
    if syntax_error:
        return [syntax_error]
    if not is_read_only(sql, dialect):
        return ["Only a single read-only SELECT query is allowed."]
    unknown = check_schema(sql, schema, dialect)
    problems = []
    if unknown["tables"]:
        problems.append(f"Unknown tables: {', '.join(unknown['tables'])}.")
    if unknown["columns"]:
        problems.append(f"Unknown columns: {', '.join(unknown['columns'])}.")
    return problems
