import sqlite3

import duckdb
import pytest

from datacopilot.pipeline import features as ft

SCHEMA = {
    "crop_production": [
        {"name": "state_name", "type": "VARCHAR", "samples": ["Punjab", "Kerala"]},
        {"name": "crop_year", "type": "BIGINT", "samples": [2010, 2011]},
        {"name": "crop", "type": "VARCHAR", "samples": ["Rice", "Wheat"]},
        {"name": "area", "type": "DOUBLE", "samples": [10.0]},
        {"name": "production", "type": "DOUBLE", "samples": [20.0]},
    ]
}
CONTEXT = ft.DatasetContext(1997, 2015, frozenset({"punjab", "kerala", "rice", "wheat", "kharif", "uttar", "pradesh"}))


def f(question, context=CONTEXT):
    return ft.extract_features(question, SCHEMA, context)


# --- helpers ---

def test_tokenize_and_stem():
    assert ft.tokenize("India's rice-growing States?") == ["india", "rice-growing", "states"]
    assert ft.stem("states") == "state" and ft.stem("districts") == "district"
    assert ft.stem("categories") == "category" and ft.stem("class") == "class"


def test_schema_tokens_split_names():
    assert {"crop", "production", "state", "name", "year", "area"} <= ft.schema_tokens(SCHEMA)


@pytest.mark.parametrize("text, years", [
    ("rice production in 2013", [2013]),
    ("between 2005 and 2013", [2005, 2013]),
    ("at least 2000 balls faced", []),
    ("more than 1,500 runs", []),
    ("no years here", []),
])
def test_find_years(text, years):
    assert ft.find_years(text) == years


def test_entities_skip_first_word_and_time_words():
    assert ft.entities("How many matches did Kerala Blasters win?") == [["kerala", "blasters"]]
    assert ft.entities("Which orders were placed on Sunday?") == []
    assert ft.entities("Delete all canceled orders.") == []
    assert ft.entities("What is Virat Kohli's net worth?") == [["virat", "kohli"]]


@pytest.mark.parametrize("question, expected", [
    ("How many orders were placed in 2017?", "qtype_how_many"),
    ("Kitne customers hain?", "qtype_how_many"),
    ("Who is the CEO of Olist?", "qtype_who"),
    ("Which state has the most sellers?", "qtype_which_what"),
    ("On what date was the first match played?", "qtype_when_where"),
    ("Is Formula 1 boring?", "qtype_yes_no"),
    ("Delete all races from the 2020 season.", "qtype_imperative"),
])
def test_question_type_one_hot(question, expected):
    types = ft.question_type(question)
    assert types[expected] == 1 and sum(types.values()) == 1


# --- extract_features ---

def test_feature_names_and_types():
    out = f("Which state grew the most rice in 2013?")
    assert tuple(out) == ft.FEATURE_NAMES
    assert all(isinstance(v, float) for v in out.values())


def test_answerable_question_matches_schema_and_values():
    out = f("Which 5 states had the largest rice production in 2013?")
    assert out["schema_overlap"] > 0.5 and out["unmatched_content_words"] == 0
    assert out["n_years"] == 1 and out["year_out_of_range"] == 0 and out["n_numbers"] == 1


def test_out_of_range_year_and_distance():
    out = f("What was rice production in 2020?")
    assert out["year_out_of_range"] == 1 and out["years_outside_by"] == 5


def test_unknown_entity():
    out = f("How much wheat was produced in Delhi?")
    assert out["n_entities"] == 1 and out["n_unmatched_entities"] == 1
    assert f("How much wheat was produced in Punjab?")["n_unmatched_entities"] == 0


def test_write_request_signals():
    out = f("Drop the crop production table.")
    assert out["has_write_verb"] == out["starts_with_write_verb"] == out["mentions_table_word"] == 1
    assert out["qtype_imperative"] == 1 and out["ends_with_question_mark"] == 0


def test_opinion_and_external_cues():
    assert f("Which crop is the best for farmers to grow?")["n_opinion_words"] == 1
    assert f("Who is India's agriculture minister?")["n_external_cues"] >= 1
    assert f("What is the population of Uttar Pradesh?")["n_external_cues"] == 1


def test_hinglish_share():
    assert f("2013 mein sabse zyada rice kis state ne ugaya?")["hinglish_share"] > 0.4
    assert f("Which state grew the most rice?")["hinglish_share"] == 0


def test_fuzzy_match_on_column_names():
    # 'productions' / 'prodution' still match the 'production' column
    assert f("Total productions by crop?")["schema_overlap"] == 1.0
    assert f("Total prodution by crop?")["schema_overlap"] == 1.0
    assert f("Total production of rice?")["schema_overlap"] == 0.5  # 'rice' is a value, not a column


def test_without_context_uses_schema_samples():
    out = ft.extract_features("How much rice was grown in Kerala in 2020?", SCHEMA)
    assert out["n_unmatched_entities"] == 0  # 'Kerala' is a sample value
    assert out["year_out_of_range"] == 0  # no year range known


def test_empty_question():
    out = f("")
    assert out["n_words"] == 0 and out["schema_overlap"] == 0 and out["hinglish_share"] == 0


# --- build_context ---

def _load(conn):
    conn.execute("CREATE TABLE crop (state_name TEXT, district_id TEXT, crop_year INTEGER, sown DATE)")
    rows = [("Punjab", f"d{i}", 2000 + i % 5, f"200{i % 5}-06-01") for i in range(2000)]
    rows.append(("Goa", "dx", 2000, "2030-01-01"))  # one rare outlier date
    conn.executemany("INSERT INTO crop VALUES (?, ?, ?, ?)", rows)


@pytest.mark.parametrize("make", [duckdb.connect, lambda: sqlite3.connect(":memory:")])
def test_build_context_years_and_values(make):
    conn = make()
    _load(conn)
    schema = {"crop": [{"name": "state_name", "type": "TEXT", "samples": []},
                       {"name": "district_id", "type": "TEXT", "samples": []},
                       {"name": "crop_year", "type": "INTEGER", "samples": []},
                       {"name": "sown", "type": "DATE", "samples": []}]}
    ctx = ft.build_context(conn, schema)
    assert (ctx.year_min, ctx.year_max) == (2000, 2004)  # 2030 is < 0.1% of rows
    assert {"punjab", "goa"} <= ctx.value_tokens
    assert "d1" not in ctx.value_tokens  # *_id columns skipped


def test_build_context_skips_high_cardinality_text():
    conn = duckdb.connect()
    conn.execute("CREATE TABLE t AS SELECT 'comment ' || i AS note FROM range(50) r(i)")
    ctx = ft.build_context(conn, {"t": [{"name": "note", "type": "VARCHAR", "samples": []}]}, max_distinct=10)
    assert ctx.value_tokens == frozenset()


def test_column_years_ignores_rare_years():
    conn = duckdb.connect()
    conn.execute("CREATE TABLE t AS SELECT CASE WHEN i = 0 THEN 1990 ELSE 2010 END AS season FROM range(5000) r(i)")
    assert ft.column_years(conn, "t", "season") == [2010]
