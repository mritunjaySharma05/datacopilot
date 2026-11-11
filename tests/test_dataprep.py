import pandas as pd
import pytest

from datacopilot.dataprep import (
    CleaningLog, Finding, changed_rows, column_name_issue, fold_text, normalize_whitespace, parse_duration_ms,
    profile_markdown, profile_table, tf_to_bool, to_snake_case, variant_groups,
)


@pytest.mark.parametrize("raw, expected", [
    ("Crop_Year", "crop_year"),
    ("State_Name", "state_name"),
    ("product_name_lenght", "product_name_length"),
    ("Free Meal Count (K-12)", "free_meal_count_k_12"),
    ("camelCaseName", "camel_case_name"),
    ("  padded ", "padded"),
    ("already_snake", "already_snake"),
])
def test_to_snake_case(raw, expected):
    assert to_snake_case(raw) == expected


def test_column_name_issue():
    assert column_name_issue("crop_year") is None
    assert column_name_issue("Crop_Year") == "not snake_case"
    assert column_name_issue("product_name_lenght") == "misspelling"
    assert column_name_issue("city ") == "leading/trailing whitespace"


def test_fold_text_ignores_accents_case_punctuation():
    assert fold_text("São  Paulo") == fold_text("sao paulo") == "sao paulo"
    assert fold_text("M.Chinnaswamy Stadium") == fold_text("M Chinnaswamy Stadium")


def test_normalize_whitespace_keeps_nan():
    s = pd.Series(["Kharif     ", "  Whole   Year ", None])
    assert normalize_whitespace(s).tolist()[:2] == ["Kharif", "Whole Year"]
    assert pd.isna(normalize_whitespace(s)[2])


def test_variant_groups_finds_spellings():
    s = pd.Series(["Rising Pune Supergiant", "Rising Pune Supergiants", "Mumbai Indians", "mumbai indians"])
    groups = variant_groups(s, key=lambda v: fold_text(v).rstrip("s"))
    assert ["Rising Pune Supergiant", "Rising Pune Supergiants"] in groups
    assert ["Mumbai Indians", "mumbai indians"] in groups


def test_profile_table_counts_issues():
    df = pd.DataFrame({
        "State_Name": ["Goa ", "Goa", "Kerala", "Kerala"],
        "qty": [1, -2, 3, 3],
        "empty": [None] * 4,
        "same": ["x"] * 4,
    })
    df.loc[3] = df.loc[2]  # exact duplicate row
    p = profile_table(df)
    cols = {c["name"]: c for c in p["column_info"]}
    assert p["rows"] == 4 and p["duplicate_rows"] == 1
    assert cols["State_Name"]["name_issue"] == "not snake_case"
    assert cols["State_Name"]["untrimmed"] == 1
    assert cols["State_Name"]["variant_groups"] == [["Goa", "Goa "]]
    assert cols["qty"]["negative"] == 1
    assert p["empty_columns"] == ["empty"] and "same" in p["constant_columns"]


def test_profile_markdown_has_summary_and_findings():
    df = pd.DataFrame({"a": [1, None]})
    md = profile_markdown("demo", {"t": profile_table(df)}, [Finding("odd thing", 3, "detail | piped")], "src")
    assert "# Data quality: demo" in md
    assert "| t | 2 | 1 | 0 | 1 | 0 |" in md
    assert "| odd thing | 3 | detail \\| piped |" in md
    assert "1 (50.0%)" in md


def test_cleaning_log_markdown_groups_by_dataset():
    log = CleaningLog()
    log.record("ipl", "deliveries", "drop dups", 2, 10, 8, "note")
    log.record("crop", "crop", "trim", 5, 8, 8)
    md = log.to_markdown()
    assert md.index("## ipl") < md.index("## crop")
    assert "| deliveries | drop dups | 2 | 10 | 8 | note |" in md


def test_changed_rows_treats_nan_as_equal():
    assert changed_rows(pd.Series(["a ", None, "b"]), pd.Series(["a", None, "b"])) == 1


def test_parse_duration_ms():
    s = pd.Series(["01:23:45.678", "00:00:22.5", "1:30.1", "bad", None])
    assert parse_duration_ms(s).tolist()[:3] == [5025678, 22500, 90100]
    assert parse_duration_ms(s)[3:].isna().all()


def test_tf_to_bool():
    assert tf_to_bool(pd.Series(["t", "f", None])).tolist()[:2] == [True, False]
