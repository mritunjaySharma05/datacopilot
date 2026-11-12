"""Tests for scripts/build_features.py on a tiny fake processed dir and question sets."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from datacopilot.pipeline.features import FEATURE_NAMES, DatasetContext
from datacopilot.questions import SET_FILES, write_jsonl

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_features.py"
spec = importlib.util.spec_from_file_location("build_features", SCRIPT)
bf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bf)

SCHEMA = {"crop_production": [{"name": "crop", "type": "VARCHAR", "samples": ["Rice"]},
                              {"name": "crop_year", "type": "BIGINT", "samples": [2010]}]}


def q(qid, question, answerable, split="train", utype=None, dataset="crop"):
    return {"id": qid, "dataset": dataset, "question": question, "answerable": answerable,
            "unanswerable_type": utype, "split": split, "gold_sql": None, "gold_answer": None,
            "difficulty": "easy"}


def test_build_feature_table_columns_and_labels():
    inputs = {"crop": (SCHEMA, DatasetContext(1997, 2015, frozenset({"rice"})))}
    table = bf.build_feature_table([q("a", "How much rice in 2010?", True),
                                    q("b", "Delete all rice rows.", False, "test", "write_request")], inputs)
    assert list(table.columns) == [*bf.META_COLUMNS, *FEATURE_NAMES]
    assert table.answerable.tolist() == [1, 0]
    assert table.loc[1, "starts_with_write_verb"] == 1 and table.loc[1, "split"] == "test"


def test_compare_classes_uses_only_requested_split():
    rows = pd.DataFrame([
        {"split": "train", "answerable": 1, **{n: 1.0 for n in FEATURE_NAMES}},
        {"split": "train", "answerable": 0, **{n: 3.0 for n in FEATURE_NAMES}},
        {"split": "test", "answerable": 0, **{n: 100.0 for n in FEATURE_NAMES}},
    ])
    means = bf.compare_classes(rows)
    assert list(means.columns) == ["answerable", "unanswerable", "difference"]
    assert means.loc["n_words"].tolist() == [1.0, 3.0, 2.0]  # the test row is ignored


@pytest.fixture
def fake_dirs(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    pd.DataFrame({"state_name": ["Punjab", "Kerala"] * 50, "crop_year": [2000, 2001] * 50,
                  "crop": ["Rice", "Wheat"] * 50, "area": [1.0] * 100, "production": [2.0] * 100}
                 ).to_csv(processed / "crop_production.csv", index=False)
    qdir = tmp_path / "questions"
    write_jsonl(qdir / SET_FILES["csv"], [q("crop-001", "How much rice did Punjab grow in 2000?", True)])
    write_jsonl(qdir / SET_FILES["unanswerable"], [
        q("crop-u001", "How much rice did Delhi grow in 2020?", False, "train", "out_of_range")])
    write_jsonl(qdir / SET_FILES["hinglish"], [
        q("crop-001-hi", "2000 mein Punjab ne kitna rice ugaya?", True)])
    write_jsonl(qdir / SET_FILES["hinglish_unanswerable"], [
        q("crop-u001-hi", "2020 mein Delhi ne kitna rice ugaya?", False, "train", "out_of_range")])
    return processed, qdir


def test_main_writes_csv(fake_dirs, tmp_path, capsys):
    processed, qdir = fake_dirs
    out = tmp_path / "features.csv"
    assert bf.main(["--questions-dir", str(qdir), "--processed-dir", str(processed), "--out", str(out)]) == 0
    table = pd.read_csv(out)
    assert table.id.tolist() == ["crop-001", "crop-u001", "crop-001-hi", "crop-u001-hi"]
    assert table.question_set.tolist() == ["csv", "unanswerable", "hinglish", "hinglish_unanswerable"]
    hi = table.set_index("id")
    assert hi.loc["crop-001-hi", "hinglish_share"] > 0 and hi.loc["crop-u001-hi", "year_out_of_range"] == 1
    unans = table.set_index("id").loc["crop-u001"]
    assert unans.year_out_of_range == 1 and unans.n_unmatched_entities == 1
    ans = table.set_index("id").loc["crop-001"]
    assert ans.year_out_of_range == 0 and ans.n_unmatched_entities == 0
    assert "Mean feature value, train split" in capsys.readouterr().out
