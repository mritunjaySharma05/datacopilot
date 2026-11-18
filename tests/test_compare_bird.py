"""Tests for scripts/compare_bird.py on hand-made BIRD runs."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("compare_bird", SCRIPTS / "compare_bird.py")
cb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cb)


def run(results: list[tuple[bool, str]]) -> pd.DataFrame:
    """Fake eval_bird output: one (correct, error) per question."""
    dbs, diffs = ["shop", "shop", "zoo", "zoo"], ["simple", "moderate", "simple", "challenging"]
    return pd.DataFrame([{"question_id": i, "db_id": dbs[i], "difficulty": diffs[i], "question": f"q{i}",
                          "correct": c, "error": e, "pred_sql": "SELECT 1", "seconds": 1.0}
                         for i, (c, e) in enumerate(results)])


V = run([(True, ""), (True, ""), (False, "Validation failed: x (after one retry)"), (False, "")])
B = run([(True, ""), (False, "Execution failed: no such column"), (False, ""), (True, "")])


def test_outcome_categories():
    rows = run([(True, ""), (False, "Gold SQL failed: x"), (False, "Validation failed: y"),
                (False, "Execution failed: z")])
    assert [cb.outcome(r) for r in rows.itertuples()] == ["correct", "gold_failed", "validation_failed",
                                                         "execution_failed"]
    assert cb.outcome(run([(False, float("nan"))] * 4).iloc[0]) == "wrong_result"


def test_pair_runs_and_mismatch():
    p = cb.pair_runs(V, B)
    assert p.correct_v.tolist() == [True, True, False, False]
    assert p.outcome_b.tolist() == ["correct", "execution_failed", "wrong_result", "correct"]
    with pytest.raises(ValueError):
        cb.pair_runs(V, B.iloc[:3])


def test_accuracy_by_orders_difficulty_and_adds_all():
    a = cb.accuracy_by(cb.pair_runs(V, B), "difficulty", cb.DIFFICULTY_ORDER)
    assert list(a.index) == ["simple", "moderate", "challenging", "all"]
    assert a.loc["all", "n"] == 4 and a.loc["all", "validated"] == 0.5 and a.loc["all", "baseline"] == 0.5


def test_report_has_mcnemar_and_disagreements():
    text = cb.report(cb.pair_runs(V, B), "BIRD test")
    assert "**Validated 50.0% vs baseline 50.0%** (2 vs 2 of 4)" in text
    assert "validated only right 1, baseline only right 1" in text
    assert "| 1 | shop | moderate | validated | execution_failed | correct |" in text
    assert "| 3 | zoo | challenging | baseline | correct | wrong_result |" in text
    assert "| shop | 2 | 100.0% | 50.0% | +50.0 pts |" in text
    assert "| execution_failed | 1 | 0 |" in text  # baseline crash fixed by validated retry


def test_newest(tmp_path):
    for stamp in ("20261118_100000", "20261118_200000"):
        (tmp_path / f"bird_validated_sample300_seed42_{stamp}.csv").write_text("x")
    assert cb.newest(tmp_path, "validated", "sample300_seed42").name.endswith("200000.csv")
    with pytest.raises(FileNotFoundError):
        cb.newest(tmp_path, "baseline", "sample300_seed42")
