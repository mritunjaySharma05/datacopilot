"""Tests for scripts/compare_modes.py on small hand-made runs."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_modes.py"
spec = importlib.util.spec_from_file_location("compare_modes", SCRIPT)
cm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cm)


def run(results: dict[str, tuple[bool, bool]], qset: str = "csv") -> pd.DataFrame:
    """Fake eval_questions output: id -> (strict, relaxed)."""
    return pd.DataFrame([{"id": i, "question_set": "hinglish" if i.endswith("-hi") else qset,
                          "dataset": i.split("-")[0], "difficulty": "easy", "question": f"q {i}?",
                          "strict": s, "relaxed": r, "status": "correct" if s else "wrong_result",
                          "pred_sql": "SELECT 1", "error": ""} for i, (s, r) in results.items()])


V = run({"crop-001": (True, True), "crop-002": (True, True), "f1-001": (False, True), "f1-001-hi": (True, True)})
B = run({"crop-001": (True, True), "crop-002": (False, False), "f1-001": (False, False), "f1-001-hi": (True, True)})


def test_pair_runs_matches_ids_and_sets_language():
    p = cm.pair_runs(V, B, {"human-001": "hinglish"})
    assert len(p) == 4
    assert dict(zip(p.id, p.language)) == {"crop-001": "en", "crop-002": "en", "f1-001": "en", "f1-001-hi": "hinglish"}
    assert p.set_index("id").loc["crop-002", ["strict_v", "strict_b"]].tolist() == [True, False]


def test_pair_runs_uses_heldout_language_map():
    v = run({"human-001": (True, True)}, "heldout")
    assert cm.pair_runs(v, v, {"human-001": "hinglish"}).language.tolist() == ["hinglish"]


def test_pair_runs_rejects_different_questions():
    with pytest.raises(ValueError):
        cm.pair_runs(V, B.iloc[:3])


def test_mcnemar_exact_counts_discordant_pairs():
    t = cm.mcnemar_exact(pd.Series([True] * 6 + [False] * 4), pd.Series([False] * 6 + [False] * 4))
    assert (t["only_validated"], t["only_baseline"], t["discordant"]) == (6, 0, 6)
    assert t["p_value"] == pytest.approx(2 * 0.5 ** 6)  # exact two-sided binomial
    assert cm.mcnemar_exact(pd.Series([True]), pd.Series([True]))["p_value"] == 1.0


def test_accuracy_by_includes_all_row():
    t = cm.accuracy_by(cm.pair_runs(V, B), "language")
    assert t.loc["all", "n"] == 4 and t.loc["en", "n"] == 3
    assert t.loc["all", "strict_v"] == 0.75 and t.loc["all", "strict_b"] == 0.5
    assert t.loc["all", "relaxed_v"] == 1.0


def test_disagreements_on_either_metric():
    d = cm.disagreements(cm.pair_runs(V, B))
    assert d.id.tolist() == ["crop-002", "f1-001"]  # f1-001 differs only on relaxed


def test_report_sections():
    text = cm.report({"Test": cm.pair_runs(V, B)})
    assert "## Test (4 questions)" in text and "### McNemar's test" in text
    assert "| crop-002 | q crop-002? | correct | wrong | wrong_result |" in text
    assert "| f1-001 | q f1-001? | relaxed only | wrong | wrong_result |" in text


def test_newest_picks_latest_stamp(tmp_path):
    for stamp in ("20261101_000000", "20261118_120000"):
        (tmp_path / f"questions_validated_heldout_test_{stamp}.csv").write_text("x")
    assert cm.newest(tmp_path, "validated", "heldout_test").name.endswith("20261118_120000.csv")
    with pytest.raises(FileNotFoundError):
        cm.newest(tmp_path, "baseline", "heldout_test")
