"""Tests for scripts/eval_gate.py (no LLM: fake SQL generator, hand-made gate decisions)."""
import importlib.util
import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from datacopilot.pipeline.features import DatasetContext

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("eval_gate", SCRIPTS / "eval_gate.py")
eg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eg)

SCHEMA = {"crop_production": [{"name": "crop", "type": "VARCHAR", "samples": ["Rice"]},
                              {"name": "production", "type": "DOUBLE", "samples": [1.0]}]}


@pytest.fixture
def conn():
    c = duckdb.connect()
    c.execute("CREATE TABLE crop_production AS SELECT * FROM (VALUES ('Rice', 10.0)) t(crop, production)")
    yield c
    c.close()


def q(qid, answerable, utype=None, qset="csv"):
    return {"id": qid, "dataset": "crop", "question_set": qset, "question": f"question {qid}?",
            "answerable": answerable, "unanswerable_type": utype}


ROWS = [q("a1", True), q("a2", True), q("a3", True), q("u1", False, "subjective", "unanswerable"),
        q("u2", False, "write_request", "unanswerable"), q("u3", False, "missing_column", "unanswerable")]
GATE = pd.DataFrame({"id": ["a1", "a2", "a3", "u1", "u2", "u3"],
                     "gate_pass": [True, False, True, False, False, True],
                     "p_answerable": [0.9, 0.2, 0.8, 0.1, 0.05, 0.7]})
ANS = pd.DataFrame({"id": ["a1", "a2", "a3"], "strict": [True, True, False], "relaxed": [True, True, True],
                    "status": ["correct", "correct", "wrong_result"]})
UNANS = pd.DataFrame({"id": ["u1", "u2", "u3"], "answered": [True, False, True],
                      "status": ["answered", "validation_failed", "answered"]})


def test_run_unanswerable_records_answer_or_failure(conn):
    r = q("u1", False, "missing_column")
    rec = eg.run_unanswerable(r, conn, SCHEMA, generate=lambda *a, **k: "SELECT COUNT(*) FROM crop_production")
    assert rec["answered"] and rec["n_rows"] == 1 and rec["status"] == "answered"
    bad = eg.run_unanswerable(r, conn, SCHEMA, generate=lambda *a, **k: "SELECT rainfall FROM crop_production")
    assert not bad["answered"] and bad["status"] == "validation_failed" and "rainfall" in bad["pred_sql"]
    write = eg.run_unanswerable(r, conn, SCHEMA, generate=lambda *a, **k: "DELETE FROM crop_production")
    assert not write["answered"]


def test_combine_scores_with_and_without_gate():
    df = eg.combine(ROWS, GATE, ANS, UNANS).set_index("id")
    # no gate: a1, a2 right; a3 wrong; u1, u3 answered (wrong); u2 failed (no answer)
    assert df.correct_no_gate_strict.to_dict() == {"a1": True, "a2": True, "a3": False,
                                                   "u1": False, "u2": True, "u3": False}
    # gate: a2 refused (lost); u1 refused (saved); u3 passes and is answered (still wrong)
    assert df.correct_gate_strict.to_dict() == {"a1": True, "a2": False, "a3": False,
                                                "u1": True, "u2": True, "u3": False}
    assert df.loc["a3", "correct_no_gate_relaxed"] and df.loc["a3", "correct_gate_relaxed"]


def test_combine_requires_a_run_for_every_question():
    with pytest.raises(ValueError, match="no pipeline run"):
        eg.combine(ROWS, GATE, ANS.iloc[:2], UNANS)


def test_summary_counts():
    s = eg.summary(eg.combine(ROWS, GATE, ANS, UNANS))
    assert (s["correct_refusals"], s["false_refusals"], s["false_refusals_correct"]) == (2, 1, 1)
    assert (s["unans_answered_no_gate"], s["unans_answered_gate"], s["unans_pipeline_failed"]) == (2, 1, 1)
    assert s["acc_no_gate_strict"] == pytest.approx(3 / 6) and s["acc_gate_strict"] == pytest.approx(3 / 6)
    assert s["mcnemar_strict"]["only_gate"] == 1 and s["mcnemar_strict"]["only_no_gate"] == 1


def test_mcnemar_exact():
    m = eg.mcnemar(pd.Series([True] * 5 + [False]), pd.Series([False] * 6))
    assert (m["only_gate"], m["only_no_gate"]) == (5, 0) and m["p_value"] == pytest.approx(2 * 0.5 ** 5)


def test_gate_predictions_use_bundle():
    class Always:
        def predict_proba(self, X):
            import numpy as np
            return np.array([[0.3, 0.7]] * len(X))
    bundle = {"model": Always(), "feature_names": ["schema_overlap"], "threshold": 0.5}
    inputs = {"crop": (SCHEMA, DatasetContext(1997, 2015, frozenset({"rice"})))}
    out = eg.gate_predictions(ROWS[:2], inputs, bundle)
    assert out.gate_pass.tolist() == [True, True] and out.p_answerable.tolist() == [0.7, 0.7]


def test_report_lists_false_refusals_and_leaks():
    text = eg.report({"Test": eg.combine(ROWS, GATE, ANS, UNANS)})
    assert "## Test (6 questions: 3 answerable, 3 unanswerable)" in text
    assert "- a2 (pipeline correct, P = 0.20)" in text
    assert "- u3 (missing_column, P = 0.70)" in text


def test_load_split_tags_heldout_language(tmp_path):
    from datacopilot.questions import write_jsonl
    write_jsonl(tmp_path / "human_heldout.jsonl", [{"id": "human-001", "split": "heldout", "language": "hinglish",
                                                    "dataset": "crop", "question": "x?", "answerable": True}])
    assert eg.load_split("heldout", tmp_path)[0]["question_set"] == "heldout_hinglish"
