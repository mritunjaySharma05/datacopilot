"""Tests for scripts/make_figures.py on hand-made results (no real runs needed)."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("make_figures", SCRIPTS / "make_figures.py")
mf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mf)

METRICS = {"logreg": {"test": {"n": 180, "accuracy": 0.9}, "heldout": {"n": 30, "accuracy": 0.6}},
           "main": {"test": {"n": 180, "accuracy": 0.85}, "heldout": {"n": 30, "accuracy": 0.8}}}
GATE = pd.DataFrame({"id": ["a1", "a2", "u1", "u2"], "answerable": [True, True, False, False],
                     "gate_pass": [True, False, False, True], "strict": [True, True, False, False],
                     "answered": [False, False, True, True],
                     "correct_no_gate_strict": [True, True, False, False],
                     "correct_gate_strict": [True, False, True, False],
                     "correct_no_gate_relaxed": [True, True, False, False],
                     "correct_gate_relaxed": [True, False, True, False]})
OURS = pd.DataFrame({"id": ["c1", "c2", "f1"], "dataset": ["crop", "crop", "f1"],
                     "strict": [True, False, False], "relaxed": [True, True, False]})
PAIRED = pd.DataFrame({"question_id": [1, 2, 3], "difficulty": ["simple", "challenging", "simple"],
                       "correct_v": [True, True, False], "correct_b": [True, False, False]})


def test_answerability_data():
    t = mf.answerability_data(METRICS)
    assert list(t.index) == ["Test (180)", "Held-out (30)"]
    assert t.loc["Held-out (30)", "Old: logreg with n_words"] == 0.6
    assert t.loc["Test (180)", "Deployed: XGBoost, no n_words"] == 0.85


def test_gate_data():
    d = mf.gate_data(GATE)
    assert d["n"] == 4 and d["n_unanswerable"] == 2
    assert d["accuracy"] == {"Without gate": 0.5, "With gate": 0.5}
    assert d["invented"] == {"Without gate": 2, "With gate": 1}


def test_our_sets_data():
    t = mf.our_sets_data(OURS)
    assert list(t.index) == ["crop", "f1", "all"]
    assert t.loc["crop", "Strict"] == 0.5 and t.loc["crop", "Relaxed"] == 1.0
    assert t.loc["all", "n"] == 3 and t.loc["all", "Relaxed"] == pytest.approx(2 / 3)


def test_bird_data():
    t = mf.bird_data(PAIRED)
    assert list(t.index) == ["simple", "challenging", "all"]
    assert t.loc["challenging", "Validated"] == 1.0 and t.loc["challenging", "Baseline"] == 0.0


def test_plots_write_pngs(tmp_path):
    paths = [mf.plot_answerability(mf.answerability_data(METRICS), tmp_path / "a.png"),
             mf.plot_gate(mf.gate_data(GATE), tmp_path / "g.png"),
             mf.plot_our_sets(mf.our_sets_data(OURS), tmp_path / "o.png"),
             mf.plot_bird(mf.bird_data(PAIRED), tmp_path / "b.png")]
    for p in paths:
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# ---------- classifier figures ----------

class Fixed:
    """Fake model: P(answerable) taken from the table's 'p' column."""

    def __init__(self, table):
        self.table = table

    def predict_proba(self, X):
        import numpy as np
        p = self.table.loc[X.index, "p"].to_numpy()
        return np.column_stack([1 - p, p])


FEATS = pd.DataFrame({"split": ["train", "test", "test", "test", "test"], "answerable": [1, 1, 1, 0, 0],
                      "f": [0, 1, 2, 3, 4], "p": [0.5, 0.9, 0.4, 0.3, 0.6]})
BUNDLE = {"model": Fixed(FEATS), "feature_names": ["f"], "threshold": 0.5}
FAMILIES = {"xgboost": {"cv_macro_f1": 0.88, "cv_std": 0.03, "test_macro_f1": 0.87},
            "logreg": {"cv_macro_f1": 0.86, "cv_std": 0.05, "test_macro_f1": 0.9}}


def test_confusion_data_labels_rows_true_columns_predicted():
    t = mf.confusion_data({"confusion_matrix": [[79, 11], [12, 78]]})
    assert t.loc["Unanswerable", "Answerable"] == 11 and t.loc["Answerable", "Unanswerable"] == 12


def test_test_scores_use_test_rows_and_unanswerable_as_positive():
    y, s = mf.test_scores(BUNDLE, FEATS)
    assert y.tolist() == [0, 0, 1, 1]
    assert s.tolist() == pytest.approx([0.1, 0.6, 0.7, 0.4])
    with pytest.raises(ValueError):
        mf.test_scores(BUNDLE, FEATS[FEATS.split == "train"])


def test_curve_data_auc_ap_and_operating_point():
    c = mf.curve_data(*mf.test_scores(BUNDLE, FEATS))
    assert c["auc"] == pytest.approx(0.75)  # 3 of 4 (unanswerable, answerable) pairs ranked right
    assert 0 < c["ap"] <= 1 and c["base_rate"] == 0.5
    # refused (P < 0.5): rows with p 0.4 (answerable) and 0.3 (unanswerable)
    assert c["point"] == {"fpr": 0.5, "tpr": 0.5, "precision": 0.5}


def test_importance_data_top_n_by_absolute_value():
    w = mf.importance_data({"a": 0.1, "b": -0.5, "c": 0.3}, top=2)
    assert w.index.tolist() == ["b", "c"]


def test_family_data_order_and_columns():
    t = mf.family_data(FAMILIES)
    assert t.index.tolist() == ["Logistic regression", "XGBoost"]
    assert t.loc["XGBoost", "Test macro F1"] == 0.87 and t.loc["Logistic regression", "cv_std"] == 0.05


def test_classifier_plots_write_pngs(tmp_path):
    paths = [mf.plot_confusion(mf.confusion_data({"confusion_matrix": [[3, 1], [1, 3]]}), tmp_path / "c.png", "t"),
             mf.plot_curves(mf.curve_data(*mf.test_scores(BUNDLE, FEATS)), tmp_path / "r.png"),
             mf.plot_importance(mf.importance_data({"a": 0.2, "b": 0.1}), tmp_path / "i.png"),
             mf.plot_families(mf.family_data(FAMILIES), tmp_path / "f.png")]
    for p in paths:
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_main_skips_figures_whose_runs_are_missing(tmp_path, capsys):
    assert mf.main(["--results-dir", str(tmp_path / "no_runs"), "--out-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Skipped gate" in out and "Skipped BIRD" in out
    assert {p.name for p in tmp_path.glob("*.png")} == {"answerability.png", "confusion_matrix.png", "roc_pr.png",
                                                       "feature_importance.png", "model_families.png"}
