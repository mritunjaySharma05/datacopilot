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
