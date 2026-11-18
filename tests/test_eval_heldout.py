"""Tests for scripts/eval_heldout.py on synthetic feature tables."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from datacopilot.pipeline.features import FEATURE_NAMES, DatasetContext

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("eval_heldout", SCRIPTS / "eval_heldout.py")
eh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eh)

GRID = {"logreg": {"clf__C": [1.0]}}
SCHEMA = {"crop_production": [{"name": "crop", "type": "VARCHAR", "samples": ["Rice"]},
                              {"name": "production", "type": "DOUBLE", "samples": [1.0]}]}


def table(n: int, seed: int, split: str, length_leak: bool = False) -> pd.DataFrame:
    """Rows where schema_overlap separates the classes; optionally n_words leaks the label too."""
    rng = np.random.default_rng(seed)
    y = np.array([1, 0] * (n // 2))
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURE_NAMES))), columns=list(FEATURE_NAMES))
    df["schema_overlap"] = y * 2 + rng.normal(scale=0.3, size=n)
    if length_leak:
        df["n_words"] = y * 3
    return df.assign(id=[f"q{i}" for i in range(n)], dataset="crop", question_set="csv", split=split,
                     answerable=y, unanswerable_type=[None if v else "subjective" for v in y],
                     question=[f"question {i}" for i in range(n)])


def test_heldout_table_tags_language_and_keeps_question():
    rows = [{"id": "human-001", "dataset": "crop", "question": "2010 me rice kitna hua", "language": "hinglish",
             "split": "heldout", "answerable": True, "unanswerable_type": None},
            {"id": "human-002", "dataset": "crop", "question": "Delete all rows.", "language": "en",
             "split": "heldout", "answerable": False, "unanswerable_type": "write_request"}]
    t = eh.heldout_table(rows, {"crop": (SCHEMA, DatasetContext(1997, 2015, frozenset({"rice"})))})
    assert t.question_set.tolist() == ["heldout_hinglish", "heldout_en"]
    assert t.question.tolist() == [r["question"] for r in rows]
    assert t.answerable.tolist() == [1, 0] and set(FEATURE_NAMES) <= set(t.columns)


def test_fit_logreg_returns_bundle_with_feature_subset():
    features = [f for f in FEATURE_NAMES if f != "n_words"]
    bundle = eh.fit_logreg(table(60, 0, "train"), features, GRID, n_folds=3)
    assert bundle["feature_names"] == features and bundle["params"] == {"clf__C": 1.0}
    assert 0 <= bundle["cv_macro_f1"] <= 1


def test_score_lists_misclassified_rows_with_probability():
    train = table(60, 0, "train")
    bundle = eh.fit_logreg(train, list(FEATURE_NAMES), GRID, n_folds=3)
    held = table(10, 1, "heldout")
    held.loc[0, "answerable"] = 0  # flip one label so at least this row is wrong
    metrics, wrong = eh.score(bundle, held)
    assert metrics["n"] == 10
    assert "q0" in wrong.id.tolist()
    assert {"question", "p_answerable"} <= set(wrong.columns)
    assert len(wrong) == metrics["confusion_matrix"][0][1] + metrics["confusion_matrix"][1][0]


def test_run_ablation_exposes_length_shortcut():
    """Trained where n_words leaks the label, the no-n_words model does better once the leak is gone."""
    train, test = table(80, 0, "train", length_leak=True), table(20, 1, "test", length_leak=True)
    held = table(20, 2, "heldout")
    held["n_words"] = (1 - held.answerable) * 3  # reversed: short answerable, long unanswerable
    main = eh.fit_logreg(train, list(FEATURE_NAMES), GRID, n_folds=3)
    results = eh.run(train, test, held, main, GRID, n_folds=3)
    assert set(results) == {"main", "logreg", "logreg_no_nwords"}
    assert results["main"]["heldout"] == results["logreg"]["heldout"]
    assert results["logreg_no_nwords"]["heldout"]["accuracy"] > results["logreg"]["heldout"]["accuracy"]


def test_report_lists_every_model_and_misclassified_question():
    train, test, held = table(60, 0, "train"), table(20, 1, "test"), table(10, 2, "heldout")
    held.loc[0, "answerable"] = 0
    results = eh.run(train, test, held, eh.fit_logreg(train, list(FEATURE_NAMES), GRID, 3), GRID, n_folds=3)
    text = eh.report(results, len(held))
    for name in ("main", "logreg", "logreg_no_nwords"):
        assert f"### {name} (" in text
    assert "| q0 | question 0 |" in text
    assert "Held-out accuracy by language" in text
