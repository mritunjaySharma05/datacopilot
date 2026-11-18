"""Tests for scripts/train_answerability.py on a small synthetic feature table."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from datacopilot.pipeline.answerability import load_model
from datacopilot.pipeline.features import FEATURE_NAMES

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_answerability.py"
spec = importlib.util.spec_from_file_location("train_answerability", SCRIPT)
ta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ta)

TINY_GRIDS = {"logreg": {"clf__C": [1.0]}, "random_forest": {"max_depth": [3], "min_samples_leaf": [1]}}
SETS = {(1, False): "csv", (0, False): "unanswerable", (1, True): "hinglish", (0, True): "hinglish_unanswerable"}
TYPES = ["missing_column", "subjective"]


def feature_table(n: int = 80, seed: int = 0) -> pd.DataFrame:
    """Rows cycle answerable/unanswerable and English/Hinglish; 25% test."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        label, hi = (i + 1) % 2, (i // 2) % 2 == 1
        src = f"q{i // 4}-{label}"
        rows.append({"id": src + ("-hi" if hi else ""), "dataset": ["crop", "f1"][i % 3 % 2],
                     "question_set": SETS[(label, hi)], "split": "test" if (i // 4) % 4 == 0 else "train",
                     "answerable": label, "unanswerable_type": None if label else TYPES[(i // 4) % 2],
                     **{f: rng.normal() for f in FEATURE_NAMES}})
        rows[-1]["schema_overlap"] = label * 2 + rng.normal(scale=0.3)
    return pd.DataFrame(rows)


def test_run_experiment_trains_on_train_rows_only():
    table = feature_table()
    r = ta.run_experiment(table, drop=[], train_sets=None, grids=TINY_GRIDS, n_folds=3)
    assert r["n_train"] == (table.split == "train").sum()
    assert r["test"]["n"] == (table.split == "test").sum()
    assert set(r["per_family"]) == {"logreg", "random_forest"}
    assert r["test"]["accuracy"] > 0.8


def test_run_experiment_drop_and_english_only():
    table = feature_table()
    r = ta.run_experiment(table, drop=["n_words"], train_sets=ta.ENGLISH_SETS, grids=TINY_GRIDS, n_folds=3)
    assert "n_words" not in r["features"] and len(r["features"]) == len(FEATURE_NAMES) - 1
    english_train = table[(table.split == "train") & table.question_set.isin(ta.ENGLISH_SETS)]
    assert r["n_train"] == len(english_train)
    assert set(r["test"]["accuracy_by_question_set"]) == set(SETS.values())  # tested on all sets


def test_train_writes_model_report_and_metrics(tmp_path):
    features = tmp_path / "features.csv"
    feature_table().to_csv(features, index=False)
    model_out = tmp_path / "models" / "a.joblib"
    assert ta.train(features, tmp_path / "results", model_out, grids=TINY_GRIDS, n_folds=3) == 0
    bundle = load_model(model_out)
    assert bundle["feature_names"] == [f for f in FEATURE_NAMES if f != "n_words"] and bundle["n_train"] == 60
    metrics = json.loads((tmp_path / "results" / "answerability_metrics.json").read_text())
    assert set(metrics) == set(ta.EXPERIMENTS)
    text = (tmp_path / "results" / "answerability.md").read_text()
    for heading in ("Experiments (test split)", "confusion matrix", "Recall per unanswerable type"):
        assert heading in text
    assert "| main |" in text and "| with_n_words |" in text and "| english_only |" in text


def test_pct_formats_percentage():
    assert ta.pct(0.9061) == "90.6%"
