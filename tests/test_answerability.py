"""Tests for datacopilot.pipeline.answerability on small synthetic feature tables."""
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from datacopilot.pipeline import answerability as ans
from datacopilot.pipeline.features import FEATURE_NAMES, DatasetContext

TINY_GRIDS = {"logreg": {"clf__C": [0.1, 1.0]}, "xgboost": {"n_estimators": [20], "max_depth": [2]}}


def synthetic(n: int = 60, seed: int = 0) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Features where schema_overlap separates the classes; ids pair each Hinglish row with its source."""
    rng = np.random.default_rng(seed)
    y = pd.Series([1, 0] * (n // 2))
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURE_NAMES))), columns=list(FEATURE_NAMES))
    X["schema_overlap"] = y * 2.0 + rng.normal(scale=0.3, size=n)
    ids = pd.Series([f"q{i // 2}{'-hi' if i % 4 >= 2 else ''}" for i in range(n)])
    return X, y, ids


@pytest.mark.parametrize("name", ["logreg", "random_forest", "xgboost"])
def test_make_estimator_fits_and_predicts_probabilities(name):
    X, y, _ = synthetic()
    model = ans.make_estimator(name).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2) and np.allclose(proba.sum(axis=1), 1)


def test_make_estimator_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown model"):
        ans.make_estimator("svm")


def test_param_grids_match_estimator_params():
    for name, grid in ans.PARAM_GRIDS.items():
        assert set(grid) <= set(ans.make_estimator(name).get_params())


def test_cv_groups_join_hinglish_rows_to_their_source():
    groups = ans.cv_groups(pd.Series(["olist-001", "olist-001-hi", "olist-u004-hi"]))
    assert groups.tolist() == ["olist-001", "olist-001", "olist-u004"]


def test_cross_validate_keeps_groups_in_one_fold(monkeypatch):
    X, y, ids = synthetic()
    groups = ans.cv_groups(ids)
    seen = []
    real_split = ans.StratifiedGroupKFold.split

    def spy(self, X_, y_, g):
        for tr, va in real_split(self, X_, y_, g):
            seen.append((set(g.iloc[tr]), set(g.iloc[va])))
            yield tr, va

    monkeypatch.setattr(ans.StratifiedGroupKFold, "split", spy)
    scores = ans.cross_validate(ans.make_estimator("logreg"), X, y, groups, n_folds=3)
    assert len(scores) == 3 and all(0 <= s <= 1 for s in scores)
    assert all(not (tr & va) for tr, va in seen)


def test_search_scores_every_config_best_first():
    X, y, ids = synthetic()
    results = ans.search(X, y, ans.cv_groups(ids), TINY_GRIDS, n_folds=3)
    assert len(results) == 3
    assert [r.mean for r in results] == sorted((r.mean for r in results), reverse=True)
    assert results[0].mean > 0.9  # schema_overlap separates the classes
    assert {r.model for r in results} == {"logreg", "xgboost"}


def test_fit_best_uses_chosen_params():
    X, y, _ = synthetic()
    best = ans.CVResult("logreg", {"clf__C": 0.5}, 0.9, 0.0)
    model = ans.fit_best(X, y, best)
    assert model.get_params()["clf__C"] == 0.5


def test_evaluate_counts_unanswerable_as_positive_class():
    meta = pd.DataFrame({
        "question_set": ["csv", "csv", "unanswerable", "unanswerable", "hinglish_unanswerable"],
        "dataset": ["crop", "f1", "crop", "f1", "f1"],
        "unanswerable_type": [None, None, "subjective", "missing_column", "missing_column"],
    }, index=[10, 11, 12, 13, 14])
    y_true = pd.Series([1, 1, 0, 0, 0], index=meta.index)
    y_pred = [1, 0, 0, 1, 0]
    m = ans.evaluate(y_true, y_pred, meta)
    assert m["n"] == 5 and m["accuracy"] == pytest.approx(0.6)
    assert m["unanswerable_precision"] == pytest.approx(2 / 3)
    assert m["unanswerable_recall"] == pytest.approx(2 / 3)
    assert m["confusion_matrix"] == [[2, 1], [1, 1]]
    assert m["recall_by_unanswerable_type"] == {"missing_column": 0.5, "subjective": 1.0}
    assert m["accuracy_by_question_set"]["csv"] == 0.5
    assert m["accuracy_by_dataset"] == {"crop": 1.0, "f1": pytest.approx(1 / 3, abs=1e-4)}


def test_feature_weights_for_logreg_and_trees():
    X, y, _ = synthetic()
    for name in ("logreg", "random_forest"):
        model = ans.make_estimator(name).fit(X, y)
        weights = ans.feature_weights(model, list(FEATURE_NAMES))
        assert weights.index[0] == "schema_overlap"
        assert len(weights) == len(FEATURE_NAMES)
    assert isinstance(ans.make_estimator("logreg"), Pipeline)


def test_save_load_round_trip(tmp_path):
    X, y, _ = synthetic()
    model = ans.make_estimator("logreg").fit(X, y)
    path = ans.save_model(model, list(FEATURE_NAMES), {"name": "logreg"}, tmp_path / "m" / "a.joblib")
    bundle = ans.load_model(path)
    assert bundle["name"] == "logreg" and bundle["feature_names"] == list(FEATURE_NAMES)
    assert bundle["threshold"] == ans.THRESHOLD
    assert (bundle["model"].predict(X) == model.predict(X)).all()


def test_load_model_missing_file_says_how_to_train(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_answerability"):
        ans.load_model(tmp_path / "nope.joblib")


def test_predict_answerable_uses_bundle_features_and_threshold():
    X, y, _ = synthetic()
    cols = ["schema_overlap", "starts_with_write_verb"]
    model = ans.make_estimator("logreg").fit(X[cols], y)
    schema = {"crop_production": [{"name": "crop", "type": "VARCHAR", "samples": ["Rice"]},
                                  {"name": "production", "type": "DOUBLE", "samples": [1.0]}]}
    ctx = DatasetContext(1997, 2015, frozenset({"rice"}))
    bundle = {"model": model, "feature_names": cols, "threshold": 0.5}
    ok, prob = ans.predict_answerable("What is the production of each crop?", schema, ctx, bundle)
    assert ok is True and 0.5 <= prob <= 1
    never = {**bundle, "threshold": 1.01}
    assert ans.predict_answerable("What is the production of each crop?", schema, ctx, never)[0] is False
