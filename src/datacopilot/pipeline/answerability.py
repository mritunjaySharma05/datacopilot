"""Answerability classifier: can this question be answered from this dataset's schema?

Trained by scripts/train_answerability.py on data/features/answerability_features.csv
(split == "train" rows only). Label 1 = answerable, 0 = unanswerable.

Model selection uses grouped, stratified 5-fold CV: a Hinglish question and its English
source share a group, so a translation never sits in a different fold than its original.
"""
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from datacopilot.pipeline.features import FEATURE_NAMES, DatasetContext, extract_features

ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / "models" / "answerability.joblib"
SEED = 42
N_FOLDS = 5
THRESHOLD = 0.5  # P(answerable) below this means "refuse"
# Left out of the deployed model (decision 2026-11-18): question length was a writing-style
# shortcut in our authored sets; on the human held-out set logreg scored 66.7% with it vs 83.3%
# without (docs/results/README.md).
EXCLUDED_FEATURES = ("n_words",)
DEPLOYED_FEATURES = tuple(f for f in FEATURE_NAMES if f not in EXCLUDED_FEATURES)


def make_estimator(name: str, seed: int = SEED):
    """Unfitted estimator for a candidate model name."""
    if name == "logreg":
        return Pipeline([("scale", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=5000, random_state=seed))])
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1)
    if name == "xgboost":
        return XGBClassifier(eval_metric="logloss", random_state=seed, n_jobs=1, verbosity=0)
    raise ValueError(f"unknown model {name!r}")


# Small grids: 420 training rows do not support a wide search.
PARAM_GRIDS = {
    "logreg": {"clf__C": [0.01, 0.1, 1.0, 10.0]},
    "random_forest": {"max_depth": [None, 5, 10], "min_samples_leaf": [1, 3, 5]},
    "xgboost": {"n_estimators": [100, 300], "max_depth": [2, 3, 5], "learning_rate": [0.05, 0.1]},
}


def cv_groups(ids: pd.Series) -> pd.Series:
    """Group id per question: a Hinglish row ("x-hi") joins its English source ("x")."""
    return ids.str.removesuffix("-hi")


@dataclass
class CVResult:
    """Cross-validated score of one model configuration."""

    model: str
    params: dict
    mean: float
    std: float
    fold_scores: list[float] = field(default_factory=list)


def cross_validate(estimator, X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                   n_folds: int = N_FOLDS, seed: int = SEED) -> list[float]:
    """Macro F1 on each fold of a grouped, stratified, shuffled K-fold split."""
    folds = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = []
    for train_idx, val_idx in folds.split(X, y, groups):
        model = clone(estimator).fit(X.iloc[train_idx], y.iloc[train_idx])
        scores.append(f1_score(y.iloc[val_idx], model.predict(X.iloc[val_idx]), average="macro"))
    return scores


def search(X: pd.DataFrame, y: pd.Series, groups: pd.Series, grids: dict[str, dict] | None = None,
           n_folds: int = N_FOLDS, seed: int = SEED) -> list[CVResult]:
    """CV score for every (model, params) in the grids, best first.

    Ties keep grid order, so the simpler model listed first wins.
    """
    results = []
    for name, grid in (grids or PARAM_GRIDS).items():
        for params in ParameterGrid(grid):
            scores = cross_validate(make_estimator(name, seed).set_params(**params), X, y, groups, n_folds, seed)
            results.append(CVResult(name, params, float(np.mean(scores)), float(np.std(scores)), scores))
    return sorted(results, key=lambda r: -r.mean)


def fit_best(X: pd.DataFrame, y: pd.Series, best: CVResult, seed: int = SEED):
    """Refit the chosen configuration on all training rows."""
    return make_estimator(best.model, seed).set_params(**best.params).fit(X, y)


def evaluate(y_true: pd.Series, y_pred, meta: pd.DataFrame) -> dict:
    """Test metrics. The 'positive' class for precision/recall is unanswerable (0).

    `meta` must have columns question_set, dataset, unanswerable_type, aligned with y_true.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0], zero_division=0)
    correct = pd.Series(y_true == y_pred, index=meta.index)
    unans = meta.unanswerable_type.notna() & (y_true == 0)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "unanswerable_precision": float(p[0]),
        "unanswerable_recall": float(r[0]),
        "unanswerable_f1": float(f[0]),
        # rows = true [unanswerable, answerable], columns = predicted [unanswerable, answerable]
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "accuracy_by_question_set": correct.groupby(meta.question_set).mean().round(4).to_dict(),
        "accuracy_by_dataset": correct.groupby(meta.dataset).mean().round(4).to_dict(),
        "recall_by_unanswerable_type": correct[unans].groupby(meta.unanswerable_type[unans]).mean().round(4).to_dict(),
    }


def feature_weights(model, feature_names: list[str]) -> pd.Series:
    """Logistic-regression coefficients (on standardized features) or tree importances."""
    if isinstance(model, Pipeline):
        values = model.named_steps["clf"].coef_[0]
    else:
        values = model.feature_importances_
    return pd.Series(values, index=feature_names).sort_values(key=abs, ascending=False)


def save_model(model, feature_names: list[str], info: dict, path: Path = MODEL_PATH) -> Path:
    """Save the fitted model with the feature order it expects and training info."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": list(feature_names), "threshold": THRESHOLD, **info}, path)
    return path


def load_model(path: Path = MODEL_PATH) -> dict:
    """Load a bundle written by save_model."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/train_answerability.py first")
    return joblib.load(path)


def predict_answerable(question: str, schema: dict[str, list[dict]], context: DatasetContext | None = None,
                       bundle: dict | None = None) -> tuple[bool, float]:
    """(answerable?, P(answerable)) for one question against one dataset's schema."""
    bundle = bundle or load_model()
    feats = extract_features(question, schema, context)
    X = pd.DataFrame([[feats[n] for n in bundle["feature_names"]]], columns=bundle["feature_names"])
    prob = float(bundle["model"].predict_proba(X)[0, 1])
    return bool(prob >= bundle.get("threshold", THRESHOLD)), prob

