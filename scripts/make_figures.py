"""Make the Module 4 report figures (PNG, 150 dpi) in docs/results/figures/.

Numbers come from the stored results, never typed in by hand:
- answerability: docs/results/heldout_metrics.json (old logreg with n_words vs deployed model)
- gate: newest data/results/gate_test_*.csv (scored with eval_gate.summary)
- our sets: newest data/results/questions_validated_csv+hinglish_test_*_rescored.csv
- BIRD: newest bird_{validated,baseline}_sample300_seed42_*.csv (paired with compare_bird)
- classifier: docs/results/answerability_metrics.json (confusion matrix, importances, CV vs test
  per model family) and models/answerability.joblib scored on the test rows of
  data/features/answerability_features.csv (ROC and precision-recall curves; test split only)

Usage:
    python scripts/make_figures.py
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import compare_bird  # noqa: E402
import eval_gate  # noqa: E402
from datacopilot.pipeline.answerability import MODEL_PATH, load_model  # noqa: E402

RESULTS_DOCS = ROOT / "docs" / "results"
RESULTS_DIR = ROOT / "data" / "results"
OUT_DIR = RESULTS_DOCS / "figures"
FEATURES_PATH = ROOT / "data" / "features" / "answerability_features.csv"
FAMILY_LABELS = {"logreg": "Logistic regression", "random_forest": "Random forest", "xgboost": "XGBoost"}
DPI = 150

# Validated categorical slots 1-2 (blue, orange) plus recessive ink for text and grid.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SURFACE = "#ffffff"


def answerability_data(metrics: dict) -> pd.DataFrame:
    """Accuracy on test and held-out for the old model (logreg + n_words) and the deployed model."""
    models = {"Old: logreg with n_words": "logreg", "Deployed: XGBoost, no n_words": "main"}
    return pd.DataFrame({label: {f"Test ({metrics[k]['test']['n']})": metrics[k]["test"]["accuracy"],
                                 f"Held-out ({metrics[k]['heldout']['n']})": metrics[k]["heldout"]["accuracy"]}
                         for label, k in models.items()})


def gate_data(combined: pd.DataFrame) -> dict:
    """End-to-end strict accuracy and invented answers, without vs with the gate."""
    s = eval_gate.summary(combined)
    return {"n": s["n"], "n_unanswerable": s["n_unanswerable"],
            "accuracy": {"Without gate": s["acc_no_gate_strict"], "With gate": s["acc_gate_strict"]},
            "invented": {"Without gate": s["unans_answered_no_gate"], "With gate": s["unans_answered_gate"]}}


def our_sets_data(run: pd.DataFrame) -> pd.DataFrame:
    """Strict and relaxed accuracy by dataset (plus 'all') for one eval_questions run."""
    t = run.groupby("dataset").agg(n=("id", "size"), Strict=("strict", "mean"), Relaxed=("relaxed", "mean"))
    t.loc["all"] = [len(run), run.strict.mean(), run.relaxed.mean()]
    t["n"] = t.n.astype(int)
    return t


def bird_data(paired: pd.DataFrame) -> pd.DataFrame:
    """Validated and baseline BIRD accuracy by difficulty (plus 'all')."""
    t = compare_bird.accuracy_by(paired, "difficulty", compare_bird.DIFFICULTY_ORDER)
    return t.rename(columns={"validated": "Validated", "baseline": "Baseline"})


def _style(ax, ylabel: str, ymax: float = 100) -> None:
    ax.set_ylim(0, ymax)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)
    ax.tick_params(colors=INK_2, length=0)
    for label in ax.get_xticklabels():
        label.set_color(INK)


def grouped_bars(ax, table: pd.DataFrame, colors: list[str], fmt: str = "{:.1f}%", scale: float = 100,
                 errors: dict[str, pd.Series] | None = None) -> None:
    """Grouped bars: one group per row, one bar per column, value labels on top (above error bars if given)."""
    n_series = len(table.columns)
    width = 0.8 / n_series
    for i, (col, color) in enumerate(zip(table.columns, colors)):
        xs = [g + (i - (n_series - 1) / 2) * width for g in range(len(table))]
        vals = [scale * v for v in table[col]]
        bars = ax.bar(xs, vals, width=width, color=color, label=col, edgecolor=SURFACE, linewidth=2)
        err = [scale * e for e in errors[col]] if errors and col in errors else [0] * len(vals)
        if any(err):
            ax.errorbar(xs, vals, yerr=err, fmt="none", ecolor=INK_2, elinewidth=1, capsize=3)
        for b, v, e in zip(bars, vals, err):
            ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height() + e), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(range(len(table)))
    ax.set_xticklabels(table.index)


def _figure(width: float = 7.0, height: float = 4.2, ncols: int = 1):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), facecolor=SURFACE)
    return fig, axes


def _finish(fig, ax, title: str, path: Path, legend_cols: int = 2) -> Path:
    fig.suptitle(title, x=0.02, ha="left", fontsize=12, color=INK, fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.02, 0.945), ncol=legend_cols,
                   frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_answerability(table: pd.DataFrame, path: Path) -> Path:
    fig, ax = _figure(6.4, 4.2)
    grouped_bars(ax, table, [ORANGE, BLUE])
    _style(ax, "Accuracy (%)")
    return _finish(fig, ax, "Answerability classifier: test vs human held-out accuracy", path)


def plot_gate(data: dict, path: Path) -> Path:
    fig, (a1, a2) = _figure(8.0, 4.2, ncols=2)
    colors = [ORANGE, BLUE]
    for ax, key, ylabel, fmt, scale, ymax, sub in (
            (a1, "accuracy", "Accuracy (%)", "{:.1f}%", 100, 100,
             f"End-to-end strict accuracy (n={data['n']})"),
            (a2, "invented", "Questions", "{:.0f}", 1, data["n_unanswerable"],
             f"Unanswerable, given an answer (of {data['n_unanswerable']})")):
        names = list(data[key])
        bars = ax.bar(names, [scale * data[key][k] for k in names], width=0.5, color=colors,
                      edgecolor=SURFACE, linewidth=2)
        for b, k in zip(bars, names):
            ax.annotate(fmt.format(scale * data[key][k]), (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, color=INK)
        _style(ax, ylabel, ymax)
        ax.set_title(sub, fontsize=10, color=INK_2, loc="left")
    fig.suptitle("Answerability gate on the test split", x=0.02, ha="left", fontsize=12, color=INK,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_our_sets(table: pd.DataFrame, path: Path) -> Path:
    t = table[["Strict", "Relaxed"]].copy()
    t.index = [f"{d}\n(n={n})" for d, n in zip(table.index, table.n)]
    fig, ax = _figure(7.0, 4.2)
    grouped_bars(ax, t, [BLUE, ORANGE])
    _style(ax, "Accuracy (%)")
    return _finish(fig, ax, "Our question sets, test split, validated mode: accuracy by dataset", path)


def plot_bird(table: pd.DataFrame, path: Path) -> Path:
    t = table[["Validated", "Baseline"]].copy()
    t.index = [f"{d}\n(n={n})" for d, n in zip(table.index, table.n)]
    fig, ax = _figure(7.0, 4.2)
    grouped_bars(ax, t, [BLUE, ORANGE])
    _style(ax, "Execution accuracy (%)")
    return _finish(fig, ax, "BIRD dev, 300-question sample: accuracy by difficulty", path)


# ---------- answerability classifier (test split only; the held-out set is never read here) ----------

def confusion_data(test_metrics: dict) -> pd.DataFrame:
    """Confusion matrix as a labelled table: rows = true class, columns = predicted class."""
    labels = ["Unanswerable", "Answerable"]
    return pd.DataFrame(test_metrics["confusion_matrix"], index=labels, columns=labels)


def test_scores(bundle: dict, table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(y, score) on the test rows: y = 1 for unanswerable, score = 1 - P(answerable)."""
    test = table[table.split == "test"]
    if test.empty:
        raise ValueError("no test rows")
    prob = bundle["model"].predict_proba(test[bundle["feature_names"]])[:, 1]
    return (1 - test.answerable.to_numpy()).astype(int), 1 - prob


def curve_data(y: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict:
    """ROC and precision-recall curves, AUC, average precision and the operating point at the threshold."""
    fpr, tpr, _ = roc_curve(y, score)
    precision, recall, _ = precision_recall_curve(y, score)
    pred = score > 1 - threshold  # refuse when P(answerable) < threshold
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    return {"fpr": fpr, "tpr": tpr, "auc": float(roc_auc_score(y, score)),
            "precision": precision, "recall": recall, "ap": float(average_precision_score(y, score)),
            "base_rate": float(np.mean(y)),
            "point": {"fpr": fp / max(int((y == 0).sum()), 1), "tpr": tp / max(int((y == 1).sum()), 1),
                      "precision": tp / max(tp + fp, 1)}}


def importance_data(weights: dict, top: int = 15) -> pd.Series:
    """Top-N features by absolute weight, largest first."""
    w = pd.Series(weights, dtype=float)
    return w.reindex(w.abs().sort_values(ascending=False).index)[:top]


def family_data(per_family: dict) -> pd.DataFrame:
    """CV macro F1 (with std) and test macro F1 for each model family's best configuration."""
    rows = {FAMILY_LABELS.get(k, k): {"CV macro F1 (train, 5-fold, ± 1 SD)": v["cv_macro_f1"], "cv_std": v["cv_std"],
                                      "Test macro F1": v["test_macro_f1"]}
            for k, v in per_family.items()}
    order = [FAMILY_LABELS[k] for k in FAMILY_LABELS if FAMILY_LABELS[k] in rows]
    return pd.DataFrame(rows).T.reindex(order)


def plot_confusion(table: pd.DataFrame, path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 4.4), facecolor=SURFACE)
    values = table.to_numpy()
    ax.imshow(values, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#eef4fc", BLUE]),
              vmin=0, vmax=values.sum(axis=1).max())
    for (i, j), v in np.ndenumerate(values):
        share = v / values[i].sum()
        ax.text(j, i, f"{v}\n({100 * share:.1f}% of row)", ha="center", va="center", fontsize=10,
                color=SURFACE if share > 0.5 else INK)
    ax.set_xticks([0, 1], table.columns)
    ax.set_yticks([0, 1], table.index)
    ax.set_xlabel("Predicted", color=INK_2)
    ax.set_ylabel("True", color=INK_2)
    ax.tick_params(length=0, colors=INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.suptitle(title, x=0.02, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, path)


def plot_curves(curves: dict, path: Path, label: str = "Deployed XGBoost") -> Path:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 4.4), facecolor=SURFACE)
    pt = curves["point"]
    a1.plot([0, 1], [0, 1], color=INK_2, linewidth=1, linestyle="--", label="Chance (AUC = 0.500)")
    a1.plot(curves["fpr"], curves["tpr"], color=BLUE, linewidth=2, label=f"{label} (AUC = {curves['auc']:.3f})")
    a1.plot(pt["fpr"], pt["tpr"], "o", color=BLUE, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
            label="Threshold 0.5")
    a1.set_xlabel("False positive rate (answerable refused)", color=INK_2)
    a1.set_ylabel("True positive rate (unanswerable refused)", color=INK_2)
    a1.set_title("ROC curve", fontsize=10, color=INK_2, loc="left")
    base = curves["base_rate"]
    a2.axhline(base, color=INK_2, linewidth=1, linestyle="--", label=f"Chance (AP = {base:.3f})")
    a2.plot(curves["recall"], curves["precision"], color=BLUE, linewidth=2,
            label=f"{label} (AP = {curves['ap']:.3f})")
    a2.plot(pt["tpr"], pt["precision"], "o", color=BLUE, markersize=8, markeredgecolor=SURFACE,
            markeredgewidth=2, label="Threshold 0.5")
    a2.set_xlabel("Recall (unanswerable)", color=INK_2)
    a2.set_ylabel("Precision (unanswerable)", color=INK_2)
    a2.set_title("Precision-recall curve", fontsize=10, color=INK_2, loc="left")
    for ax in (a1, a2):
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(0, 1.02)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(INK_2)
        ax.spines["bottom"].set_color(INK_2)
        ax.tick_params(colors=INK_2, length=0)
        ax.legend(loc="lower right" if ax is a1 else "lower left", frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("Answerability classifier, test split (positive class = unanswerable)", x=0.02, ha="left",
                 fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, path)


def plot_importance(weights: pd.Series, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 5.2), facecolor=SURFACE)
    w = weights[::-1]
    bars = ax.barh(range(len(w)), w.to_numpy(), height=0.7, color=BLUE, edgecolor=SURFACE, linewidth=2)
    for b, v in zip(bars, w):
        ax.annotate(f"{v:.3f}", (b.get_width(), b.get_y() + b.get_height() / 2), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(w)), w.index)
    ax.set_xlim(0, w.max() * 1.15)
    ax.set_xlabel("XGBoost importance (gain share)", color=INK_2)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(INK_2)
    ax.tick_params(colors=INK_2, length=0)
    for label in ax.get_yticklabels():
        label.set_color(INK)
    fig.suptitle(f"Deployed model: top {len(w)} features by importance", x=0.02, ha="left", fontsize=12,
                 color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, path)


def plot_families(table: pd.DataFrame, path: Path) -> Path:
    t = table.drop(columns="cv_std")
    fig, ax = _figure(7.0, 4.2)
    grouped_bars(ax, t, [BLUE, ORANGE], fmt="{:.3f}", scale=1, errors={t.columns[0]: table.cv_std})
    _style(ax, "Macro F1", ymax=1.0)
    return _finish(fig, ax, "Model families: CV (train) vs test macro F1, best configuration each", path)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    return path


def classifier_figures(out_dir: Path, metrics_path: Path = RESULTS_DOCS / "answerability_metrics.json",
                       features_path: Path = FEATURES_PATH, model_path: Path = MODEL_PATH) -> list[Path]:
    """The four classifier figures (deployed model = experiment 'main'); test split only."""
    main_r = json.loads(metrics_path.read_text(encoding="utf-8"))["main"]
    curves = curve_data(*test_scores(load_model(model_path), pd.read_csv(features_path)))
    return [plot_confusion(confusion_data(main_r["test"]), out_dir / "confusion_matrix.png",
                           f"Deployed model, test split (n={main_r['test']['n']}), threshold 0.5"),
            plot_curves(curves, out_dir / "roc_pr.png"),
            plot_importance(importance_data(main_r["weights"]), out_dir / "feature_importance.png"),
            plot_families(family_data(main_r["per_family"]), out_dir / "model_families.png")]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Make the Module 4 report figures.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    groups = {
        "answerability": lambda: [plot_answerability(answerability_data(json.loads(
            (RESULTS_DOCS / "heldout_metrics.json").read_text(encoding="utf-8"))), args.out_dir / "answerability.png")],
        "gate": lambda: [plot_gate(gate_data(pd.read_csv(eval_gate.newest("gate_test_*.csv", args.results_dir))),
                                   args.out_dir / "gate.png")],
        "our sets": lambda: [plot_our_sets(our_sets_data(pd.read_csv(eval_gate.newest(
            "questions_validated_csv+hinglish_test_*_rescored.csv", args.results_dir))), args.out_dir / "our_sets.png")],
        "BIRD": lambda: [plot_bird(bird_data(compare_bird.pair_runs(*(
            pd.read_csv(compare_bird.newest(args.results_dir, m, compare_bird.DEFAULT_TAG))
            for m in ("validated", "baseline")))), args.out_dir / "bird.png")],
        "classifier": lambda: classifier_figures(args.out_dir),
    }
    for name, make in groups.items():
        try:
            for p in make():
                print(f"Wrote {p}")
        except FileNotFoundError as e:  # e.g. data/results runs not present on this machine
            print(f"Skipped {name} figure(s): {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
