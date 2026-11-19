"""Make the Module 4 report figures (PNG, 150 dpi) in docs/results/figures/.

Numbers come from the stored results, never typed in by hand:
- answerability: docs/results/heldout_metrics.json (old logreg with n_words vs deployed model)
- gate: newest data/results/gate_test_*.csv (scored with eval_gate.summary)
- our sets: newest data/results/questions_validated_csv+hinglish_test_*_rescored.csv
- BIRD: newest bird_{validated,baseline}_sample300_seed42_*.csv (paired with compare_bird)

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
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import compare_bird  # noqa: E402
import eval_gate  # noqa: E402

RESULTS_DOCS = ROOT / "docs" / "results"
RESULTS_DIR = ROOT / "data" / "results"
OUT_DIR = RESULTS_DOCS / "figures"
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


def grouped_bars(ax, table: pd.DataFrame, colors: list[str], fmt: str = "{:.1f}%", scale: float = 100) -> None:
    """Grouped bars: one group per row, one bar per column, value labels on top."""
    n_series = len(table.columns)
    width = 0.8 / n_series
    for i, (col, color) in enumerate(zip(table.columns, colors)):
        xs = [g + (i - (n_series - 1) / 2) * width for g in range(len(table))]
        vals = [scale * v for v in table[col]]
        bars = ax.bar(xs, vals, width=width, color=color, label=col, edgecolor=SURFACE, linewidth=2)
        for b, v in zip(bars, vals):
            ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 3),
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


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Make the Module 4 report figures.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    metrics = json.loads((RESULTS_DOCS / "heldout_metrics.json").read_text(encoding="utf-8"))
    gate = pd.read_csv(eval_gate.newest("gate_test_*.csv", args.results_dir))
    ours = pd.read_csv(eval_gate.newest("questions_validated_csv+hinglish_test_*_rescored.csv", args.results_dir))
    paired = compare_bird.pair_runs(*(pd.read_csv(compare_bird.newest(args.results_dir, m, compare_bird.DEFAULT_TAG))
                                      for m in ("validated", "baseline")))
    for p in (plot_answerability(answerability_data(metrics), args.out_dir / "answerability.png"),
              plot_gate(gate_data(gate), args.out_dir / "gate.png"),
              plot_our_sets(our_sets_data(ours), args.out_dir / "our_sets.png"),
              plot_bird(bird_data(paired), args.out_dir / "bird.png")):
        print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
