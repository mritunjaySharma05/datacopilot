"""Compare validated vs baseline runs of scripts/eval_questions.py on the same questions.

Pairs the two per-question CSVs by id and reports strict / relaxed accuracy by dataset and
language, every question where the two modes disagree, and an exact McNemar test on the
paired results (two-sided binomial test on the discordant pairs; appropriate for small
counts).

Usage (defaults: the newest matching CSVs in data/results/):
    python scripts/compare_modes.py
    python scripts/compare_modes.py --test V.csv B.csv --heldout V.csv B.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datacopilot.questions import QUESTIONS_DIR, load_heldout  # noqa: E402

RESULTS_DIR = ROOT / "data" / "results"
REPORT = ROOT / "docs" / "results" / "modes_comparison.md"
KEYS = ["id", "question_set", "dataset", "difficulty", "question"]
METRICS = ("strict", "relaxed")


def newest(results_dir: Path, mode: str, label: str) -> Path:
    """Newest data/results/questions_<mode>_<label>_<stamp>.csv."""
    paths = sorted(results_dir.glob(f"questions_{mode}_{label}_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No questions_{mode}_{label}_*.csv in {results_dir}")
    return paths[-1]


def pair_runs(validated: pd.DataFrame, baseline: pd.DataFrame, languages: dict[str, str] | None = None) -> pd.DataFrame:
    """One row per question with _v (validated) and _b (baseline) columns, plus language."""
    cols = KEYS + ["strict", "relaxed", "status", "pred_sql", "error"]
    paired = validated[cols].merge(baseline[cols], on=KEYS, suffixes=("_v", "_b"), validate="one_to_one")
    if len(paired) != len(validated) or len(paired) != len(baseline):
        raise ValueError("validated and baseline runs cover different questions")
    languages = languages or {}
    paired["language"] = [languages.get(i, "hinglish" if s == "hinglish" else "en")
                          for i, s in zip(paired.id, paired.question_set)]
    return paired


def mcnemar_exact(validated: pd.Series, baseline: pd.Series) -> dict:
    """Exact McNemar test: b = validated right & baseline wrong, c = the reverse."""
    v, b = validated.astype(bool).to_numpy(), baseline.astype(bool).to_numpy()
    only_v, only_b = int((v & ~b).sum()), int((~v & b).sum())
    n = only_v + only_b
    p = binomtest(min(only_v, only_b), n, 0.5).pvalue if n else 1.0
    return {"only_validated": only_v, "only_baseline": only_b, "discordant": n, "p_value": float(p)}


def accuracy_by(paired: pd.DataFrame, by: str) -> pd.DataFrame:
    """Accuracy of each mode and metric per group, plus an 'all' row."""
    cols = [f"{m}_{s}" for m in METRICS for s in ("v", "b")]
    table = paired.groupby(by)[cols].mean()
    table.loc["all"] = paired[cols].mean()
    table.insert(0, "n", paired.groupby(by).size().reindex(table.index).fillna(len(paired)).astype(int))
    return table


def disagreements(paired: pd.DataFrame) -> pd.DataFrame:
    """Questions where the modes differ on the strict or the relaxed metric."""
    mask = (paired.strict_v != paired.strict_b) | (paired.relaxed_v != paired.relaxed_b)
    return paired[mask].sort_values("id").reset_index(drop=True)


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _mark(strict: bool, relaxed: bool) -> str:
    return "correct" if strict else ("relaxed only" if relaxed else "wrong")


def section(title: str, paired: pd.DataFrame) -> list[str]:
    """Markdown for one paired comparison."""
    lines = [f"## {title} ({len(paired)} questions)", ""]
    for by in ("dataset", "language"):
        t = accuracy_by(paired, by)
        lines += [f"### Accuracy by {by}", "",
                  f"| {by} | n | validated strict | baseline strict | validated relaxed | baseline relaxed |",
                  "|---|---|---|---|---|---|"]
        for grp, r in t.iterrows():
            name = f"**{grp}**" if grp == "all" else grp
            lines.append(f"| {name} | {r.n} | {_pct(r.strict_v)} | {_pct(r.strict_b)} | "
                         f"{_pct(r.relaxed_v)} | {_pct(r.relaxed_b)} |")
        lines.append("")
    lines += ["### McNemar's test (exact, two-sided)", "",
              "| Metric | validated only right | baseline only right | discordant | p-value |",
              "|---|---|---|---|---|"]
    for m in METRICS:
        t = mcnemar_exact(paired[f"{m}_v"], paired[f"{m}_b"])
        lines.append(f"| {m} | {t['only_validated']} | {t['only_baseline']} | {t['discordant']} | {t['p_value']:.3f} |")
    d = disagreements(paired)
    lines += ["", f"### Questions where the modes disagree ({len(d)})", ""]
    if d.empty:
        return lines + ["None.", ""]
    lines += ["| Id | Question | Validated | Baseline | Baseline status |", "|---|---|---|---|---|"]
    for r in d.itertuples():
        q = r.question.replace("|", "\\|")
        lines.append(f"| {r.id} | {q} | {_mark(r.strict_v, r.relaxed_v)} | {_mark(r.strict_b, r.relaxed_b)} | "
                     f"{r.status_b} |")
    return lines + [""]


def report(sections: dict[str, pd.DataFrame]) -> str:
    """Full markdown report."""
    lines = ["# Validated vs baseline pipeline", "",
             "Generated by `scripts/compare_modes.py` from `scripts/eval_questions.py` runs (qwen2.5-coder:7b,",
             "temperature 0). Validated = schema check + one retry with the error; baseline = one attempt,",
             "no schema check (read-only guard kept). Strict = same rows and values as the gold answer;",
             "relaxed = every gold column present, numbers rounded to the gold precision.",
             "McNemar's test uses only the questions where exactly one mode is right.", ""]
    for title, paired in sections.items():
        lines += section(title, paired)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Compare validated vs baseline eval_questions runs.")
    parser.add_argument("--test", nargs=2, type=Path, metavar=("VALIDATED", "BASELINE"))
    parser.add_argument("--heldout", nargs=2, type=Path, metavar=("VALIDATED", "BASELINE"))
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--questions-dir", type=Path, default=QUESTIONS_DIR)
    parser.add_argument("--out", type=Path, default=REPORT)
    args = parser.parse_args(argv)

    runs = {
        "Test split (csv + hinglish)": args.test or [newest(args.results_dir, m, "csv+hinglish_test")
                                                     for m in ("validated", "baseline")],
        "Human held-out, answerable": args.heldout or [newest(args.results_dir, m, "heldout_test")
                                                       for m in ("validated", "baseline")],
    }
    languages = {r["id"]: r["language"] for r in load_heldout(args.questions_dir)}
    sections = {}
    for title, (v_path, b_path) in runs.items():
        print(f"{title}: {v_path.name} vs {b_path.name}")
        sections[title] = pair_runs(pd.read_csv(v_path), pd.read_csv(b_path), languages)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report(sections), encoding="utf-8")
    print(f"Report: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
