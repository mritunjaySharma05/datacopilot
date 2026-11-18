"""Evaluate the SQL pipeline on our own answerable question sets (Module 4, step 3).

Each question runs through runner.answer in validated mode (schema check + one retry)
and/or baseline mode (one attempt, no schema check; read-only guard kept), and the result
is compared with the stored gold_answer:

  strict   same rows and values as the gold answer (datacopilot.questions.answers_match;
           row order only matters if the gold SQL has an outer ORDER BY)
  relaxed  every gold column appears in the result, numbers rounded to the gold precision
           (extra columns and unrounded averages allowed; answers_match_relaxed)

Olist runs as SQLite on olist.sqlite; IPL, F1 and crop as DuckDB on the processed CSVs.

Usage:
    python scripts/eval_questions.py                         # test split, csv + hinglish, both modes
    python scripts/eval_questions.py --sets heldout          # human held-out answerable questions
    python scripts/eval_questions.py --modes validated --limit 5
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datacopilot.pipeline.profiler import get_schema, get_sqlite_schema  # noqa: E402
from datacopilot.pipeline.runner import SQLAttemptError, answer  # noqa: E402
from datacopilot.pipeline.sql_generator import generate_sql  # noqa: E402
from datacopilot.questions import (  # noqa: E402
    DIALECT, PROCESSED_DIR, QUESTIONS_DIR, answers_match, answers_match_relaxed, connect, is_ordered,
    load_heldout, load_question_set, normalize_value,
)

DEFAULT_OUT_DIR = ROOT / "data" / "results"
DEFAULT_REPORT = ROOT / "docs" / "results" / "questions.md"
SETS = ("csv", "hinglish", "heldout")
MODES = ("validated", "baseline")


def load_questions(sets: list[str], split: str, questions_dir: Path = QUESTIONS_DIR) -> list[dict]:
    """Answerable questions from the chosen sets. `split` (train / test / all) does not apply to heldout."""
    rows = []
    for name in sets:
        if name == "heldout":
            rows += [{**r, "question_set": "heldout"} for r in load_heldout(questions_dir) if r["answerable"]]
        else:
            chosen = None if split == "all" else split
            rows += [{**r, "question_set": name} for r in load_question_set(name, chosen, questions_dir)]
    return rows


class Datasets:
    """One open connection and schema per dataset, reused across questions."""

    def __init__(self, processed_dir: Path = PROCESSED_DIR):
        self.processed_dir, self._cache = processed_dir, {}

    def __call__(self, dataset: str) -> tuple[object, dict]:
        if dataset not in self._cache:
            conn = connect(dataset, self.processed_dir)
            schema = get_sqlite_schema(conn) if DIALECT[dataset] == "sqlite" else get_schema(conn)
            self._cache[dataset] = (conn, schema)
        return self._cache[dataset]


def result_rows(df: pd.DataFrame) -> list[list]:
    """DataFrame -> rows normalized like the stored gold answers."""
    return [[normalize_value(v) for v in row] for row in df.itertuples(index=False, name=None)]


def status_of(error: str) -> str:
    """Failure category from a SQLAttemptError message."""
    return "validation_failed" if error.startswith("Validation failed") else "execution_failed"


def evaluate_question(row: dict, conn, schema: dict, use_validator: bool = True,
                      generate: Callable[..., str] | None = None) -> dict:
    """Run one question through the pipeline and score it against its gold answer."""
    record = {"id": row["id"], "question_set": row["question_set"], "dataset": row["dataset"],
              "difficulty": row["difficulty"], "question": row["question"], "gold_sql": row["gold_sql"],
              "pred_sql": "", "status": "", "strict": False, "relaxed": False, "error": "", "seconds": 0.0}
    start = time.monotonic()
    try:
        sql, df = answer(conn, row["question"], schema, generate or generate_sql,
                         dialect=DIALECT[row["dataset"]], use_validator=use_validator)
    except SQLAttemptError as err:
        record.update(pred_sql=err.sql, error=str(err), status=status_of(str(err)))
    else:
        rows, ordered = result_rows(df), is_ordered(row["gold_sql"])
        record.update(pred_sql=sql, strict=answers_match(row["gold_answer"], rows, ordered),
                      relaxed=answers_match_relaxed(row["gold_answer"], rows, ordered))
        record["status"] = "correct" if record["strict"] else "wrong_result"
    record["seconds"] = round(time.monotonic() - start, 2)
    return record


def accuracy_table(records: pd.DataFrame, by: str) -> pd.DataFrame:
    """Strict / relaxed accuracy per mode and group (plus 'all')."""
    parts = []
    for (mode, group), g in records.groupby(["mode", by]):
        parts.append({"mode": mode, by: group, "n": len(g), "strict": g.strict.mean(), "relaxed": g.relaxed.mean()})
    for mode, g in records.groupby("mode"):
        parts.append({"mode": mode, by: "all", "n": len(g), "strict": g.strict.mean(), "relaxed": g.relaxed.mean()})
    return pd.DataFrame(parts)


def report(records: pd.DataFrame, title: str) -> str:
    """Markdown summary: accuracy by set, dataset, difficulty; failure categories."""
    lines = [f"# {title}", "", "Generated by `scripts/eval_questions.py`. Strict = same rows and values as the",
             "gold answer; relaxed = every gold column present, numbers rounded to the gold precision.", ""]
    for by in ("question_set", "dataset", "difficulty"):
        t = accuracy_table(records, by)
        modes = sorted(t["mode"].unique(), reverse=True)  # validated before baseline
        groups = [g for g in t[by].unique() if g != "all"] + ["all"]
        lines += [f"## Accuracy by {by.replace('_', ' ')}", "",
                  "| " + by + " | n | " + " | ".join(f"{m} strict | {m} relaxed" for m in modes) + " |",
                  "|---|---|" + "---|---|" * len(modes)]
        for grp in groups:
            cells, n = [], ""
            for m in modes:
                hit = t[(t["mode"] == m) & (t[by] == grp)]
                if hit.empty:
                    cells += ["", ""]
                    continue
                n = n or int(hit.n.iloc[0])  # first mode's count
                cells += [f"{100 * hit.strict.iloc[0]:.1f}%", f"{100 * hit.relaxed.iloc[0]:.1f}%"]
            name = f"**{grp}**" if grp == "all" else grp
            lines.append(f"| {name} | {n} | " + " | ".join(cells) + " |")
        lines.append("")
    counts = records.groupby(["mode", "status"]).size().unstack(fill_value=0)
    lines += ["## Outcomes", "", "| mode | " + " | ".join(counts.columns) + " |",
              "|---|" + "---|" * len(counts.columns)]
    lines += [f"| {m} | " + " | ".join(str(int(v)) for v in counts.loc[m]) + " |" for m in counts.index]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Execution accuracy on our own question sets.")
    parser.add_argument("--sets", nargs="+", choices=SETS, default=["csv", "hinglish"])
    parser.add_argument("--split", choices=("train", "test", "all"), default="test")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--limit", type=int, default=None, help="Only the first N questions (smoke test)")
    parser.add_argument("--questions-dir", type=Path, default=QUESTIONS_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report", type=Path, default=None, help=f"Markdown summary (e.g. {DEFAULT_REPORT})")
    args = parser.parse_args(argv)

    questions = load_questions(args.sets, args.split, args.questions_dir)[: args.limit]
    datasets = Datasets(args.processed_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = f"{'+'.join(args.sets)}_{args.split}"
    all_records = []
    for mode in args.modes:
        print(f"Mode: {mode} | sets: {', '.join(args.sets)} | split: {args.split} | questions: {len(questions)}")
        records = []
        for i, row in enumerate(questions, 1):
            conn, schema = datasets(row["dataset"])
            try:
                rec = evaluate_question(row, conn, schema, use_validator=mode == "validated")
            except requests.ConnectionError:
                print("Could not reach Ollama. Is it running? (check OLLAMA_HOST in .env)", file=sys.stderr)
                return 1
            records.append({**rec, "mode": mode})
            mark = "OK  " if rec["strict"] else ("ok~ " if rec["relaxed"] else "FAIL")
            note = f" ({rec['error'][:70]})" if rec["error"] else ""
            print(f"[{i}/{len(questions)}] {mark} {rec['id']} {rec['seconds']:.1f}s{note}")
        df = pd.DataFrame(records)
        out = args.out_dir / f"questions_{mode}_{label}_{stamp}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"{mode}: strict {df.strict.mean():.1%}, relaxed {df.relaxed.mean():.1%} | saved {out}\n")
        all_records += records
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        title = f"Pipeline accuracy: {', '.join(args.sets)} ({args.split if args.sets != ['heldout'] else 'heldout'})"
        args.report.write_text(report(pd.DataFrame(all_records), title), encoding="utf-8")
        print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
