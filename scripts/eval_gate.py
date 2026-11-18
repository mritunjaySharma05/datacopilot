"""Answerability gate experiment (Module 4, step 4).

Gate = the deployed answerability classifier (models/answerability.joblib, threshold 0.5,
nothing tuned here). With the gate, a question predicted unanswerable is refused before
any SQL is generated; without it, every question goes through the validated pipeline.

End-to-end correctness per question:
  answerable    correct if it is not refused and the pipeline's answer matches the gold
                answer (strict; relaxed reported as secondary)
  unanswerable  correct if no answer is returned: refused by the gate, or the pipeline
                produced no result (validation / execution failure). Pipeline failures are
                reported separately so they are not mistaken for deliberate refusals.

Answerable questions reuse the stored validated runs of scripts/eval_questions.py (the gate
only removes questions, it does not change the SQL of the rest). Unanswerable questions
are run through the validated pipeline here.

Usage:
    python scripts/eval_gate.py                 # test split + human held-out, newest stored runs
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd
import requests
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_features import dataset_inputs  # noqa: E402
from datacopilot.pipeline.answerability import load_model, predict_answerable  # noqa: E402
from datacopilot.pipeline.runner import SQLAttemptError, answer  # noqa: E402
from datacopilot.pipeline.sql_generator import generate_sql  # noqa: E402
from datacopilot.questions import (  # noqa: E402
    DIALECT, PROCESSED_DIR, QUESTIONS_DIR, SET_FILES, load_heldout, load_question_set,
)
from eval_questions import Datasets  # noqa: E402

RESULTS_DIR = ROOT / "data" / "results"
REPORT = ROOT / "docs" / "results" / "gate.md"


def load_split(split: str, questions_dir: Path = QUESTIONS_DIR) -> list[dict]:
    """Test rows of all four sets, or the human held-out set (split == "heldout")."""
    if split == "heldout":
        return [{**r, "question_set": f"heldout_{r['language']}"} for r in load_heldout(questions_dir)]
    return [{**r, "question_set": name} for name in SET_FILES for r in load_question_set(name, split, questions_dir)]


def gate_predictions(rows: list[dict], inputs: dict, bundle: dict) -> pd.DataFrame:
    """id, gate_pass (predicted answerable), p_answerable for every question."""
    out = []
    for r in rows:
        schema, context = inputs[r["dataset"]]
        ok, prob = predict_answerable(r["question"], schema, context, bundle)
        out.append({"id": r["id"], "gate_pass": ok, "p_answerable": round(prob, 4)})
    return pd.DataFrame(out)


def run_unanswerable(row: dict, conn, schema: dict, generate: Callable[..., str] | None = None) -> dict:
    """Run an unanswerable question through the validated pipeline: did it return an answer?"""
    rec = {"id": row["id"], "pred_sql": "", "answered": False, "n_rows": 0, "status": "", "error": ""}
    start = time.monotonic()
    try:
        sql, df = answer(conn, row["question"], schema, generate or generate_sql,
                         dialect=DIALECT[row["dataset"]], use_validator=True)
    except SQLAttemptError as err:
        failed = "validation_failed" if str(err).startswith("Validation failed") else "execution_failed"
        rec.update(pred_sql=err.sql, status=failed, error=str(err))
    else:
        rec.update(pred_sql=sql, answered=True, n_rows=len(df), status="answered")
    rec["seconds"] = round(time.monotonic() - start, 2)
    return rec


def combine(rows: list[dict], gate: pd.DataFrame, answerable_runs: pd.DataFrame,
            unanswerable_runs: pd.DataFrame) -> pd.DataFrame:
    """One row per question with gate decision, pipeline outcome and correctness with / without gate."""
    meta = pd.DataFrame([{"id": r["id"], "dataset": r["dataset"], "question_set": r["question_set"],
                          "question": r["question"], "answerable": bool(r["answerable"]),
                          "unanswerable_type": r["unanswerable_type"]} for r in rows])
    df = meta.merge(gate, on="id", validate="one_to_one")
    ans = answerable_runs[["id", "strict", "relaxed", "status"]].rename(columns={"status": "pipeline_status"})
    una = unanswerable_runs[["id", "answered", "status"]].rename(columns={"status": "pipeline_status"})
    df = df.merge(pd.concat([ans, una], ignore_index=True), on="id", how="left", validate="one_to_one")
    missing = df[df.pipeline_status.isna()].id.tolist()
    if missing:
        raise ValueError(f"no pipeline run for: {missing[:5]}")
    a = df.answerable
    df["answered"] = df.answered.astype("boolean").fillna(df.pipeline_status.isin(["correct", "wrong_result"]))
    for metric in ("strict", "relaxed"):
        hit = df[metric].astype("boolean").fillna(False).astype(bool)
        df[f"correct_no_gate_{metric}"] = (a & hit) | (~a & ~df.answered)
        df[f"correct_gate_{metric}"] = (a & df.gate_pass & hit) | (~a & (~df.gate_pass | ~df.answered))
    return df


def mcnemar(x: pd.Series, y: pd.Series) -> dict:
    """Exact McNemar test on paired booleans (x only right vs y only right)."""
    x, y = x.astype(bool).to_numpy(), y.astype(bool).to_numpy()
    only_x, only_y = int((x & ~y).sum()), int((~x & y).sum())
    n = only_x + only_y
    return {"only_gate": only_x, "only_no_gate": only_y,
            "p_value": float(binomtest(min(only_x, only_y), n, 0.5).pvalue) if n else 1.0}


def summary(df: pd.DataFrame) -> dict:
    """Headline numbers for one set of questions."""
    a, u = df[df.answerable], df[~df.answerable]
    return {
        "n": len(df), "n_answerable": len(a), "n_unanswerable": len(u),
        "correct_refusals": int((~u.gate_pass).sum()),                      # gate refuses unanswerable
        "false_refusals": int((~a.gate_pass).sum()),                        # gate refuses answerable
        "false_refusals_correct": int((~a.gate_pass & a.strict.astype(bool)).sum()),  # ... pipeline had it right
        "unans_answered_no_gate": int(u.answered.sum()),                    # wrong answer delivered
        "unans_answered_gate": int((u.gate_pass & u.answered).sum()),
        "unans_pipeline_failed": int((~u.answered).sum()),
        **{f"acc_{c}_{m}": float(df[f"correct_{c}_{m}"].mean())
           for c in ("no_gate", "gate") for m in ("strict", "relaxed")},
        **{f"acc_{c}_{m}_{part}": float(sub[f"correct_{c}_{m}"].mean()) if len(sub) else float("nan")
           for c in ("no_gate", "gate") for m in ("strict", "relaxed")
           for part, sub in (("answerable", a), ("unanswerable", u))},
        "mcnemar_strict": mcnemar(df.correct_gate_strict, df.correct_no_gate_strict),
    }


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def section(title: str, df: pd.DataFrame) -> list[str]:
    """Markdown for one evaluation set."""
    s = summary(df)
    m = s["mcnemar_strict"]
    lines = [f"## {title} ({s['n']} questions: {s['n_answerable']} answerable, {s['n_unanswerable']} unanswerable)", "",
             "| | Without gate | With gate |", "|---|---|---|",
             f"| **End-to-end accuracy (strict)** | {_pct(s['acc_no_gate_strict'])} | {_pct(s['acc_gate_strict'])} |",
             f"| End-to-end accuracy (relaxed) | {_pct(s['acc_no_gate_relaxed'])} | {_pct(s['acc_gate_relaxed'])} |",
             f"| Answerable answered correctly (strict) | {_pct(s['acc_no_gate_strict_answerable'])} | "
             f"{_pct(s['acc_gate_strict_answerable'])} |",
             f"| Unanswerable with no answer returned | {_pct(s['acc_no_gate_strict_unanswerable'])} | "
             f"{_pct(s['acc_gate_strict_unanswerable'])} |",
             f"| Unanswerable that got an (invented) answer | {s['unans_answered_no_gate']} | {s['unans_answered_gate']} |",
             "",
             f"- Gate: correct refusals {s['correct_refusals']} / {s['n_unanswerable']} unanswerable; "
             f"false refusals {s['false_refusals']} / {s['n_answerable']} answerable "
             f"({s['false_refusals_correct']} of them the pipeline would have answered correctly).",
             f"- Without the gate, {s['unans_pipeline_failed']} unanswerable questions produced no answer only because "
             "the pipeline failed (validation or execution error), not because anything recognised them.",
             f"- McNemar (strict, exact): gate only right {m['only_gate']}, no gate only right {m['only_no_gate']}, "
             f"p = {m['p_value']:.4f}.", ""]
    u = df[~df.answerable]
    lines += ["| Unanswerable type | n | Refused by gate | Answer returned without gate | Answer returned with gate |",
              "|---|---|---|---|---|"]
    for t, g in u.groupby("unanswerable_type"):
        lines.append(f"| {t} | {len(g)} | {int((~g.gate_pass).sum())} | {int(g.answered.sum())} | "
                     f"{int((g.gate_pass & g.answered).sum())} |")
    lines += ["", "| Question set | n | Accuracy without gate (strict) | Accuracy with gate (strict) |", "|---|---|---|---|"]
    for qs_name, g in df.groupby("question_set"):
        lines.append(f"| {qs_name} | {len(g)} | {_pct(g.correct_no_gate_strict.mean())} | {_pct(g.correct_gate_strict.mean())} |")
    refused = df[df.answerable & ~df.gate_pass]
    lines += ["", f"False refusals ({len(refused)}):", ""]
    lines += [f"- {r.id} ({'pipeline correct' if r.strict else 'pipeline wrong anyway'}, "
              f"P = {r.p_answerable:.2f}): {r.question}" for r in refused.itertuples()] or ["- none"]
    leaked = u[u.gate_pass & u.answered]
    lines += ["", f"Unanswerable questions that pass the gate and get an answer ({len(leaked)}):", ""]
    lines += [f"- {r.id} ({r.unanswerable_type}, P = {r.p_answerable:.2f}): {r.question}"
              for r in leaked.itertuples()] or ["- none"]
    return lines + [""]


def report(sections: dict[str, pd.DataFrame]) -> str:
    """Full markdown report."""
    lines = ["# Answerability gate experiment", "",
             "Generated by `scripts/eval_gate.py`. Gate = deployed classifier (XGBoost, no `n_words`),",
             "threshold 0.5, nothing tuned. Pipeline = validated mode, qwen2.5-coder:7b, temperature 0.",
             "Answerable questions: correct if not refused and the answer matches the gold answer.",
             "Unanswerable questions: correct if no answer is returned (gate refusal or pipeline failure).",
             "The held-out set was used once before (to drop `n_words`), so its numbers are optimistic.", ""]
    for title, df in sections.items():
        lines += section(title, df)
    return "\n".join(lines)


def newest(pattern: str, results_dir: Path) -> Path:
    paths = sorted(results_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No {pattern} in {results_dir}; run scripts/eval_questions.py first")
    return paths[-1]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Answerability gate experiment.")
    parser.add_argument("--questions-dir", type=Path, default=QUESTIONS_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=REPORT)
    args = parser.parse_args(argv)

    bundle = load_model()
    datasets = Datasets(args.processed_dir)
    inputs = {ds: dataset_inputs(ds, args.processed_dir) for ds in ("olist", "ipl", "f1", "crop")}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stored = {"test": "questions_validated_csv+hinglish_test_*.csv", "heldout": "questions_validated_heldout_test_*.csv"}
    titles = {"test": "Test split", "heldout": "Human held-out"}
    sections = {}
    for split in ("test", "heldout"):
        rows = load_split(split, args.questions_dir)
        answerable_path = newest(stored[split], args.results_dir)
        print(f"{titles[split]}: {len(rows)} questions; answerable runs from {answerable_path.name}")
        runs = []
        for i, r in enumerate([r for r in rows if not r["answerable"]], 1):
            conn, schema = datasets(r["dataset"])
            try:
                rec = run_unanswerable(r, conn, schema)
            except requests.ConnectionError:
                print("Could not reach Ollama. Is it running? (check OLLAMA_HOST in .env)", file=sys.stderr)
                return 1
            runs.append(rec)
            print(f"  [{i}] {rec['id']} {rec['status']} {rec['seconds']:.1f}s")
        unans = pd.DataFrame(runs)
        unans.to_csv(args.results_dir / f"gate_unanswerable_{split}_{stamp}.csv", index=False)
        df = combine(rows, gate_predictions(rows, inputs, bundle), pd.read_csv(answerable_path), unans)
        df.to_csv(args.results_dir / f"gate_{split}_{stamp}.csv", index=False)
        s = summary(df)
        print(f"  accuracy strict: no gate {s['acc_no_gate_strict']:.1%}, gate {s['acc_gate_strict']:.1%}")
        sections[titles[split]] = df
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report(sections), encoding="utf-8")
    print(f"Report: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
