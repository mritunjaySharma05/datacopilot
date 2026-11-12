"""Build answerability features for every question in csv_questions + unanswerable_questions.

Output: data/features/answerability_features.csv, one row per question:
    id, dataset, split, answerable, unanswerable_type   <- metadata / label (do not train on)
    <FEATURE_NAMES ...>                                  <- model inputs

The classifier is trained in Module 4 on split == "train" rows only; test rows are kept
here so they can be scored later, never used for fitting or feature selection.

Usage:
    python scripts/build_features.py
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datacopilot.pipeline.features import FEATURE_NAMES, build_context, extract_features  # noqa: E402
from datacopilot.pipeline.profiler import get_schema, get_sqlite_schema  # noqa: E402
from datacopilot.questions import PROCESSED_DIR, QUESTIONS_DIR, connect, load_question_set  # noqa: E402

OUT_PATH = ROOT / "data" / "features" / "answerability_features.csv"
META_COLUMNS = ("id", "dataset", "split", "answerable", "unanswerable_type")
SOURCE_SETS = ("csv", "unanswerable")


def dataset_inputs(dataset: str, processed_dir: Path = PROCESSED_DIR) -> tuple[dict, object]:
    """(schema, context) for one dataset, read from the processed data."""
    conn = connect(dataset, processed_dir)
    schema = get_sqlite_schema(conn) if dataset == "olist" else get_schema(conn)
    return schema, build_context(conn, schema)


def build_feature_table(questions: list[dict], inputs: dict[str, tuple[dict, object]]) -> pd.DataFrame:
    """One row per question: metadata columns, then FEATURE_NAMES in order."""
    rows = []
    for q in questions:
        schema, context = inputs[q["dataset"]]
        rows.append({"id": q["id"], "dataset": q["dataset"], "split": q["split"],
                     "answerable": int(q["answerable"]), "unanswerable_type": q["unanswerable_type"],
                     **extract_features(q["question"], schema, context)})
    return pd.DataFrame(rows, columns=[*META_COLUMNS, *FEATURE_NAMES])


def compare_classes(table: pd.DataFrame, split: str = "train") -> pd.DataFrame:
    """Mean of each feature for answerable vs unanswerable rows of one split."""
    rows = table[table.split == split]
    means = rows.groupby("answerable")[list(FEATURE_NAMES)].mean().T
    means.columns = ["unanswerable" if c == 0 else "answerable" for c in means.columns]
    means["difference"] = means["unanswerable"] - means["answerable"]
    return means[["answerable", "unanswerable", "difference"]]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build answerability features.")
    parser.add_argument("--questions-dir", type=Path, default=QUESTIONS_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args(argv)

    questions = [q for name in SOURCE_SETS for q in load_question_set(name, questions_dir=args.questions_dir)]
    inputs = {ds: dataset_inputs(ds, args.processed_dir) for ds in sorted({q["dataset"] for q in questions})}
    table = build_feature_table(questions, inputs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False, float_format="%.4f")
    print(f"Wrote {len(table)} rows x {len(FEATURE_NAMES)} features to {args.out}")
    print(table.groupby(["split", "answerable"]).size().rename("rows").to_string())
    print("\nMean feature value, train split:")
    print(compare_classes(table).round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
