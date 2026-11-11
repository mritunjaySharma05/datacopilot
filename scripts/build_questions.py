"""Build and validate the evaluation question sets in data/questions/.

The JSONL files are the source of truth for the authored fields (id, dataset, question,
gold_sql, answerable, unanswerable_type, difficulty; Hinglish rows also source_id).

    python scripts/build_questions.py          # fill gold_answer + split, drop failing rows, validate
    python scripts/build_questions.py --check  # validate only (CI): exits 1 on any problem

Build: runs every gold_sql on data/processed and stores the result as gold_answer; drops
answerable rows whose SQL fails or returns nothing; assigns the seeded stratified split;
copies gold_sql / gold_answer / difficulty / split from the English original to each
Hinglish row (so a test question never leaks into train through its translation).

Check: field schema and allowed values, unique ids and questions, balanced unanswerable
types, expected counts, splits equal to the seeded assignment (~70/30), Hinglish linked to
its English source, and every gold_sql re-run gives the stored gold_answer.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datacopilot.questions import (  # noqa: E402
    DATASETS, DIFFICULTIES, FIELDS, QUESTIONS_DIR, PROCESSED_DIR, SET_FILES, TEST_SHARE, UNANSWERABLE_TYPES,
    answers_match, assign_splits, connect, is_ordered, read_jsonl, run_gold, write_jsonl, MAX_ANSWER_ROWS,
)

# Expected rows per dataset in each file.
EXPECTED_PER_DATASET = {"csv": 50, "hinglish": 25, "unanswerable": 50}
SPLIT_TOLERANCE = 0.05  # allowed deviation of each dataset's test share from TEST_SHARE


def norm_question(text: str) -> str:
    return " ".join(text.lower().split())


class Connections:
    """Lazily opened read-only connection per dataset."""

    def __init__(self, processed_dir: Path):
        self.processed_dir, self._conns = processed_dir, {}

    def __call__(self, dataset: str):
        if dataset not in self._conns:
            self._conns[dataset] = connect(dataset, self.processed_dir)
        return self._conns[dataset]


# ---------------------------------------------------------------- build

def build_answerable(rows: list[dict], conns: Connections) -> tuple[list[dict], list[str]]:
    """Fill gold_answer by running gold_sql; drop rows that fail, are empty or too large."""
    kept, dropped = [], []
    for row in rows:
        try:
            answer = run_gold(conns(row["dataset"]), row["gold_sql"], row["dataset"])
        except Exception as err:  # noqa: BLE001 - any failure means the gold SQL is unusable
            dropped.append(f"{row['id']}: gold_sql failed ({type(err).__name__}: {str(err)[:120]})")
            continue
        if not answer or all(v is None for r in answer for v in r):
            dropped.append(f"{row['id']}: gold_sql returned no data")
            continue
        if len(answer) > MAX_ANSWER_ROWS:
            dropped.append(f"{row['id']}: {len(answer)} rows (max {MAX_ANSWER_ROWS})")
            continue
        kept.append({**row, "answerable": True, "unanswerable_type": None, "gold_answer": answer})
    return kept, dropped


def build_unanswerable(rows: list[dict]) -> list[dict]:
    """Unanswerable rows have no gold SQL and no answer."""
    return [{**row, "answerable": False, "gold_sql": None, "gold_answer": None} for row in rows]


def with_splits(rows: list[dict]) -> list[dict]:
    splits = assign_splits(rows)
    return [{**row, "split": splits[row["id"]]} for row in rows]


def build_hinglish(rows: list[dict], english: dict[str, dict]) -> tuple[list[dict], list[str]]:
    """Copy gold fields and split from each Hinglish row's English source."""
    kept, dropped = [], []
    for row in rows:
        src = english.get(row.get("source_id"))
        if src is None:
            dropped.append(f"{row['id']}: source {row.get('source_id')} not in csv_questions")
            continue
        kept.append({**row, "dataset": src["dataset"], "gold_sql": src["gold_sql"], "gold_answer": src["gold_answer"],
                     "answerable": True, "unanswerable_type": None, "difficulty": src["difficulty"],
                     "split": src["split"]})
    return kept, dropped


def build(questions_dir: Path, processed_dir: Path) -> list[str]:
    """Rebuild all three files in place; return messages about dropped rows."""
    conns = Connections(processed_dir)
    messages = []
    csv_rows, dropped = build_answerable(read_jsonl(questions_dir / SET_FILES["csv"]), conns)
    messages += dropped
    csv_rows = with_splits(csv_rows)
    write_jsonl(questions_dir / SET_FILES["csv"], csv_rows)

    un_rows = with_splits(build_unanswerable(read_jsonl(questions_dir / SET_FILES["unanswerable"])))
    write_jsonl(questions_dir / SET_FILES["unanswerable"], un_rows)

    hi_path = questions_dir / SET_FILES["hinglish"]
    if hi_path.exists():
        hi_rows, dropped = build_hinglish(read_jsonl(hi_path), {r["id"]: r for r in csv_rows})
        messages += dropped
        write_jsonl(hi_path, hi_rows)
    return messages


# ---------------------------------------------------------------- check

def check_schema(name: str, rows: list[dict]) -> list[str]:
    """Fields, types and allowed values for one file."""
    problems = []
    required = set(FIELDS) | ({"source_id"} if name == "hinglish" else set())
    for row in rows:
        rid = row.get("id", "?")
        missing = required - set(row)
        if missing:
            problems.append(f"{name}/{rid}: missing fields {sorted(missing)}")
            continue
        if row["dataset"] not in DATASETS:
            problems.append(f"{name}/{rid}: unknown dataset {row['dataset']!r}")
        if row["difficulty"] not in DIFFICULTIES:
            problems.append(f"{name}/{rid}: bad difficulty {row['difficulty']!r}")
        if row["split"] not in ("train", "test"):
            problems.append(f"{name}/{rid}: bad split {row['split']!r}")
        if not isinstance(row["question"], str) or not row["question"].strip():
            problems.append(f"{name}/{rid}: empty question")
        if row["answerable"] is True:
            if row["unanswerable_type"] is not None:
                problems.append(f"{name}/{rid}: answerable but unanswerable_type set")
            if not row["gold_sql"] or not row["gold_answer"]:
                problems.append(f"{name}/{rid}: answerable but no gold_sql / gold_answer")
        elif row["answerable"] is False:
            if row["unanswerable_type"] not in UNANSWERABLE_TYPES:
                problems.append(f"{name}/{rid}: bad unanswerable_type {row['unanswerable_type']!r}")
            if row["gold_sql"] is not None or row["gold_answer"] is not None:
                problems.append(f"{name}/{rid}: unanswerable rows must have null gold_sql / gold_answer")
        else:
            problems.append(f"{name}/{rid}: answerable must be true/false")
    expected_answerable = name != "unanswerable"
    wrong = [r["id"] for r in rows if r.get("answerable") is not expected_answerable]
    if wrong:
        problems.append(f"{name}: rows with answerable != {expected_answerable}: {wrong[:5]}")
    return problems


def check_counts(name: str, rows: list[dict]) -> list[str]:
    """Expected rows per dataset; unanswerable types balanced within each dataset."""
    problems = []
    per_ds = Counter(r["dataset"] for r in rows)
    for ds in DATASETS:
        if per_ds[ds] != EXPECTED_PER_DATASET[name]:
            problems.append(f"{name}: {ds} has {per_ds[ds]} rows, expected {EXPECTED_PER_DATASET[name]}")
    if name == "unanswerable":
        for ds in DATASETS:
            types = Counter(r["unanswerable_type"] for r in rows if r["dataset"] == ds)
            if len(set(types.values())) > 1 or set(types) != set(UNANSWERABLE_TYPES):
                problems.append(f"unanswerable: {ds} types not balanced: {dict(types)}")
    return problems


def check_splits(name: str, rows: list[dict], english: dict[str, dict]) -> list[str]:
    """Splits follow the seeded assignment (Hinglish: the English source's split), ~70/30."""
    problems = []
    if name == "hinglish":
        for r in rows:
            src = english.get(r["source_id"])
            if src is None:
                problems.append(f"hinglish/{r['id']}: source_id {r['source_id']} not found")
            elif (r["split"], r["gold_sql"], r["gold_answer"]) != (src["split"], src["gold_sql"], src["gold_answer"]):
                problems.append(f"hinglish/{r['id']}: split/gold_sql/gold_answer differ from {src['id']}")
            elif norm_question(r["question"]) == norm_question(src["question"]):
                problems.append(f"hinglish/{r['id']}: question identical to the English original")
    else:
        expected = assign_splits(rows)
        changed = [r["id"] for r in rows if r["split"] != expected[r["id"]]]
        if changed:
            problems.append(f"{name}: splits differ from the seeded assignment: {changed[:5]}")
    for ds in DATASETS:
        ds_rows = [r for r in rows if r["dataset"] == ds]
        if ds_rows:
            share = sum(r["split"] == "test" for r in ds_rows) / len(ds_rows)
            if abs(share - TEST_SHARE) > SPLIT_TOLERANCE:
                problems.append(f"{name}: {ds} test share {share:.0%} (target {TEST_SHARE:.0%})")
    return problems


def check_duplicates(all_rows: dict[str, list[dict]]) -> list[str]:
    """Ids unique across all files; question text unique within each file."""
    problems = []
    ids = Counter(r["id"] for rows in all_rows.values() for r in rows)
    problems += [f"duplicate id {i}" for i, n in ids.items() if n > 1]
    for name, rows in all_rows.items():
        texts = Counter((r["dataset"], norm_question(r["question"])) for r in rows)
        problems += [f"{name}: duplicate question in {ds}: {q!r}" for (ds, q), n in texts.items() if n > 1]
    return problems


def check_answers(rows: list[dict], conns: Connections) -> list[str]:
    """Re-run every gold_sql and compare with the stored gold_answer."""
    problems = []
    for row in rows:
        try:
            fresh = run_gold(conns(row["dataset"]), row["gold_sql"], row["dataset"])
        except Exception as err:  # noqa: BLE001
            problems.append(f"{row['id']}: gold_sql failed: {type(err).__name__}: {str(err)[:120]}")
            continue
        if not answers_match(row["gold_answer"], fresh, ordered=is_ordered(row["gold_sql"])):
            problems.append(f"{row['id']}: stored gold_answer differs from a fresh run")
    return problems


def check(questions_dir: Path, processed_dir: Path, run_sql: bool = True) -> list[str]:
    """Validate all files; return a list of problems (empty = valid)."""
    rows = {name: read_jsonl(questions_dir / f) for name, f in SET_FILES.items()}
    english = {r["id"]: r for r in rows["csv"]}
    problems, schema_ok = [], {}
    for name, file_rows in rows.items():
        schema = check_schema(name, file_rows)
        schema_ok[name] = not schema
        problems += schema
        if not schema:
            problems += check_counts(name, file_rows) + check_splits(name, file_rows, english)
    problems += check_duplicates(rows)
    # Re-run gold SQL whenever the English rows are well-formed, even if other checks failed,
    # so a wrong stored answer is always reported.
    if run_sql and schema_ok["csv"]:
        problems += check_answers(rows["csv"], Connections(processed_dir))
    return problems


def summary(questions_dir: Path) -> str:
    """One line per file: rows, train/test, per-difficulty counts."""
    lines = []
    for name, f in SET_FILES.items():
        rows = read_jsonl(questions_dir / f)
        split = Counter(r["split"] for r in rows)
        diff = Counter(r["difficulty"] for r in rows)
        lines.append(f"  {f:30} {len(rows):4} rows | train {split['train']:3} test {split['test']:3} | "
                     + " ".join(f"{d} {diff[d]}" for d in DIFFICULTIES))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build / validate data/questions/*.jsonl")
    parser.add_argument("--check", action="store_true", help="Validate only; do not rewrite files")
    parser.add_argument("--questions-dir", type=Path, default=QUESTIONS_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args(argv)

    if not args.check:
        for message in build(args.questions_dir, args.processed_dir):
            print(f"dropped {message}")
    problems = check(args.questions_dir, args.processed_dir)
    print(summary(args.questions_dir))
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nAll question sets valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
