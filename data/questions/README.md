# Evaluation question sets

Our own question sets for evaluating DataCopilot on the four Module 3 datasets
(`olist`, `ipl`, `f1`, `crop`). Built and validated by `scripts/build_questions.py`.

| File | Rows | Content |
|---|---|---|
| `csv_questions.jsonl` | 200 | 50 answerable questions per dataset (easy / medium / hard) |
| `hinglish_questions.jsonl` | 100 | Hinglish versions of 25 English questions per dataset |
| `unanswerable_questions.jsonl` | 200 | 50 per dataset, 10 of each unanswerable type |

## Row format (one JSON object per line)

| Field | Meaning |
|---|---|
| `id` | `olist-001` (English), `olist-001-hi` (Hinglish), `olist-u001` (unanswerable) |
| `dataset` | `olist`, `ipl`, `f1` or `crop` |
| `question` | The natural-language question |
| `gold_sql` | Reference SQL (`null` if unanswerable). Olist: SQLite dialect on `data/processed/olist.sqlite`. Others: DuckDB on the processed CSVs (table = file name, e.g. `ipl_matches`) |
| `gold_answer` | Result of running `gold_sql`: list of rows, floats rounded to 4 places (`null` if unanswerable) |
| `answerable` | `true` / `false` |
| `unanswerable_type` | `missing_column`, `out_of_range`, `external_knowledge`, `subjective`, `write_request`, or `null` |
| `difficulty` | `easy` / `medium` / `hard` (see below) |
| `split` | `train` / `test` |
| `source_id` | Hinglish only: id of the English original |

**Difficulty.** Answerable: easy = single filter / count, medium = group by / order /
aggregate, hard = joins (Olist), subqueries, window functions, multi-step logic.
Unanswerable (how hard it is to detect): easy = `write_request`, `subjective`;
medium = `missing_column`, `external_knowledge`; hard = `out_of_range` (needs the data
to see that a year or entity is absent).

## Splits: test questions are off limits

70% train / 30% test, stratified by dataset, answerable and unanswerable type, seed 42
(`datacopilot.questions.assign_splits`). Each Hinglish row inherits the split of its
English original, so the same gold SQL never appears in both splits.

**Never use test questions for prompt examples, few-shot selection, fine-tuning or
classifier training.** Load examples with `datacopilot.questions.train_examples()`,
which only returns train rows. Test rows are for final evaluation only.

## Rebuilding

    python scripts/build_questions.py          # re-run gold SQL, refresh answers + splits, validate
    python scripts/build_questions.py --check  # validate only (gold SQL re-run must match stored answers)

Needs `data/processed/` (run `scripts/prepare_data.py` first). Answers depend on the
processed data: if the raw data or cleaning changes, rebuild and review the diff.
