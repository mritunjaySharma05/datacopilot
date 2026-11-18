# Evaluation question sets

Our own question sets for evaluating DataCopilot on the four Module 3 datasets
(`olist`, `ipl`, `f1`, `crop`). Built and validated by `scripts/build_questions.py`.

| File | Rows | Content |
|---|---|---|
| `csv_questions.jsonl` | 200 | 50 answerable questions per dataset (easy / medium / hard) |
| `hinglish_questions.jsonl` | 100 | Hinglish versions of 25 English questions per dataset |
| `unanswerable_questions.jsonl` | 200 | 50 per dataset, 10 of each unanswerable type |
| `hinglish_unanswerable.jsonl` | 100 | Hinglish versions of 100 unanswerable questions (25 per dataset, 5 of each type per dataset) |
| `human_heldout.jsonl` | 30 | Human held-out set: 15 answerable, 15 unanswerable, casual English and Hinglish. **Evaluation only** (see below) |

## Row format (one JSON object per line)

| Field | Meaning |
|---|---|
| `id` | `olist-001` (English), `olist-001-hi` (Hinglish), `olist-u001` (unanswerable), `olist-u001-hi` (Hinglish unanswerable) |
| `dataset` | `olist`, `ipl`, `f1` or `crop` |
| `question` | The natural-language question |
| `gold_sql` | Reference SQL (`null` if unanswerable). Olist: SQLite dialect on `data/processed/olist.sqlite`. Others: DuckDB on the processed CSVs (table = file name, e.g. `ipl_matches`) |
| `gold_answer` | Result of running `gold_sql`: list of rows, floats rounded to 4 places (`null` if unanswerable) |
| `answerable` | `true` / `false` |
| `unanswerable_type` | `missing_column`, `out_of_range`, `external_knowledge`, `subjective`, `write_request`, or `null` |
| `difficulty` | `easy` / `medium` / `hard` (see below) |
| `split` | `train` / `test` |
| `source_id` | Hinglish sets only: id of the English original |

**Difficulty.** Answerable: easy = single filter / count, medium = group by / order /
aggregate, hard = joins (Olist), subqueries, window functions, multi-step logic.
Unanswerable (how hard it is to detect): easy = `write_request`, `subjective`;
medium = `missing_column`, `external_knowledge`; hard = `out_of_range` (needs the data
to see that a year or entity is absent).

## Splits: test questions are off limits

70% train / 30% test, stratified by dataset, answerable and unanswerable type, seed 42
(`datacopilot.questions.assign_splits`). Each Hinglish row (both Hinglish sets) inherits
the split, label, type and difficulty of its English original, so a question and its
translation never end up on different sides of the split.

**Never use test questions for prompt examples, few-shot selection, fine-tuning or
classifier training.** Load examples with `datacopilot.questions.train_examples()`,
which only returns train rows. Test rows are for final evaluation only.

## Rebuilding

    python scripts/build_questions.py          # re-run gold SQL, refresh answers + splits, validate
    python scripts/build_questions.py --check  # validate only (gold SQL re-run must match stored answers)

Needs `data/processed/` (run `scripts/prepare_data.py` first). Answers depend on the
processed data: if the raw data or cleaning changes, rebuild and review the diff.

## Human held-out set (`human_heldout.jsonl`)

30 questions written to look like real user input: lowercase, abbreviations ("avg",
"csk", "gp"), missing question marks, long chatty requests, and Hinglish with varied
spelling ("me" / "mein"). **They were drafted with AI help, then reviewed and edited by
the project author** (2026-11-18); they are not independent user questions, so results on
this set are a robustness check, not a measure of real-user performance.

- Ids `human-001` ... `human-030`, `split = "heldout"`, plus a `language` field
  (`en` / `hinglish`). Labels are the author's; `unanswerable_type` was assigned where the
  type was clear. human-027 (which crop a Rajasthan farmer should grow "given how prices
  have moved") is labeled `subjective` but also needs price data that does not exist.
- Gold SQL was written against `data/processed` and its answers are re-verified by
  `build_questions.py --check`, like the other sets.
- **Never use these rows for training, feature selection, model choice, threshold tuning
  or prompt examples.** They are not in `SET_FILES`, so `train_examples()` and
  `build_features.py` never see them; load them with `datacopilot.questions.load_heldout()`.
- Some questions paraphrase questions in the main sets (e.g. human-009 "avg review score?"
  vs olist-008, human-011 vs olist-009, human-001 vs ipl-018). The texts differ, but the
  intent overlaps, so the held-out set is not fully independent of the training data.
