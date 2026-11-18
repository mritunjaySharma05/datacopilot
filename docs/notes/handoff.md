# Handoff: DataCopilot, state as of 2026-11-18

For a fresh Claude Code session on a new machine (MacBook M2, 16 GB). Read this first,
then `README.md`, `data/questions/README.md` and `data/features/README.md`.

Repo: https://github.com/mritunjaySharma05/datacopilot (branch `main`, 44 commits,
233 tests passing on Windows / Python 3.13.2).

---

## 1. First steps on the new machine

**Two local files are gitignored and will NOT come with the clone. Restore them first**
(see the last line of this file):

1. `CLAUDE.md` (project rules for Claude Code)
2. `.claude/settings.json` (turns off Co-Authored-By / AI attribution in commits)

Then set up the environment:

```bash
git clone https://github.com/mritunjaySharma05/datacopilot.git && cd datacopilot
python3 -m venv .venv && source .venv/bin/activate      # Python 3.11+ (3.13 used so far)
brew install libomp                                      # xgboost on Apple Silicon needs it
pip install -r requirements.txt                          # top-level versions pinned
cp .env.example .env                                     # then fill DATA_GOV_IN_API_KEY only if needed
```

Ollama (Apple Silicon runs it on the GPU via Metal; 16 GB is enough for a 7B Q4 model):

```bash
brew install ollama   # or the macOS app from ollama.com
ollama serve &        # the app starts it automatically
ollama pull qwen2.5-coder:7b
```

`.env` defaults: `OLLAMA_HOST=http://localhost:11434`, `OLLAMA_MODEL=qwen2.5-coder:7b`,
`OLLAMA_NUM_CTX=8192` (needed: Ollama's default context silently truncates wide schemas).

**Rebuild the gitignored data** (nothing under `data/raw`, `data/processed`,
`data/benchmarks`, `data/results` is in git):

```bash
python scripts/download_bird.py          # BIRD dev set, ~346 MB zip, ~1.8 GB unzipped
python scripts/download_datasets.py      # Olist, IPL, F1 automatically (~200 MB)
# Crop: download manually (see 1a), then:
python scripts/data_quality.py           # optional: regenerates docs/data_quality/*.md
python scripts/prepare_data.py           # -> data/processed/ (olist.sqlite + CSVs)
python scripts/build_questions.py --check   # gold answers must still match: verifies the rebuild
python scripts/build_features.py         # regenerates data/features/answerability_features.csv
pytest
```

**1a. Crop data is a manual download.** It came from the Kaggle mirror
`abhinand05/crop-production-in-india` (sign in to Kaggle, download, save as
`data/raw/crop_production/crop_production.csv`; expected 15,316,741 bytes, 246,091 rows).
The official data.gov.in route needs a free API key (`DATA_GOV_IN_API_KEY`); see
`docs/planning/data_sources.md`.

If `build_questions.py --check` reports "stored gold_answer differs", the rebuilt data is
not identical to the original (e.g. a newer Jolpica F1 dump or Cricsheet update). Do not
blindly rebuild answers: diff first, then decide.

---

## 2. Module status

| Module | Status |
|---|---|
| 1. Proposal, charter, architecture, repo setup | Submitted |
| 2. Research, design, basic prototype | Submitted |
| 3. Data collection, cleaning, EDA, question sets, features | Submitted (report: `docs/reports/DataCopilot_Module3_Submission.pdf`, committed 2026-11-12) |
| 4. Answerability classifier + baseline vs validated experiments | **In progress.** Classifier trained (step 2 done 2026-11-17); experiments next |
| 5-8 | Not started |

### What exists (by area)

**Pipeline** (`src/datacopilot/pipeline/`)
- `profiler.py`: `get_schema` (DuckDB) / `get_sqlite_schema` (SQLite): tables, columns,
  types, 3 sample values; `format_schema` for prompts (long samples truncated).
- `sql_generator.py`: Ollama `/api/generate`, temperature 0, `num_ctx` from env; prompt
  names the dialect (DuckDB / SQLite), optional evidence hint and retry feedback;
  `extract_sql` handles ```sql blocks and prose.
- `validator.py`: sqlglot. `is_read_only` (single SELECT; UNION / INTERSECT / EXCEPT
  allowed; blocks writes anywhere incl. CTEs and `SELECT ... INTO`), `find_syntax_error`
  (line/column message for retry feedback), `check_schema` (unknown tables / columns,
  unqualified columns must exist in a table the query uses), `validate`.
- `executor.py`: `load_csv` into DuckDB, `connect_sqlite` (read-only `mode=ro`,
  per-statement timeout), `run_query` (read-only guard, row limit 1000 or `None`).
- `runner.py`: `answer()` = generate → validate → execute, ONE retry with the validation
  or execution error; `use_validator=False` = baseline (one attempt, no schema check,
  read-only guard kept).
- `features.py`: answerability features (see `data/features/README.md`).
- `answerability.py`: grouped-CV model search (logreg / RF / XGBoost), `evaluate`,
  `save_model` / `load_model`, `predict_answerable`. Model in `models/answerability.joblib`
  (3 KB, committed; rebuild with `scripts/train_answerability.py`). Results: `docs/results/`.
- `explainer.py`: stub (Module 6).

**Scripts** (`scripts/`)
- `ask.py`: CLI, CSV question → SQL + result (`python scripts/ask.py data/sample/sales.csv "..."`).
- `download_bird.py`, `eval_bird.py`: BIRD dev execution accuracy, `--n`, `--no-validator`;
  per-question CSVs in `data/results/`.
- `download_datasets.py`, `data_quality.py`, `prepare_data.py`: Module 3 data pipeline.
- `build_questions.py`: build / `--check` the question sets.
- `build_features.py`: feature table + train-split class comparison.

**Data and docs in git**
- `data/sample/sales.csv` (50-row demo), `data/questions/*.jsonl`, `data/features/*.csv`.
- `docs/data_quality/` (4 reports + cleaning log), `docs/eda/` (6 report charts),
  `notebooks/03_eda.ipynb` (executed), `docs/planning/` (milestones, datasets, sources).

### Key numbers
- BIRD dev, first 10 questions (all `california_schools`), qwen2.5-coder:7b:
  **validated 7/10, baseline 6/10** (2026-11-11). Smoke test only, not a benchmark.
- Question sets: 200 English answerable, 100 Hinglish answerable, 200 English
  unanswerable, 100 Hinglish unanswerable (50 added 2026-11-17); 70/30 train/test (seed 42).
- Human held-out set: 30 questions (`human_heldout.jsonl`, AI-drafted, human-edited,
  split "heldout", evaluation only). It was used once to decide to drop `n_words`, so it
  is no longer fully untouched. Deployed classifier (XGBoost, 23 features, no `n_words`):
  test 87.2%, held-out 86.7% (`docs/results/README.md`).
- Features: 600 rows x 24 features. Strongest train-split signals: `schema_overlap`
  (0.54 answerable vs 0.29 unanswerable), `unmatched_share`, write-verb / imperative,
  opinion and external-cue counts, out-of-range years, unmatched entities.

---

## 3. Key design decisions (and why)

1. **LLM writes SQL only; the database computes every number.** Core project rule.
2. **Read-only enforced twice:** validator rejects non-SELECT, and the executor refuses
   anything `is_read_only` rejects; SQLite opened with `mode=ro` as a third layer.
3. **One retry with the exact error** (validation or execution). Feeding a precise error
   back fixed real BIRD failures; a misleading message ("not read-only" for a syntax
   error) made the model repeat itself, hence `find_syntax_error`.
4. **Unqualified columns must exist in a table the query uses.** SQLite turns an unknown
   `"double-quoted"` name into a string literal and silently returns wrong data.
5. **Baseline = no schema check, no retry, read-only guard still on** (CLAUDE.md forbids
   ever executing writes).
6. **Execution accuracy = same set of rows** (BIRD's official rule; order/duplicates ignored).
7. **SQLite via stdlib `sqlite3`**, not SQLAlchemy yet (not installed; needed for PostgreSQL later).
8. **BIRD original `dev.zip`** is the benchmark (published results use it); the cleaned
   `bird_sql_dev_20251106` split is an optional secondary check (`docs/planning/datasets.md`).
9. **F1 from Jolpica-F1** (Ergast successor; free delayed dump, CC BY-NC-SA 4.0), seasons
   2014+ in processed data. **Olist from Kaggle's public endpoint**, crop from a Kaggle mirror.
10. **Cleaning keeps renames visible:** IPL team names kept as played, `*_franchise`
    columns hold current names; renamed grounds (Feroz Shah Kotla vs Arun Jaitley) NOT merged.
11. **IPL `ball` dropped** in processed CSVs; use `over` + `ball_in_over` ("2.10" read as a
    float collides with "2.1").
12. **Question splits:** seeded, stratified by dataset / answerable / unanswerable type.
    Hinglish rows inherit their English source's split, so a question and its translation
    never straddle train/test. **Test rows are never used for prompts, few-shot or
    training**; use `datacopilot.questions.train_examples()`.
13. **Gold answers are stored and re-verified** (`build_questions.py --check`), so data
    drift is caught instead of silently changing the benchmark.
14. **Features use a data context**, not just 3 schema samples: year range (rare outlier
    years ignored) and value vocabulary from text columns.
15. **Commits:** small, clear messages, never AI attribution (history was rewritten once
    on 2026-11-11 to remove old Co-Authored-By trailers).

---

## 4. Known issues

**Data quirks (handled or documented)**
- Crop: coconut production is in nuts, not tonnes (never sum across crops); 2014 is
  missing some states (e.g. Tamil Nadu, Madhya Pradesh rice), 2015 has 562 rows; 3,730
  rows lack production; Hoshangabad 2012 rice (6.64 t/ha) is almost certainly a data error.
- Olist: 2016 and Sep/Oct 2018 are partial months; 4 `shipping_limit_date` values in
  2020; geolocation had ~280k duplicate rows (removed) and stray-symbol city names.
- F1: 2026 season in progress; pit "duration" is pit-lane time; ~540 race "stops" over
  120 s are red-flag stoppages; some 12 s pit-lane times look like timing artifacts;
  `status_code` is an integer enum defined only in Jolpica's source.
- IPL: 25 matches without a winner; 23 D/L-affected matches.

**Code / tooling caveats**
- `requirements.txt` pins top-level packages only (2026-11-15); transitive dependencies
  are not locked, so a fresh install may differ slightly between machines.
- `prepare_data.py <one dataset>` overwrites `docs/data_quality/cleaning_log.md` with only
  that dataset's steps; run it with no arguments for the full log.
- **Feature vocabulary gap (known limitation, for Module 6; decision 2026-11-18: do NOT
  add features based on held-out misclassifications).** Derived-metric words ("sixes",
  "economy", "podiums") are not column names, so `schema_overlap` is 0, and lowercase
  player names ("kohli", "bumrah", "verstappen") are not counted as entities although they
  match data values ("V Kohli"). Short Hinglish metric questions get refused (human-002,
  human-004). Fix in Module 6 and validate on new questions, not the held-out set.
- Question-feature caveats: `n_words` may be a writing-style artifact (unanswerable
  questions are shorter); Hinglish rows are 100 answerable vs 50 unanswerable, so overall
  `hinglish_share` leans answerable; "India" is treated as generic, so "races in India"
  (F1) isn't flagged by the entity feature.
- `data/questions/review.csv` was a local review sheet (excluded via `.git/info/exclude`);
  all rows were reviewed and applied. Not needed on the new machine.

---

## 5. Error log (what went wrong, so it isn't repeated)

| Date | What happened | Fix / lesson |
|---|---|---|
| 11-09 | A stray `cat >` in a shell command waited on stdin, hung, and created an empty file outside the repo | Removed; never start commands with a bare redirect |
| 11-11 | `grep -v` in a `&&` commit chain returned exit 1 when it filtered everything, stopping the chain after one commit | Don't put filters that can "fail" inside `&&` chains |
| 11-11 | Ollama wasn't running after a reboot; eval reported "Could not reach Ollama" | Start `ollama serve` (or the app) before LLM runs |
| 11-11 | First BIRD run: validator mislabeled syntax errors as "not read-only", and accepted columns from tables not in the query (SQLite string-literal fallback) | Added `find_syntax_error` and scoped column check; rerun 6/10 → 7/10 |
| 11-11 | Commit message for download_bird tests said "skip/resume" (no resume exists) | Left as is (no force-push); write messages from the diff |
| 11-11 | Old commits carried Co-Authored-By trailers | History rewritten with backup branch + `--force-with-lease`; attribution disabled in `.claude/settings.json` + CLAUDE.md rule |
| 11-11 | pandas 3 reads text as `str` dtype, not `object`; dtype checks silently found nothing | Use `pd.api.types.is_string_dtype` |
| 11-11 | EDA takeaways drafted before checking outputs had 5 wrong numbers | Always verify every stated number against executed output |
| 11-11 | `review.csv` write failed: file open in Excel (Windows lock) | Close Excel before scripted edits |
| 11-12 | Inline heredocs with apostrophes broke in the Bash tool; `\b` inside a heredoc became a literal backspace char, silently breaking a regex | Put edit scripts in files; check for `\x08` after scripted edits; tests caught it |
| 11-12 | Question `crop-u015` (Ladakh) labeled out-of-range, but LEH LADAKH is a district in the data | Replaced with Bhutan (verified absent); always check states AND districts |

---

## 6. Open decisions (need the user)

1. ~~Hinglish class balance~~ Resolved 2026-11-17: 50 Hinglish unanswerable questions
   added (100 total, 5 per dataset x type).
2. ~~`n_words`~~ Resolved 2026-11-18: dropped from the deployed model after the fair
   logreg ablation on the human held-out set (66.7% with it vs 83.3% without; test 90.6%
   vs 90.0%). Retrained with the same grouped CV: XGBoost, test 87.2%, held-out 86.7%.
   **The held-out set was used once, for this decision only; it is no longer fully
   untouched** (see `docs/results/README.md`). Do not tune anything else on it.
3. ~~Module 3 status in `docs/planning/milestones.md`~~ Resolved 2026-11-15: Module 3
   "Submitted", Module 4 "In progress".
4. ~~BIRD scale~~ Resolved 2026-11-17: fixed-seed sample of 300 dev questions stratified
   across all 11 databases; cleaned 20251106 split skipped for now.
5. ~~Pin dependencies now or at Module 5.~~ Resolved 2026-11-15: top-level packages
   pinned in `requirements.txt` to the versions installed on the Windows machine.
6. ~~SQLAlchemy~~ Resolved 2026-11-17: keep stdlib `sqlite3` until PostgreSQL work.

---

## 7. Module 4: exact next steps

Goal (milestones): **answerability classifier + baseline vs validated pipeline experiments.**

1. **Setup check** (section 1): data rebuilt, `build_questions.py --check` passes,
   `pytest` green, Ollama responds (`python scripts/ask.py data/sample/sales.csv "Which city had the highest total revenue?"` → Mumbai, 1,185,500).
2. **DONE 2026-11-17. Train the classifier** (new `scripts/train_answerability.py`, `src/datacopilot/pipeline/answerability.py`):
   - Input: `data/features/answerability_features.csv`, columns `FEATURE_NAMES` only;
     label `answerable`; **fit on `split == "train"` only**.
   - Models: logistic regression (baseline, interpretable), random forest, XGBoost
     (all in `requirements.txt`). Stratified 5-fold CV on train for model choice and
     hyperparameters; no test rows in CV, feature selection or threshold tuning.
   - Report on test once: accuracy, precision/recall/F1 for the unanswerable class,
     confusion matrix, and recall per `unanswerable_type` and per `question_set`
     (English vs Hinglish).
   - Ablations: without `n_words`; English-only vs with Hinglish.
   - Save the model (e.g. `models/answerability.joblib`, decide whether to commit) and
     add `predict_answerable(question, schema, context) -> (bool, probability)`.
   - Tests for every new function (CLAUDE.md rule).
3. **DONE 2026-11-18 (results in `docs/results/README.md`: validated = baseline, 33.3% strict on
   test, McNemar p = 1.0; relaxed metric redefined after seeing results, BIRD metric unchanged).
   Our-question-set evaluation** (new `scripts/eval_questions.py`, modelled on `eval_bird.py`):
   - Run `csv_questions` + `hinglish_questions` (test split for the final numbers) through
     `runner.answer` in validated and baseline modes, compare with stored `gold_answer`
     using `datacopilot.questions.answers_match`.
   - Olist uses dialect `sqlite` on `olist.sqlite`; others `duckdb` on processed CSVs
     (`datacopilot.questions.connect` already does this).
4. **DONE 2026-11-18 (`docs/results/gate.md`: end-to-end strict 28.9% to 57.8% on test,
   12 false refusals, McNemar p < 0.0001).** **Answerability gate experiment:** classifier first; if "unanswerable", refuse instead of
   generating SQL. Measure correct refusals on the unanswerable sets and false refusals on
   answerable ones, and end-to-end accuracy with vs without the gate.
5. **BIRD:** rerun `eval_bird.py` on a larger agreed subset in both modes (decision 4).
6. **Write up** results tables + error analysis for the Module 4 report; update milestones.

Few-shot prompting, if tried, must draw examples from `train_examples()` only.

---

Copy CLAUDE.md and .claude/settings.json from the private Google Drive folder datacopilot-private.
