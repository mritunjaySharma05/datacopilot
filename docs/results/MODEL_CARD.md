# Model card: DataCopilot answerability classifier

| | |
|---|---|
| File | `models/answerability.joblib` (about 3 KB, committed): a dict with `model`, `feature_names`, `threshold`, `name`, `params`, `cv_macro_f1`, `n_train` |
| Code | `src/datacopilot/pipeline/answerability.py` (`predict_answerable`), features in `src/datacopilot/pipeline/features.py` |
| Model | XGBoost classifier (`xgboost.XGBClassifier`), 23 hand-made features |
| Trained | 2026-11-18, by `scripts/train_answerability.py` (experiment `main`) |
| Version note | Replaces the first model (2026-11-17, logistic regression with `n_words`) |

## Purpose

DataCopilot answers natural-language questions about a dataset by having an LLM write SQL.
Before any SQL is generated, this classifier decides whether the question **can be answered
from the dataset's schema and data at all**. If P(answerable) < 0.5 the pipeline refuses
instead of producing an answer to a question it cannot answer (in step 4, without the gate,
68 of 90 unanswerable test questions got an invented answer; with it, 10).

**Intended use:** a cheap pre-filter in front of the text-to-SQL pipeline for the four
project datasets (Olist e-commerce, IPL cricket, Formula 1, Indian crop production), English
and Hinglish questions. **Not intended** as a general-purpose classifier for other schemas
without re-validation, or as a security control: write requests are blocked by the SQL
validator (SELECT only) regardless of this model.

Output: `predict_answerable(question, schema, context) -> (answerable: bool, P(answerable))`.
The positive class in the reported precision/recall is **unanswerable** (the class the gate
must catch).

## Features

23 numeric features computed from the question, the schema (table and column names) and a
small dataset context (year range, text values). Full definitions:
[`data/features/README.md`](../../data/features/README.md).

Groups: question length in content words; overlap of question words with schema names and
data values; unmatched words and capitalised entities; write-request signals; opinion and
external-knowledge cue words; years and whether they fall outside the data's range; other
numbers; question type (one-hot); share of Hinglish function words.

**Excluded:** `n_words` (raw question length). In our authored sets unanswerable questions
are shorter (about 7 vs 10 words), so length predicted the label on our test split but not on
human-written questions (held-out accuracy 66.7% with it vs 83.3% without, logistic regression).
It was dropped on 2026-11-18 (`answerability.EXCLUDED_FEATURES`).

Top 5 features by XGBoost importance (gain share): `qtype_imperative` 0.173,
`starts_with_write_verb` 0.158, `year_out_of_range` 0.133, `unmatched_share` 0.087,
`n_numbers` 0.069. Full list: `answerability_metrics.json`, figure
`figures/feature_importance.png`.

## Training data and split

`data/features/answerability_features.csv`: 600 questions written for this project
(`data/questions/`), balanced 50/50:

| Set | Train | Test |
|---|---|---|
| English answerable (`csv`) | 140 | 60 |
| Hinglish answerable (`hinglish`) | 70 | 30 |
| English unanswerable (`unanswerable`) | 140 | 60 |
| Hinglish unanswerable (`hinglish_unanswerable`) | 70 | 30 |
| **Total** | **420** | **180** |

Unanswerable types: missing column, out of range, external knowledge, subjective, write
request. The split was fixed when the question sets were written (about 70/30 per dataset);
a Hinglish translation `x-hi` always shares the split of its English source `x` (0 groups
cross train/test).

A separate **human held-out set** (`data/questions/human_heldout.jsonl`, 30 questions, 15/15,
AI-drafted and human-edited) is never used for training.

## Training procedure and hyperparameters

1. Grouped, stratified 5-fold CV (`StratifiedGroupKFold`, shuffled, seed 42) on the 420 train
   rows only; groups = a question and its Hinglish translation; score = macro F1.
2. Grid search over three families (25 configurations): logistic regression
   (C in {0.01, 0.1, 1, 10}, standardised), random forest (300 trees; max_depth in {None, 5, 10},
   min_samples_leaf in {1, 3, 5}), XGBoost (n_estimators in {100, 300}, max_depth in {2, 3, 5},
   learning_rate in {0.05, 0.1}).
3. The best mean CV score wins; it is refit on all 420 train rows.

**Chosen:** XGBoost, `n_estimators=300`, `max_depth=5`, `learning_rate=0.05`,
`eval_metric="logloss"`, `random_state=42`, `n_jobs=1`; CV macro F1 0.886 ± 0.033.

| Family (best config) | CV macro F1 (train) | Test macro F1 |
|---|---|---|
| Logistic regression (C = 0.1) | 0.869 ± 0.051 | 0.900 |
| Random forest (max_depth 10, min_samples_leaf 1) | 0.881 ± 0.048 | 0.900 |
| **XGBoost (deployed)** | **0.886 ± 0.033** | 0.872 |

Test scores per family are reported, not used for selection: choosing by test would leak.
The three families are within one CV standard deviation of each other.

**Threshold: 0.5** on P(answerable), the default; **not tuned**. Any future tuning must use
CV on the train split, never test or held-out rows.

## Metrics

Test split (180 questions, scored once; positive class = unanswerable):

| Metric | Value |
|---|---|
| Accuracy | 87.2% |
| Macro F1 | 0.872 |
| Unanswerable precision / recall | 86.8% / 87.8% |
| ROC AUC | 0.951 |
| Average precision | 0.957 |
| Confusion matrix (rows true, columns predicted; unanswerable, answerable) | [[79, 11], [12, 78]] |

By subgroup (test accuracy): English answerable 90.0%, Hinglish answerable 80.0%, English
unanswerable 86.7%, Hinglish unanswerable 90.0%; Crop 91.1%, F1 93.5%, IPL 79.5%, Olist
84.4%. Recall by unanswerable type: subjective 100%, write request 100%, out of range 94.1%,
external knowledge 88.2%, **missing column 60.0%**.

Human held-out set (30): accuracy 86.7%, unanswerable precision 82.4%, recall 93.3%.

In the pipeline (step 4, `gate.md`): end-to-end strict accuracy 28.9% without vs 57.8% with
the gate on the test split (exact McNemar p < 0.0001), at the cost of 12 false refusals of 90
answerable questions.

Figures: `figures/confusion_matrix.png`, `figures/roc_pr.png`, `figures/feature_importance.png`,
`figures/model_families.png`, `figures/answerability.png`.

## Limitations

- **Small, self-written data.** 420 training questions written by the project author for four
  datasets. The 50/50 class balance does not reflect real use, where most questions are
  answerable, so the share of false refusals in practice will be higher than here.
- **Held-out set reused once.** It was used to decide to drop `n_words`, so its 86.7% is
  somewhat optimistic. Nothing else was tuned on it. It is also small (one question = 3.3
  points) and AI-drafted, with some paraphrases of training questions.
- **Vocabulary gap.** Questions about computed metrics ("sixes", "economy", "podiums") and
  lowercase names ("kohli") match no column name, so `schema_overlap` is 0 and short
  questions of this kind are refused. Planned for Module 6.
- **`missing_column` is the weakest type** (60% recall on test): questions whose words partly
  match real columns ("average age of customers") pass the gate.
- **Hinglish answerable questions are refused more often** (80.0% vs 90.0% English).
- **Lexical features only.** No semantic model; spelling variants, synonyms and new datasets
  with unusual column names may behave differently. Re-validate before using on new schemas.
- The gate is not a safety mechanism: the SQL validator alone guarantees read-only execution.

## How to reproduce

From the repository root, Python 3.13:

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Train (reads the committed data/features/answerability_features.csv only):
#    CV model selection, refit, test scoring, ablations; writes models/answerability.joblib,
#    docs/results/answerability.md and answerability_metrics.json
python scripts/train_answerability.py

# 3. Optional, to rebuild the feature table and score the held-out set, the processed data is
#    needed (Crop is a manual download, see docs/notes/handoff.md section 1a):
python scripts/download_datasets.py
python scripts/prepare_data.py
python scripts/build_questions.py --check   # gold answers must still match
python scripts/build_features.py            # regenerates data/features/answerability_features.csv
python scripts/eval_heldout.py              # writes docs/results/heldout.md and heldout_metrics.json

# 4. Figures (classifier and answerability figures need only steps 1-2; the gate, our-sets and
#    BIRD figures need the evaluation runs in data/results/ and are skipped with a message if absent)
python scripts/make_figures.py

# 5. Step-by-step notebook (executes the same procedure; held-out read only in its last cell)
jupyter nbconvert --to notebook --execute --inplace notebooks/module4_model_development.ipynb

# 6. Tests
pytest
```

Training is deterministic (seed 42, `n_jobs=1`): the notebook's refit model gives the same
test probabilities as the committed `models/answerability.joblib`.
