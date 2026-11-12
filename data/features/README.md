# Answerability features

`answerability_features.csv` is built by `scripts/build_features.py` with
`datacopilot.pipeline.features.extract_features`. One row per question from all four
question sets in `data/questions/` (550 rows: 200 English answerable, 200 English
unanswerable, 100 Hinglish answerable, 50 Hinglish unanswerable).

## Columns

**Metadata and label — never use as model inputs:** `id`, `dataset`, `question_set`,
`split`, `answerable` (the label), `unanswerable_type` (for error analysis only).

**Features** (`FEATURE_NAMES`, all numeric):

| Feature | Meaning |
|---|---|
| `n_words`, `n_content_words` | Question length; content words exclude stopwords, analysis words ("total", "average"), time words and Hinglish function words |
| `schema_overlap` | Share of content words matching a table or column name (case-insensitive, light stemming, fuzzy for words of 5+ letters) |
| `value_overlap` | Share of content words found among the dataset's text values |
| `unmatched_content_words`, `unmatched_share` | Content words matching neither names nor values |
| `has_write_verb`, `starts_with_write_verb`, `mentions_table_word` | Signals of a request to change data (delete, update, drop, ...) |
| `n_opinion_words` | best, should, greatest, achha, behtar, ... |
| `n_external_cues` | CEO, owner, current, population, born, mantri, paida, ... |
| `n_years`, `year_out_of_range`, `years_outside_by` | Years mentioned and how far outside the data's year range they fall ("2000 balls" is a quantity, not a year) |
| `n_numbers` | Other numbers (thresholds, top-N) |
| `n_entities`, `n_unmatched_entities` | Capitalized names, and those not found in the data's values |
| `qtype_*` | One-hot question type: how_many, which_what, who, when_where, yes_no, imperative (incl. Hinglish "... kar do") |
| `hinglish_share` | Share of words that are Hinglish function words (kitne, mein, sabse, ...) |

The year range and value vocabulary come from `build_context(conn, schema)`: date and
year columns (rare outlier years under 0.1% of a column's rows are ignored) and text
columns with at most 20,000 distinct values (id columns and free text are skipped).

## Known caveats

- **`n_words` may be a style artifact.** The unanswerable questions were written shorter
  than the answerable ones (about 7 vs 10 words). A model can learn "short means
  unanswerable" from how the questions were authored rather than from answerability.
  Check its weight in Module 4 and compare against a model trained without it.
- `ends_with_question_mark` was removed for the same reason: written write-requests
  end with "." while every answerable question ends with "?".
- Hinglish rows are unbalanced: 100 answerable vs 50 unanswerable, so across all rows
  `hinglish_share` is higher for answerable questions (0.16 vs 0.08 on train). Within
  Hinglish rows the two classes are similar (0.49 vs 0.41).

## Splits

Train the classifier on `split == "train"` rows only. Test rows are kept here so the
final model can be scored, never for fitting, feature selection or threshold tuning.
