# Datasets

## BIRD dev set (primary benchmark)
- **Source:** official `dev.zip` linked from https://bird-bench.github.io/
  (`https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip`).
- **Why this version:** it is the dev set that published BIRD results report on,
  so our numbers stay comparable to the leaderboard and papers.
- **Get it:** `python scripts/download_bird.py` saves and unzips it into
  `data/benchmarks/bird_dev/` (gitignored, never committed).

## Possible extra check later: cleaned dev split (2025-11-06)
- In Nov 2025 the BIRD team released a cleaner dev split, `bird_sql_dev_20251106`,
  on Hugging Face: https://huggingface.co/datasets/birdsql/bird_sql_dev_20251106
- It fixes annotation errors found in the original dev set, so some of our "wrong"
  answers on the original may be correct against the cleaned labels.
- **Plan:** not used for the main results. If time allows (after Module 4), rerun
  the evaluation on this split as a secondary check, and label those numbers
  clearly as "dev 20251106" so they are not compared directly with the main results.
