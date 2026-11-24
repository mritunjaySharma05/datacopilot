"""Tests for scripts/compare_providers.py on hand-made runs."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("compare_providers", SCRIPTS / "compare_providers.py")
cp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cp)


def run(results: dict[str, tuple[bool, bool, str]], seconds: float) -> pd.DataFrame:
    """Fake eval_questions output: id -> (strict, relaxed, status)."""
    return pd.DataFrame([{"id": i, "question_set": "hinglish" if i.endswith("-hi") else "csv",
                          "dataset": i.split("-")[0], "difficulty": "easy", "question": f"q {i}?",
                          "strict": s, "relaxed": r, "status": st, "pred_sql": "SELECT 1", "error": "",
                          "seconds": seconds} for i, (s, r, st) in results.items()])


OLLAMA = run({"crop-001": (True, True, "correct"), "crop-002": (False, False, "validation_failed"),
              "ipl-001": (False, True, "wrong_result"), "ipl-001-hi": (True, True, "correct")}, 10.0)
GROQ = run({"crop-001": (True, True, "correct"), "crop-002": (True, True, "correct"),
            "ipl-001": (True, True, "correct"), "ipl-001-hi": (False, False, "wrong_result")}, 2.0)


def test_pair_names_columns_by_role_and_keeps_seconds():
    p = cp.pair(OLLAMA, GROQ).set_index("id")
    assert p.loc["crop-002", ["strict_c", "strict_r"]].tolist() == [True, False]
    assert p.loc["ipl-001-hi", "language"] == "hinglish"
    assert (p.seconds_c == 2.0).all() and (p.seconds_r == 10.0).all()


def test_pair_rejects_different_questions():
    with pytest.raises(ValueError):
        cp.pair(OLLAMA, GROQ.iloc[:3])


def test_by_group_and_mcnemar():
    p = cp.pair(OLLAMA, GROQ)
    t = cp.by_group(p, "language")
    assert t.loc["all", "strict_c"] == 0.75 and t.loc["all", "strict_r"] == 0.5 and t.loc["en", "n"] == 3
    m = cp.mcnemar(p, "strict")
    assert (m["only_candidate"], m["only_reference"], m["discordant"]) == (2, 1, 3)
    assert m["p_value"] == pytest.approx(1.0)


def test_report_sections():
    text = cp.report(cp.pair(OLLAMA, GROQ), "Ollama", "Groq", "o.csv", "g.csv")
    assert "**Strict: Groq 75.0% vs Ollama 50.0%** (3 vs 2 of 4)" in text
    assert "| **all** | 4 | 50.0% | 75.0% | 75.0% | 75.0% |" in text
    assert "| validation_failed | 1 | 0 |" in text
    assert "| ipl-001-hi | q ipl-001-hi? | Ollama | correct | wrong_result |" in text
    assert "| median | 10.0 | 2.0 |" in text


def test_main_picks_rescored_ollama_and_groq_runs(tmp_path):
    OLLAMA.to_csv(tmp_path / "questions_validated_csv+hinglish_test_20261118_225419.csv", index=False)
    OLLAMA.to_csv(tmp_path / "questions_validated_csv+hinglish_test_20261118_225419_rescored.csv", index=False)
    GROQ.to_csv(tmp_path / "questions_validated-groq_csv+hinglish_test_20261124_120000.csv", index=False)
    out = tmp_path / "r.md"
    assert cp.main(["--results-dir", str(tmp_path), "--out", str(out)]) == 0
    assert "_rescored.csv" in out.read_text() and "validated-groq" in out.read_text()
