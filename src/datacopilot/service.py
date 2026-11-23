"""Answer one question end to end for the API: gate -> LLM writes SQL -> validator -> executor.

Kept free of HTTP so it can be tested directly; `datacopilot.api.main` is a thin layer on top.
Datasets are opened lazily and cached; each has a lock because one DuckDB / SQLite
connection is shared by the API's worker threads.
"""
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from datacopilot.pipeline.answerability import MODEL_PATH, THRESHOLD, load_model, predict_answerable
from datacopilot.pipeline.features import DatasetContext, build_context
from datacopilot.pipeline.llm import LLMProvider, describe, get_provider
from datacopilot.pipeline.profiler import get_schema, get_sqlite_schema
from datacopilot.pipeline.runner import SQLAttemptError, answer
from datacopilot.pipeline.sql_generator import generate_sql
from datacopilot.questions import CSV_GLOB, DATASETS, DIALECT, PROCESSED_DIR, connect

MAX_RESPONSE_ROWS = 200  # rows sent back to the client (the query itself is capped at executor.MAX_ROWS)
MAX_QUESTION_CHARS = 500


class UnknownDatasetError(KeyError):
    """The dataset name is not one this server knows."""


class QuestionError(ValueError):
    """The question is empty or too long."""


class DatasetUnavailableError(RuntimeError):
    """The dataset is known but its files are not on this server."""


@dataclass
class Dataset:
    """An open dataset: connection, schema and gate context, plus a lock for the connection."""

    name: str
    dialect: str
    conn: object
    schema: dict
    context: DatasetContext
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class DatasetStore:
    """Opens the project datasets from `processed_dir` on first use and keeps them open."""

    def __init__(self, processed_dir: Path | None = None) -> None:
        self.processed_dir = Path(processed_dir or os.getenv("DATA_DIR") or PROCESSED_DIR)
        self._open: dict[str, Dataset] = {}
        self._lock = threading.Lock()

    def available(self, name: str) -> bool:
        """True if the dataset's processed files are present."""
        if name == "olist":
            return (self.processed_dir / "olist.sqlite").is_file()
        return any(self.processed_dir.glob(CSV_GLOB[name]))

    def list(self) -> list[dict]:
        return [{"name": n, "dialect": DIALECT[n], "available": self.available(n)} for n in DATASETS]

    def get(self, name: str) -> Dataset:
        if name not in DATASETS:
            raise UnknownDatasetError(name)
        with self._lock:
            if name not in self._open:
                if not self.available(name):
                    raise DatasetUnavailableError(f"Dataset {name!r} has no data files in {self.processed_dir}")
                conn = connect(name, self.processed_dir, check_same_thread=False)
                schema = get_sqlite_schema(conn) if DIALECT[name] == "sqlite" else get_schema(conn)
                self._open[name] = Dataset(name, DIALECT[name], conn, schema, build_context(conn, schema))
            return self._open[name]


def _rows(df: pd.DataFrame, limit: int) -> list[list]:
    """JSON-safe rows (NaN -> null, numpy and dates -> plain values)."""
    return json.loads(df.head(limit).to_json(orient="values", date_format="iso"))


def ask(
    store: DatasetStore,
    dataset: str,
    question: str,
    bundle: dict | None = None,
    provider: LLMProvider | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict:
    """Answer `question` about `dataset`; returns a JSON-ready dict with status answered/refused/failed.

    LLM errors (llm.LLMError, requests exceptions) propagate so the API can map them to HTTP errors.
    """
    question = question.strip()
    if not question or len(question) > MAX_QUESTION_CHARS:
        raise QuestionError(f"question must be 1 to {MAX_QUESTION_CHARS} characters")
    t0 = clock()
    ds = store.get(dataset)
    bundle = bundle or load_model(Path(os.getenv("MODEL_PATH") or MODEL_PATH))
    provider = provider or get_provider()
    t_open = clock()
    answerable, p = predict_answerable(question, ds.schema, ds.context, bundle)
    t_gate = clock()
    result = {"request_id": uuid.uuid4().hex[:12], "dataset": dataset, "question": question,
              "gate": {"answerable": answerable, "p_answerable": round(p, 4),
                       "threshold": bundle.get("threshold", THRESHOLD)},
              "llm": describe(provider), "sql": None, "columns": [], "rows": [], "row_count": 0,
              "truncated": False, "error": None}
    llm_ms = [0.0]

    def timed_generate(*args, **kwargs) -> str:
        start = clock()
        try:
            return generate_sql(*args, provider=provider, **kwargs)
        finally:
            llm_ms[0] += 1000 * (clock() - start)

    if not answerable:
        result.update(status="refused",
                      message=f"This question does not look answerable from the {dataset} data "
                              f"(P(answerable) = {p:.2f}), so no SQL was run.")
    else:
        try:
            with ds.lock:
                sql, df = answer(ds.conn, question, ds.schema, timed_generate, dialect=ds.dialect)
            result.update(status="answered", sql=sql, columns=[str(c) for c in df.columns],
                          rows=_rows(df, MAX_RESPONSE_ROWS), row_count=len(df),
                          truncated=len(df) > MAX_RESPONSE_ROWS)
        except SQLAttemptError as err:
            result.update(status="failed", sql=err.sql or None, error=str(err))
    t_end = clock()
    result["timings_ms"] = {"open": round(1000 * (t_open - t0), 1), "gate": round(1000 * (t_gate - t_open), 1),
                            "llm": round(llm_ms[0], 1),
                            "sql": round(max(1000 * (t_end - t_gate) - llm_ms[0], 0.0), 1),
                            "total": round(1000 * (t_end - t0), 1)}
    return result
