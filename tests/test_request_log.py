"""Tests for structured request logging (datacopilot.request_log) and the /ask log line."""
import io
import json
import logging

import pytest
from fastapi.testclient import TestClient

from datacopilot import request_log, service
from datacopilot.api import main as api
from datacopilot.pipeline.llm import LLMError, LLMRateLimitError
from test_service import BUNDLE, FakeProvider, store  # noqa: F401  (store is a fixture)

KEY = "gsk_SECRETsecret1234567890"


def lines(buf: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in buf.getvalue().splitlines() if line.strip()]


# ---------- redaction and formatting ----------

def test_redact_env_secrets_patterns_nested_and_long(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "abc123xyz789")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5-coder:7b")  # not secret-looking: kept
    out = request_log.redact({"a": "key abc123xyz789 here", "b": ["Bearer tok.en-123456789", KEY],
                              "c": "qwen2.5-coder:7b", "n": 3, "long": "x" * 3000})
    assert out["a"] == "key [REDACTED] here" and out["b"] == ["[REDACTED]", "[REDACTED]"]
    assert out["c"] == "qwen2.5-coder:7b" and out["n"] == 3
    assert len(out["long"]) < 2100 and out["long"].endswith("[truncated]")


def test_secret_values_ignores_short_and_empty():
    assert request_log.secret_values({"A_KEY": "", "B_TOKEN": "12345", "DB_PASSWORD": "hunter22", "MODEL": "m" * 9}) \
        == ["hunter22"]


def test_json_formatter_one_line_with_extra_fields():
    buf = io.StringIO()
    logger = request_log.configure(buf)
    request_log.configure(buf)  # idempotent: still one handler
    assert len(logger.handlers) == 1
    logger.info("ask", extra={"request_id": "r1", "timings_ms": {"total": 1.5}})
    [line] = lines(buf)
    assert line["event"] == "ask" and line["level"] == "INFO" and line["logger"] == "datacopilot.requests"
    assert line["request_id"] == "r1" and line["timings_ms"] == {"total": 1.5} and line["ts"].endswith("+00:00")


def test_ask_record_from_error_without_result():
    r = request_log.ask_record(None, dataset="crop", question="q", http_status=503, error="down", latency_ms=4.0)
    assert (r["status"], r["http_status"], r["error"], r["latency_ms"], r["sql"]) == ("error", 503, "down", 4.0, None)


# ---------- /ask writes one line per request ----------

@pytest.fixture
def logged(store, monkeypatch):  # noqa: F811
    buf = io.StringIO()
    request_log.configure(buf)
    holder = {"provider": FakeProvider()}
    monkeypatch.setattr(api, "get_store", lambda: store)
    monkeypatch.setattr(service, "get_provider", lambda: holder["provider"])
    monkeypatch.setattr(service, "load_model", lambda path: BUNDLE)
    yield TestClient(api.app), buf, holder
    request_log.configure()


def test_answered_request_logs_gate_sql_and_latency(logged):
    client, buf, _ = logged
    client.post("/ask", json={"dataset": "crop", "question": "How many states?"})
    client.get("/health")  # not logged
    [line] = lines(buf)
    assert line["event"] == "ask" and line["status"] == "answered" and line["http_status"] == 200
    assert line["question"] == "How many states?" and line["dataset"] == "crop"
    assert line["gate_answerable"] is True and line["p_answerable"] == 0.9
    assert line["sql"].startswith("SELECT COUNT(DISTINCT state_name)") and line["row_count"] == 1
    assert (line["llm_provider"], line["llm_model"]) == ("fake", "fake-model")
    assert line["latency_ms"] >= 0 and set(line["timings_ms"]) == {"open", "gate", "llm", "sql", "total"}
    assert len(line["request_id"]) == 12


def test_refused_request_logged_without_sql(logged):
    client, buf, _ = logged
    client.post("/ask", json={"dataset": "crop", "question": "Delete all rows for Kerala"})
    [line] = lines(buf)
    assert line["status"] == "refused" and line["gate_answerable"] is False and line["sql"] is None


def test_errors_logged_as_warning_with_http_status(logged):
    client, buf, holder = logged
    holder["provider"] = FakeProvider(error=LLMRateLimitError("Groq rate limit reached (HTTP 429).", 3.0))
    client.post("/ask", json={"dataset": "crop", "question": "How many states?"})
    client.post("/ask", json={"dataset": "nope", "question": "q?"})
    first, second = lines(buf)
    assert (first["level"], first["http_status"], first["status"]) == ("WARNING", 429, "error")
    assert (second["http_status"], second["error"]) == (404, "Unknown dataset 'nope'")


def test_secret_never_reaches_log_or_response(logged, monkeypatch):
    client, buf, holder = logged
    monkeypatch.setenv("GROQ_API_KEY", KEY)
    holder["provider"] = FakeProvider(error=LLMError(f"provider said: bad key {KEY}"))  # a leaky error message
    r = client.post("/ask", json={"dataset": "crop", "question": f"what is {KEY}?"})
    assert r.status_code == 503 and KEY not in r.text
    assert KEY not in buf.getvalue() and "[REDACTED]" in buf.getvalue()


def test_unexpected_error_returns_500_and_is_logged(logged, monkeypatch):
    client, buf, _ = logged

    def boom(*a, **k):
        raise RuntimeError("bug")
    monkeypatch.setattr(service, "ask", boom)
    r = client.post("/ask", json={"dataset": "crop", "question": "q?"})
    assert r.status_code == 500 and r.json() == {"detail": "Internal error (RuntimeError)"}
    events = [line["event"] for line in lines(buf)]
    assert events == ["ask_unhandled", "ask"] and "exception" in lines(buf)[0]
    assert logging.getLogger(request_log.LOGGER_NAME).propagate is False
