"""Structured request logging: one JSON line per /ask request on stdout.

Each line has the question, dataset, gate decision, SQL, outcome, row count, LLM provider/model
and latency per stage. Docker keeps stdout; in AWS, CloudWatch Logs collects it and the JSON
fields are searchable (Logs Insights).

No secrets: values of environment variables whose names look secret (…_KEY, …_TOKEN,
…_SECRET, …PASSWORD) and API-key / bearer-token patterns are replaced by "[REDACTED]" in every
string field before a line is written, as a second line of defence (nothing logs keys on purpose).
"""
import datetime as dt
import json
import logging
import os
import re
import sys
from typing import IO

LOGGER_NAME = "datacopilot.requests"
MAX_FIELD_CHARS = 2000
REDACTED = "[REDACTED]"
_SECRET_ENV = re.compile(r"(_KEY|_TOKEN|_SECRET|PASSWORD)$", re.IGNORECASE)
_SECRET_PATTERNS = re.compile(r"gsk_[A-Za-z0-9]{8,}|sk-[A-Za-z0-9_-]{16,}|Bearer\s+[A-Za-z0-9._~+/=-]{8,}")
_RESERVED = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


def secret_values(env: dict | None = None) -> list[str]:
    """Values of secret-looking environment variables (longest first, so overlaps redact fully)."""
    env = os.environ if env is None else env
    return sorted({v for k, v in env.items() if _SECRET_ENV.search(k) and v and len(v) >= 6}, key=len, reverse=True)


def redact(value, secrets: list[str] | None = None):
    """Copy of `value` (str / dict / list / scalar) with secrets replaced and long strings cut."""
    if isinstance(value, str):
        for s in secret_values() if secrets is None else secrets:
            value = value.replace(s, REDACTED)
        value = _SECRET_PATTERNS.sub(REDACTED, value)
        return value if len(value) <= MAX_FIELD_CHARS else value[:MAX_FIELD_CHARS] + "...[truncated]"
    if isinstance(value, dict):
        return {k: redact(v, secrets) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v, secrets) for v in value]
    return value


class JSONFormatter(logging.Formatter):
    """One JSON object per line: ts, level, logger, event, then the record's extra fields."""

    def format(self, record: logging.LogRecord) -> str:
        fields = {k: v for k, v in vars(record).items() if k not in _RESERVED}
        line = {"ts": dt.datetime.fromtimestamp(record.created, dt.timezone.utc).isoformat(timespec="milliseconds"),
                "level": record.levelname, "logger": record.name, "event": record.getMessage(), **fields}
        if record.exc_info:
            line["exception"] = self.formatException(record.exc_info)
        return json.dumps(redact(line, secret_values()), default=str, ensure_ascii=False)


def configure(stream: IO[str] | None = None, level: str | None = None) -> logging.Logger:
    """Send the request logger to `stream` (default stdout) as JSON lines; safe to call twice."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    logger.setLevel((level or os.getenv("LOG_LEVEL") or "INFO").upper())
    logger.propagate = False
    return logger


def ask_record(result: dict | None, *, dataset: str, question: str, http_status: int,
               error: str | None = None, latency_ms: float | None = None) -> dict:
    """Fields for one /ask log line, from a service.ask result or from an error before it returned."""
    r = result or {}
    gate = r.get("gate") or {}
    llm = r.get("llm") or {}
    return {
        "request_id": r.get("request_id"), "http_status": http_status,
        "dataset": dataset, "question": question,
        "status": r.get("status", "error"),
        "gate_answerable": gate.get("answerable"), "p_answerable": gate.get("p_answerable"),
        "sql": r.get("sql"), "row_count": r.get("row_count"),
        "error": error if error is not None else r.get("error"),
        "llm_provider": llm.get("provider"), "llm_model": llm.get("model"),
        "timings_ms": r.get("timings_ms"),
        "latency_ms": latency_ms if latency_ms is not None else (r.get("timings_ms") or {}).get("total"),
    }


def log_ask(logger: logging.Logger, record: dict) -> None:
    """Write one /ask line: INFO for answered / refused / failed, WARNING for HTTP errors."""
    level = logging.WARNING if record["http_status"] >= 400 else logging.INFO
    logger.log(level, "ask", extra=record)
