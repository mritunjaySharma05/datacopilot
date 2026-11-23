"""Tests for the Streamlit UI and its API client (HTTP mocked; no server, no LLM)."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests
from streamlit.testing.v1 import AppTest

from datacopilot.ui import client

APP = str(Path(__file__).resolve().parents[1] / "src" / "datacopilot" / "ui" / "app.py")
ANSWER = {"request_id": "abc123", "status": "answered", "sql": "SELECT 2 AS n", "columns": ["n"], "rows": [[2]],
          "row_count": 1, "truncated": False, "error": None,
          "gate": {"answerable": True, "p_answerable": 0.9, "threshold": 0.5},
          "timings_ms": {"llm": 1200.0, "total": 1500.0}}


def response(status=200, body=None, reason="OK"):
    r = MagicMock(status_code=status, reason=reason)
    r.json.return_value = body
    return r


# ---------- client ----------

def test_api_url_from_env(monkeypatch):
    monkeypatch.setenv("API_URL", "http://api:8000/")
    assert client.api_url() == "http://api:8000"
    monkeypatch.delenv("API_URL")
    assert client.api_url() == "http://localhost:8000"


def test_datasets_keeps_available_only(monkeypatch):
    monkeypatch.setattr(client.requests, "get", lambda url, timeout: response(
        body=[{"name": "crop", "available": True}, {"name": "ipl", "available": False}]))
    assert client.datasets("http://x") == ["crop"]


def test_ask_posts_dataset_and_question(monkeypatch):
    post = MagicMock(return_value=response(body=ANSWER))
    monkeypatch.setattr(client.requests, "post", post)
    assert client.ask("crop", "q?", "http://x") == ANSWER
    assert post.call_args.args[0] == "http://x/ask"
    assert post.call_args.kwargs["json"] == {"dataset": "crop", "question": "q?"}


@pytest.mark.parametrize("resp, message", [
    (response(429, {"detail": "Groq rate limit reached (HTTP 429). Try again in 7 s."}, "Too Many Requests"),
     "HTTP 429: Groq rate limit reached"),
    (response(422, {"detail": [{"msg": "String should have at least 1 character"}]}), "at least 1 character"),
])
def test_ask_error_details(monkeypatch, resp, message):
    monkeypatch.setattr(client.requests, "post", lambda *a, **k: resp)
    with pytest.raises(client.APIError, match=message):
        client.ask("crop", "q?", "http://x")


def test_unreachable_api(monkeypatch):
    def down(*a, **k):
        raise requests.ConnectionError("refused")
    monkeypatch.setattr(client.requests, "get", down)
    with pytest.raises(client.APIError, match="Cannot reach the API"):
        client.health("http://x")


# ---------- Streamlit app ----------

@pytest.fixture
def fake_api(monkeypatch):
    calls = []
    monkeypatch.setattr(client, "health", lambda base=None: {"status": "ok",
                                                             "llm": {"provider": "fake", "model": "m"}})
    monkeypatch.setattr(client, "datasets", lambda base=None: ["crop", "ipl"])

    def ask(dataset, question, base=None, timeout=None):
        calls.append((dataset, question))
        return ANSWER
    monkeypatch.setattr(client, "ask", ask)
    return calls


def test_app_renders_and_answers(fake_api):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.selectbox[0].options == ["crop", "ipl"]
    at.text_area[0].input("How many states?").run()
    at.button[0].click().run()
    assert fake_api == [("crop", "How many states?")]
    assert at.code[0].value == "SELECT 2 AS n"
    assert at.dataframe[0].value.iloc[0, 0] == 2


def test_app_shows_api_error(monkeypatch):
    def down(base=None):
        raise client.APIError("Cannot reach the API at http://localhost:8000 (ConnectionError).")
    monkeypatch.setattr(client, "health", down)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert "Cannot reach the API" in at.error[0].value
