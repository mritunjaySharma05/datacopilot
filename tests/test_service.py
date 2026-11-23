"""Tests for datacopilot.service and the FastAPI app (fake LLM provider and fake gate; no network)."""
import numpy as np
import pytest
import requests
from fastapi.testclient import TestClient

from datacopilot import service
from datacopilot.api import main as api
from datacopilot.pipeline.llm import LLMError, LLMRateLimitError

CROP_CSV = """state_name,crop_year,season,crop,area,production
Kerala,2001,Kharif,Rice,100,250
Kerala,2002,Rabi,Wheat,50,80
Punjab,2001,Rabi,Wheat,300,900
"""


class FakeProvider:
    name, model = "fake", "fake-model"

    def __init__(self, replies=("SELECT COUNT(DISTINCT state_name) AS n FROM crop_production",), error=None):
        self.replies, self.error, self.prompts = list(replies), error, []

    def complete(self, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


class WriteVerbGate:
    """Fake gate model: refuses questions with a write verb, accepts the rest."""

    def predict_proba(self, X):
        p = np.where(X["has_write_verb"].to_numpy() > 0, 0.1, 0.9)
        return np.column_stack([1 - p, p])


BUNDLE = {"model": WriteVerbGate(), "feature_names": ["has_write_verb"], "threshold": 0.5}


@pytest.fixture
def store(tmp_path):
    (tmp_path / "crop_production.csv").write_text(CROP_CSV)
    return service.DatasetStore(tmp_path)


def test_store_lists_available_datasets_and_opens_lazily(store):
    listed = {d["name"]: d["available"] for d in store.list()}
    assert listed == {"olist": False, "ipl": False, "f1": False, "crop": True}
    ds = store.get("crop")
    assert ds is store.get("crop") and "crop_production" in ds.schema and ds.dialect == "duckdb"
    with pytest.raises(service.UnknownDatasetError):
        store.get("nope")
    with pytest.raises(service.DatasetUnavailableError):
        store.get("ipl")


def test_ask_answers_with_sql_rows_and_timings(store):
    r = service.ask(store, "crop", "  How many states are there?  ", BUNDLE, FakeProvider())
    assert r["status"] == "answered" and r["question"] == "How many states are there?"
    assert r["sql"].startswith("SELECT COUNT(DISTINCT state_name)")
    assert (r["columns"], r["rows"], r["row_count"], r["truncated"]) == (["n"], [[2]], 1, False)
    assert r["gate"] == {"answerable": True, "p_answerable": 0.9, "threshold": 0.5}
    assert r["llm"] == {"provider": "fake", "model": "fake-model"}
    assert set(r["timings_ms"]) == {"open", "gate", "llm", "sql", "total"} and len(r["request_id"]) == 12


def test_ask_refused_by_gate_never_calls_llm(store):
    fake = FakeProvider()
    r = service.ask(store, "crop", "Delete all rows for Kerala", BUNDLE, fake)
    assert r["status"] == "refused" and r["sql"] is None and fake.prompts == []
    assert "not look answerable" in r["message"]


def test_ask_failed_after_retry_reports_sql_and_error(store):
    fake = FakeProvider(["SELECT rainfall FROM crop_production"])
    r = service.ask(store, "crop", "Average rainfall?", BUNDLE, fake)
    assert r["status"] == "failed" and "rainfall" in r["sql"] and "after one retry" in r["error"]
    assert len(fake.prompts) == 2  # first attempt + one retry, same as Module 4


def test_ask_never_runs_write_sql(store, tmp_path):
    r = service.ask(store, "crop", "How many rows?", BUNDLE, FakeProvider(["DROP TABLE crop_production"]))
    assert r["status"] == "failed"
    assert store.get("crop").conn.execute("SELECT COUNT(*) FROM crop_production").fetchone()[0] == 3


def test_ask_truncates_rows_sent_back(store, monkeypatch):
    monkeypatch.setattr(service, "MAX_RESPONSE_ROWS", 2)
    r = service.ask(store, "crop", "List all rows", BUNDLE, FakeProvider(["SELECT * FROM crop_production"]))
    assert r["row_count"] == 3 and len(r["rows"]) == 2 and r["truncated"]


@pytest.mark.parametrize("question", ["   ", "x" * 501])
def test_ask_rejects_empty_or_long_questions(store, question):
    with pytest.raises(service.QuestionError):
        service.ask(store, "crop", question, BUNDLE, FakeProvider())


# ---------- API ----------

@pytest.fixture
def client(store, monkeypatch):
    holder = {"provider": FakeProvider()}
    monkeypatch.setattr(api, "get_store", lambda: store)
    monkeypatch.setattr(api, "get_provider", lambda: holder["provider"])
    monkeypatch.setattr(service, "get_provider", lambda: holder["provider"])
    monkeypatch.setattr(service, "load_model", lambda path: BUNDLE)
    c = TestClient(api.app)
    c.holder = holder
    return c


def test_health_shows_provider_and_datasets_without_llm_call(client):
    r = client.get("/health").json()
    assert r == {"status": "ok", "llm": {"provider": "fake", "model": "fake-model"}, "datasets": ["crop"]}
    assert client.holder["provider"].prompts == []


def test_health_reports_llm_config_error(client, monkeypatch):
    def broken():
        raise LLMError("LLM_PROVIDER=groq needs GROQ_API_KEY in the environment (.env).")
    monkeypatch.setattr(api, "get_provider", broken)
    assert "GROQ_API_KEY" in client.get("/health").json()["llm"]["error"]


def test_datasets_endpoint(client):
    assert {"name": "crop", "dialect": "duckdb", "available": True} in client.get("/datasets").json()


def test_ask_endpoint_answers(client):
    r = client.post("/ask", json={"dataset": "crop", "question": "How many states?"})
    assert r.status_code == 200 and r.json()["rows"] == [[2]]


@pytest.mark.parametrize("body, status", [
    ({"dataset": "nope", "question": "q?"}, 404),
    ({"dataset": "ipl", "question": "q?"}, 503),
    ({"dataset": "crop", "question": ""}, 422),
    ({"dataset": "crop", "question": "   "}, 422),
    ({"question": "q?"}, 422),
])
def test_ask_endpoint_client_errors(client, body, status):
    assert client.post("/ask", json=body).status_code == status


@pytest.mark.parametrize("error, status", [
    (LLMRateLimitError("Groq rate limit reached (HTTP 429). Try again in 7 s.", 7.0), 429),
    (LLMError("Groq rejected the request (HTTP 401): check GROQ_API_KEY."), 503),
    (requests.ConnectionError("down"), 503),
    (requests.HTTPError("500 error"), 502),
])
def test_ask_endpoint_maps_llm_errors(client, error, status):
    client.holder["provider"] = FakeProvider(error=error)
    r = client.post("/ask", json={"dataset": "crop", "question": "How many states?"})
    assert r.status_code == status
    if status == 429:
        assert r.headers["retry-after"] == "7"
