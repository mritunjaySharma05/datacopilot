"""FastAPI backend: /health, /datasets, /ask.

Run locally:
    uvicorn datacopilot.api.main:app --reload --app-dir src
"""
import time
from functools import lru_cache

import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from datacopilot import request_log, service
from datacopilot.pipeline.llm import LLMError, LLMRateLimitError, describe, get_provider

app = FastAPI(title="DataCopilot", version="0.5.0")
logger = request_log.configure()


class AskRequest(BaseModel):
    dataset: str = Field(..., examples=["crop"])
    question: str = Field(..., min_length=1, max_length=service.MAX_QUESTION_CHARS,
                          examples=["How many distinct states are in the data?"])


@lru_cache(maxsize=1)
def get_store() -> service.DatasetStore:
    """One dataset store per process (datasets open lazily on first question)."""
    return service.DatasetStore()


@app.get("/health")
def health() -> dict:
    """Liveness plus configuration summary; never calls the LLM and never shows keys."""
    try:
        llm = describe(get_provider())
    except LLMError as err:
        llm = {"error": str(err)}
    return {"status": "ok", "llm": llm,
            "datasets": [d["name"] for d in get_store().list() if d["available"]]}


@app.get("/datasets")
def datasets() -> list[dict]:
    return get_store().list()


@app.post("/ask")
def ask(req: AskRequest):
    """Answer a question about one dataset (status answered / refused / failed); logs one JSON line."""
    start = time.perf_counter()
    result, status, error, headers = None, 200, None, None
    try:
        result = service.ask(get_store(), req.dataset, req.question)
    except service.UnknownDatasetError:
        status, error = 404, f"Unknown dataset {req.dataset!r}"
    except service.DatasetUnavailableError as err:
        status, error = 503, str(err)
    except service.QuestionError as err:
        status, error = 422, str(err)
    except LLMRateLimitError as err:
        status, error = 429, str(err)
        headers = {"Retry-After": f"{err.retry_after_s:.0f}"} if err.retry_after_s is not None else None
    except LLMError as err:
        status, error = 503, str(err)
    except (requests.ConnectionError, requests.Timeout):
        status, error = 503, "The LLM service is unreachable (check OLLAMA_HOST or the network)."
    except requests.HTTPError as err:
        status, error = 502, f"The LLM service returned an error: {err}"
    except Exception as err:  # log unexpected failures too, then return a plain 500
        status, error = 500, f"Internal error ({type(err).__name__})"
        logger.exception("ask_unhandled", extra={"dataset": req.dataset, "question": req.question})
    request_log.log_ask(logger, request_log.ask_record(
        result, dataset=req.dataset, question=req.question, http_status=status, error=error,
        latency_ms=round(1000 * (time.perf_counter() - start), 1)))
    if status != 200:
        return JSONResponse({"detail": request_log.redact(error)}, status_code=status, headers=headers)
    return result
