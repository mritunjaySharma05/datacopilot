"""FastAPI backend: /health, /datasets, /ask.

Run locally:
    uvicorn datacopilot.api.main:app --reload --app-dir src
"""
from functools import lru_cache

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from datacopilot import service
from datacopilot.pipeline.llm import LLMError, LLMRateLimitError, describe, get_provider

app = FastAPI(title="DataCopilot", version="0.5.0")


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
    """Answer a question about one dataset (status answered / refused / failed)."""
    try:
        return service.ask(get_store(), req.dataset, req.question)
    except service.UnknownDatasetError:
        raise HTTPException(404, f"Unknown dataset {req.dataset!r}")
    except service.DatasetUnavailableError as err:
        raise HTTPException(503, str(err))
    except service.QuestionError as err:
        raise HTTPException(422, str(err))
    except LLMRateLimitError as err:
        headers = {"Retry-After": f"{err.retry_after_s:.0f}"} if err.retry_after_s is not None else None
        return JSONResponse({"detail": str(err)}, status_code=429, headers=headers)
    except LLMError as err:
        raise HTTPException(503, str(err))
    except (requests.ConnectionError, requests.Timeout):
        raise HTTPException(503, "The LLM service is unreachable (check OLLAMA_HOST or the network).")
    except requests.HTTPError as err:
        raise HTTPException(502, f"The LLM service returned an error: {err}")
