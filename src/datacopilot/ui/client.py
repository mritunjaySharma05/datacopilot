"""HTTP client the Streamlit UI uses to call the DataCopilot API (the UI never imports the pipeline)."""
import os

import requests

DEFAULT_API_URL = "http://localhost:8000"
TIMEOUT_S = 180  # the API may retry the LLM once; local Ollama can take a while


class APIError(Exception):
    """The API could not be reached or answered with an error status."""


def api_url() -> str:
    return (os.getenv("API_URL") or DEFAULT_API_URL).rstrip("/")


def _get(path: str, base: str | None = None, timeout: float = 10):
    try:
        r = requests.get(f"{base or api_url()}{path}", timeout=timeout)
    except requests.RequestException as err:
        raise APIError(f"Cannot reach the API at {base or api_url()} ({type(err).__name__}).") from err
    if r.status_code >= 400:
        raise APIError(_detail(r))
    return r.json()


def _detail(r: requests.Response) -> str:
    """Readable error text from an API error response."""
    try:
        detail = r.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, list):  # FastAPI validation errors
        detail = "; ".join(str(d.get("msg", d)) for d in detail)
    return f"HTTP {r.status_code}: {detail or r.reason}"


def health(base: str | None = None) -> dict:
    return _get("/health", base)


def datasets(base: str | None = None) -> list[str]:
    """Names of the datasets that are available on the server."""
    return [d["name"] for d in _get("/datasets", base) if d.get("available")]


def ask(dataset: str, question: str, base: str | None = None, timeout: float = TIMEOUT_S) -> dict:
    """POST /ask; returns the result dict (status answered / refused / failed) or raises APIError."""
    try:
        r = requests.post(f"{base or api_url()}/ask", json={"dataset": dataset, "question": question},
                          timeout=timeout)
    except requests.RequestException as err:
        raise APIError(f"Cannot reach the API at {base or api_url()} ({type(err).__name__}).") from err
    if r.status_code >= 400:
        raise APIError(_detail(r))
    return r.json()
