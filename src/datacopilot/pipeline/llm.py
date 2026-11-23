"""LLM providers: send one prompt, get the reply text back.

The provider is chosen with LLM_PROVIDER in the environment (.env):
  ollama (default)  local Ollama, the Module 4 model (OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_NUM_CTX)
  groq              Groq's hosted, OpenAI-compatible API (GROQ_API_KEY, GROQ_MODEL)

Every provider gets the same prompt and runs at temperature 0; SQL extraction, validation
and the retry stay in sql_generator / runner, so they are identical for all providers.
API keys are read from the environment only and never appear in repr() or describe().
"""
import os
from dataclasses import dataclass, field
from typing import Mapping, Protocol

import requests

REQUEST_TIMEOUT_S = 120

OLLAMA_DEFAULT_HOST = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL = "qwen2.5-coder:7b"
OLLAMA_DEFAULT_NUM_CTX = 8192  # Ollama's default context is small; wide schemas would be truncated

GROQ_DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"

PROVIDERS = ("ollama", "groq")


class LLMError(Exception):
    """The LLM provider is misconfigured or refused the request."""


class LLMRateLimitError(LLMError):
    """The provider's rate limit was hit (HTTP 429)."""

    def __init__(self, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class LLMProvider(Protocol):
    """Anything that turns a prompt into reply text."""

    name: str
    model: str

    def complete(self, prompt: str) -> str: ...


def describe(provider: LLMProvider) -> dict[str, str]:
    """Provider name and model, safe to log (no keys or hosts)."""
    return {"provider": provider.name, "model": provider.model}


@dataclass
class OllamaProvider:
    """Local Ollama server (/api/generate)."""

    host: str = OLLAMA_DEFAULT_HOST
    model: str = OLLAMA_DEFAULT_MODEL
    num_ctx: int = OLLAMA_DEFAULT_NUM_CTX
    timeout_s: float = REQUEST_TIMEOUT_S
    name: str = field(default="ollama", init=False)

    def complete(self, prompt: str) -> str:
        response = requests.post(
            f"{self.host.rstrip('/')}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0, "num_ctx": self.num_ctx}},
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return response.json()["response"]


@dataclass
class GroqProvider:
    """Groq's OpenAI-compatible chat completions API; the prompt is sent as one user message."""

    api_key: str = field(repr=False)
    model: str
    base_url: str = GROQ_DEFAULT_BASE_URL
    timeout_s: float = REQUEST_TIMEOUT_S
    name: str = field(default="groq", init=False)

    def complete(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "temperature": 0,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=self.timeout_s,
        )
        if response.status_code == 429:
            retry_after = _float_or_none(response.headers.get("retry-after"))
            wait = f" Try again in {retry_after:.0f} s." if retry_after is not None else ""
            raise LLMRateLimitError(f"Groq rate limit reached (HTTP 429).{wait}", retry_after)
        if response.status_code in (401, 403):
            raise LLMError(f"Groq rejected the request (HTTP {response.status_code}): check GROQ_API_KEY.")
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


def _float_or_none(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def get_provider(env: Mapping[str, str] | None = None) -> LLMProvider:
    """Build the provider named by LLM_PROVIDER (default ollama) from environment settings."""
    env = os.environ if env is None else env
    name = (env.get("LLM_PROVIDER") or "ollama").strip().lower()
    if name == "ollama":
        return OllamaProvider(host=env.get("OLLAMA_HOST") or OLLAMA_DEFAULT_HOST,
                              model=env.get("OLLAMA_MODEL") or OLLAMA_DEFAULT_MODEL,
                              num_ctx=int(env.get("OLLAMA_NUM_CTX") or OLLAMA_DEFAULT_NUM_CTX))
    if name == "groq":
        key, model = (env.get("GROQ_API_KEY") or "").strip(), (env.get("GROQ_MODEL") or "").strip()
        if not key:
            raise LLMError("LLM_PROVIDER=groq needs GROQ_API_KEY in the environment (.env).")
        if not model:
            raise LLMError("LLM_PROVIDER=groq needs GROQ_MODEL (a model id from Groq's model list).")
        return GroqProvider(api_key=key, model=model,
                            base_url=env.get("GROQ_BASE_URL") or GROQ_DEFAULT_BASE_URL)
    raise LLMError(f"Unknown LLM_PROVIDER {name!r}; use one of: {', '.join(PROVIDERS)}.")
