"""Tests for datacopilot.pipeline.llm (HTTP is mocked; no network)."""
from unittest.mock import MagicMock

import pytest
import requests

from datacopilot.pipeline import llm
from datacopilot.pipeline.llm import GroqProvider, LLMError, LLMRateLimitError, OllamaProvider, describe, get_provider

SECRET = "gsk_test_secret_123"


def fake_post(monkeypatch, status=200, body=None, headers=None):
    response = MagicMock(status_code=status, headers=headers or {})
    response.json.return_value = body or {}
    response.raise_for_status.side_effect = (requests.HTTPError(f"{status} error") if status >= 400 else None)
    post = MagicMock(return_value=response)
    monkeypatch.setattr(llm.requests, "post", post)
    return post


# --- get_provider ---

def test_default_provider_is_ollama_with_module4_model():
    p = get_provider({})
    assert isinstance(p, OllamaProvider)
    assert (p.model, p.host, p.num_ctx) == ("qwen2.5-coder:7b", "http://localhost:11434", 8192)


def test_ollama_settings_from_env():
    p = get_provider({"LLM_PROVIDER": "Ollama", "OLLAMA_HOST": "http://host.docker.internal:11434",
                      "OLLAMA_MODEL": "m", "OLLAMA_NUM_CTX": "4096"})
    assert (p.host, p.model, p.num_ctx) == ("http://host.docker.internal:11434", "m", 4096)


def test_groq_provider_from_env():
    p = get_provider({"LLM_PROVIDER": "groq", "GROQ_API_KEY": SECRET, "GROQ_MODEL": "some-model"})
    assert isinstance(p, GroqProvider) and p.model == "some-model" and p.base_url == llm.GROQ_DEFAULT_BASE_URL


@pytest.mark.parametrize("env, message", [
    ({"LLM_PROVIDER": "groq", "GROQ_MODEL": "m"}, "GROQ_API_KEY"),
    ({"LLM_PROVIDER": "groq", "GROQ_API_KEY": SECRET}, "GROQ_MODEL"),
    ({"LLM_PROVIDER": "openai"}, "Unknown LLM_PROVIDER"),
])
def test_get_provider_errors(env, message):
    with pytest.raises(LLMError, match=message):
        get_provider(env)


def test_key_never_in_repr_or_describe():
    p = get_provider({"LLM_PROVIDER": "groq", "GROQ_API_KEY": SECRET, "GROQ_MODEL": "m"})
    assert SECRET not in repr(p) and SECRET not in str(describe(p))
    assert describe(p) == {"provider": "groq", "model": "m"}


# --- OllamaProvider ---

def test_ollama_posts_prompt_at_temperature_zero(monkeypatch):
    post = fake_post(monkeypatch, body={"response": "SELECT 1"})
    assert OllamaProvider(host="http://ollama.test:1234/", model="test-model", num_ctx=4096).complete("hi") == "SELECT 1"
    assert post.call_args.args[0] == "http://ollama.test:1234/api/generate"
    body = post.call_args.kwargs["json"]
    assert body == {"model": "test-model", "prompt": "hi", "stream": False,
                    "options": {"temperature": 0, "num_ctx": 4096}}


# --- GroqProvider ---

def test_groq_sends_same_prompt_as_one_user_message(monkeypatch):
    post = fake_post(monkeypatch, body={"choices": [{"message": {"content": "```sql\nSELECT 1\n```"}}]})
    out = GroqProvider(api_key=SECRET, model="m").complete("the prompt")
    assert out == "```sql\nSELECT 1\n```"
    assert post.call_args.args[0] == "https://api.groq.com/openai/v1/chat/completions"
    assert post.call_args.kwargs["json"] == {"model": "m", "temperature": 0,
                                             "messages": [{"role": "user", "content": "the prompt"}]}
    assert post.call_args.kwargs["headers"] == {"Authorization": f"Bearer {SECRET}"}


def test_groq_rate_limit_raises_with_retry_after(monkeypatch):
    fake_post(monkeypatch, status=429, headers={"retry-after": "7"})
    with pytest.raises(LLMRateLimitError, match="Try again in 7 s") as err:
        GroqProvider(api_key=SECRET, model="m").complete("p")
    assert err.value.retry_after_s == 7.0


def test_groq_bad_key_message_has_no_key(monkeypatch):
    fake_post(monkeypatch, status=401)
    with pytest.raises(LLMError) as err:
        GroqProvider(api_key=SECRET, model="m").complete("p")
    assert "GROQ_API_KEY" in str(err.value) and SECRET not in str(err.value)


def test_groq_other_http_errors_propagate(monkeypatch):
    fake_post(monkeypatch, status=500)
    with pytest.raises(requests.HTTPError):
        GroqProvider(api_key=SECRET, model="m").complete("p")
