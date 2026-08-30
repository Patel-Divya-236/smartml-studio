"""Unit tests for LLMClient.

No network access: requests.post is replaced throughout. The point of these tests is
that every failure path returns None rather than raising, because callers rely on
falling back to static explanations instead of surfacing an error to the user.
"""

import pytest
import requests

from src.llm.client import LLMClient


class FakeResponse:
    """Minimal stand-in for a requests.Response."""

    def __init__(self, status_code: int, payload=None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or str(payload)

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def _ok_payload(text: str = "A narration."):
    return {"choices": [{"message": {"content": text}}]}


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """Keep ambient environment variables out of these tests."""
    for key in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(key, raising=False)


def test_unavailable_without_api_key():
    """No key means unavailable, and calls short-circuit before any request."""
    client = LLMClient()
    assert client.is_available is False
    assert client.complete(system="s", user="u") is None
    assert "LLM_API_KEY" in client.last_error


def test_available_with_api_key():
    """A configured key makes the client available."""
    assert LLMClient(api_key="test-key").is_available is True


def test_successful_completion_returns_text(monkeypatch):
    """A well-formed 200 response yields the assistant text."""
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(200, _ok_payload()))
    client = LLMClient(api_key="test-key")
    assert client.complete(system="s", user="u") == "A narration."
    assert client.last_error is None


def test_request_carries_model_and_messages(monkeypatch):
    """The outgoing request contains the configured model and both message roles."""
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return FakeResponse(200, _ok_payload())

    monkeypatch.setattr(requests, "post", fake_post)
    LLMClient(api_key="k", base_url="https://example.test/v1", model="some-model").complete(
        system="SYSTEM", user="USER"
    )

    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["json"]["model"] == "some-model"
    assert [m["role"] for m in captured["json"]["messages"]] == ["system", "user"]
    assert captured["headers"]["Authorization"] == "Bearer k"


def test_rate_limit_returns_none_with_explanation(monkeypatch):
    """429 is handled as a fallback condition, not an exception."""
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(429, text="slow down"))
    client = LLMClient(api_key="k")
    assert client.complete(system="s", user="u") is None
    assert "rate limit" in client.last_error.lower()


def test_unknown_model_returns_none_with_actionable_error(monkeypatch):
    """404 tells the user the model id is stale rather than failing opaquely."""
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(404, text="no such model"))
    client = LLMClient(api_key="k", model="retired-model")
    assert client.complete(system="s", user="u") is None
    assert "retired-model" in client.last_error


def test_server_error_returns_none(monkeypatch):
    """Any other non-200 is a fallback condition."""
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(500, text="boom"))
    client = LLMClient(api_key="k")
    assert client.complete(system="s", user="u") is None
    assert "500" in client.last_error


def test_malformed_response_returns_none(monkeypatch):
    """A 200 with an unexpected body shape must not raise."""
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(200, {"unexpected": True}))
    client = LLMClient(api_key="k")
    assert client.complete(system="s", user="u") is None
    assert client.last_error is not None


def test_empty_content_returns_none(monkeypatch):
    """Whitespace-only output is treated as no answer."""
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(200, _ok_payload("   ")))
    assert LLMClient(api_key="k").complete(system="s", user="u") is None


def test_timeout_returns_none(monkeypatch):
    """A timeout falls back rather than propagating."""
    def raise_timeout(*a, **k):
        raise requests.Timeout()

    monkeypatch.setattr(requests, "post", raise_timeout)
    client = LLMClient(api_key="k")
    assert client.complete(system="s", user="u") is None
    assert "timed out" in client.last_error.lower()


def test_connection_error_returns_none(monkeypatch):
    """An unreachable provider falls back rather than propagating."""
    def raise_conn(*a, **k):
        raise requests.ConnectionError("no route")

    monkeypatch.setattr(requests, "post", raise_conn)
    client = LLMClient(api_key="k")
    assert client.complete(system="s", user="u") is None
    assert "could not reach" in client.last_error.lower()


def test_environment_configuration_is_read(monkeypatch):
    """Configuration resolves from the environment when not passed explicitly."""
    monkeypatch.setenv("LLM_API_KEY", "env-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://env.test/v1/")
    monkeypatch.setenv("LLM_MODEL", "env-model")
    client = LLMClient()
    assert client.api_key == "env-key"
    assert client.base_url == "https://env.test/v1"  # trailing slash stripped
    assert client.model == "env-model"


def test_list_models_returns_sorted_ids(monkeypatch):
    """The model picker gets a sorted list of currently served ids."""
    monkeypatch.setattr(
        requests,
        "get",
        lambda *a, **k: FakeResponse(200, {"data": [{"id": "zeta"}, {"id": "alpha"}]}),
    )
    assert LLMClient(api_key="k").list_models() == ["alpha", "zeta"]
