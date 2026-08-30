"""Provider-agnostic LLM client.

Talks to any OpenAI-compatible ``/chat/completions`` endpoint — Groq, OpenRouter,
Together, Hugging Face, a local Ollama server — so the provider is a configuration
choice rather than a code dependency. Plain ``requests`` is used instead of a vendor
SDK for three reasons: it is already installed, exactly what leaves the machine is
visible in one place, and this module stays free of vendor lock-in.

Configuration comes from environment variables only, so the package has no
dependency on any particular front end or web framework.

Environment variables (all optional except the key):
    LLM_API_KEY    provider API key — without it the client reports unavailable
    LLM_BASE_URL   overrides SETTINGS.LLM_BASE_URL
    LLM_MODEL      overrides SETTINGS.LLM_MODEL
"""

import logging
import os
from typing import Any

import requests

from config.settings import SETTINGS

logger = logging.getLogger(__name__)


class LLMClient:
    """Thin client over an OpenAI-compatible chat-completions endpoint.

    Every method returns ``None`` instead of raising when the LLM is unusable —
    missing key, network failure, rate limit, bad response. Callers fall back to
    their static behaviour rather than surfacing a stack trace to the user.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        """Resolve configuration from arguments, then environment, then SETTINGS."""
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or SETTINGS.LLM_BASE_URL).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL") or SETTINGS.LLM_MODEL
        self.timeout = timeout if timeout is not None else SETTINGS.LLM_TIMEOUT_SECONDS

        self._last_error: str | None = None
        logger.info(
            "LLMClient initialised (base_url=%s, model=%s, key=%s).",
            self.base_url,
            self.model,
            "set" if self.api_key else "MISSING",
        )

    # ── Status ────────────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """True when a key is configured. Does not perform a network call."""
        return bool(self.api_key)

    @property
    def last_error(self) -> str | None:
        """Human-readable description of the most recent failure, if any."""
        return self._last_error

    # ── Requests ──────────────────────────────────────────────────────

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str | None:
        """Send a single-turn completion and return the text, or None on any failure.

        Args:
            system: System prompt establishing the task and grounding rules.
            user: The payload to narrate.
            max_tokens: Output cap; defaults to SETTINGS.LLM_MAX_TOKENS.
            temperature: Sampling temperature; defaults to SETTINGS.LLM_TEMPERATURE.

        Returns:
            The assistant's text, or None if the LLM was unavailable or failed.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.chat(messages, max_tokens=max_tokens, temperature=temperature)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str | None:
        """Send a multi-turn conversation and return the reply text, or None on failure."""
        if not self.is_available:
            self._last_error = "No LLM_API_KEY configured."
            logger.info("LLM call skipped: no API key configured.")
            return None

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or SETTINGS.LLM_MAX_TOKENS,
            "temperature": temperature if temperature is not None else SETTINGS.LLM_TEMPERATURE,
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
        except requests.Timeout:
            self._last_error = f"LLM request timed out after {self.timeout:.0f}s."
            logger.warning(self._last_error)
            return None
        except requests.RequestException as exc:
            self._last_error = f"Could not reach the LLM provider: {exc}"
            logger.warning("LLM request failed: %s", exc)
            return None

        return self._parse(response)

    def list_models(self) -> list[str] | None:
        """Return model ids the provider currently serves, or None on failure.

        Model ids change without notice, so this is how the UI offers a live picker
        instead of failing on a stale value in SETTINGS.
        """
        if not self.is_available:
            return None
        try:
            response = requests.get(
                f"{self.base_url}/models", headers=self._headers(), timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json().get("data", [])
            return sorted(str(m.get("id")) for m in data if m.get("id"))
        except (requests.RequestException, ValueError) as exc:
            self._last_error = f"Could not list models: {exc}"
            logger.warning("Model listing failed: %s", exc)
            return None

    # ── Internals ─────────────────────────────────────────────────────

    def _headers(self) -> dict[str, str]:
        """Build request headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _parse(self, response: Any) -> str | None:
        """Extract assistant text from a provider response, or None on any problem."""
        if response.status_code == 429:
            self._last_error = "LLM rate limit reached. Falling back to standard explanations."
            logger.warning("LLM rate limited (429).")
            return None

        if response.status_code == 404:
            self._last_error = (
                f"Model '{self.model}' was not found on this provider. "
                "Model ids change — pick a current one from the model list."
            )
            logger.warning(self._last_error)
            return None

        if response.status_code != 200:
            snippet = str(response.text)[:200]
            self._last_error = f"LLM provider returned HTTP {response.status_code}: {snippet}"
            logger.warning(self._last_error)
            return None

        try:
            choices = response.json()["choices"]
            text = choices[0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            self._last_error = f"Unexpected response shape from the LLM provider: {exc}"
            logger.warning(self._last_error)
            return None

        if not text or not str(text).strip():
            self._last_error = "LLM returned an empty response."
            logger.warning(self._last_error)
            return None

        self._last_error = None
        return str(text).strip()
