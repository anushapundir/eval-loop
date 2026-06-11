"""Model abstraction — the ONLY module that calls a model provider.

Every completion in the system goes through :func:`generate` (CLAUDE.md §6).
Two backends:

* **Ollama** (``qwen2.5:7b``) — the free, local default for the high-volume
  agent. Called over HTTP via httpx.
* **Anthropic Claude Haiku** — used sparingly for the sampled LLM judge, and
  optionally as the agent in all-Haiku mode. The system/rubric prompt is sent
  with ``cache_control`` so repeated calls pay ~10% of input cost.

All calls retry with backoff so one transient failure never crashes a batch.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import httpx

from config.logging import get_logger
from config.settings import Settings, get_settings

log = get_logger(__name__)

Provider = Literal["ollama", "haiku"]


@dataclass(frozen=True)
class Completion:
    """The result of a model call."""

    text: str
    provider: str
    model: str
    latency_ms: float


class LLMError(RuntimeError):
    """Raised when a model call fails after exhausting retries."""


def _retry(fn: Callable[[], Completion], *, attempts: int, what: str) -> Completion:
    """Call ``fn`` with exponential backoff; raise :class:`LLMError` on failure."""
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - we re-raise as LLMError below
            last_exc = exc
            wait = 2.0**i
            log.warning("%s failed (attempt %d/%d): %s - retrying in %.1fs",
                        what, i + 1, attempts, exc, wait)
            time.sleep(wait)
    raise LLMError(f"{what} failed after {attempts} attempts") from last_exc


def _ollama_chat(
    system: str | None, prompt: str, settings: Settings
) -> Completion:
    """One chat completion from the local Ollama server."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body = {
        "model": settings.ollama_model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": settings.temperature,
            "num_predict": settings.max_tokens,
        },
    }

    def _call() -> Completion:
        start = time.perf_counter()
        with httpx.Client(timeout=settings.request_timeout_s) as client:
            resp = client.post(
                f"{settings.ollama_base_url}/api/chat", json=body
            )
            resp.raise_for_status()
        latency_ms = (time.perf_counter() - start) * 1000.0
        text = resp.json()["message"]["content"].strip()
        return Completion(
            text=text,
            provider="ollama",
            model=settings.ollama_model,
            latency_ms=latency_ms,
        )

    return _retry(_call, attempts=settings.max_retries, what="ollama.chat")


def _haiku_chat(
    system: str | None, prompt: str, settings: Settings, *, cache_system: bool
) -> Completion:
    """One chat completion from Anthropic Claude Haiku.

    When ``cache_system`` is set, the system block carries ``cache_control`` so
    the rubric/system prompt is cached across the judging batch.
    """
    # Imported lazily so the package works without the SDK when only Ollama is used.
    from anthropic import Anthropic

    if not settings.anthropic_api_key:
        raise LLMError("ANTHROPIC_API_KEY is not set; cannot call Haiku.")

    client = Anthropic(api_key=settings.anthropic_api_key)
    system_blocks = None
    if system:
        block: dict = {"type": "text", "text": system}
        if cache_system:
            block["cache_control"] = {"type": "ephemeral"}
        system_blocks = [block]

    def _call() -> Completion:
        start = time.perf_counter()
        kwargs: dict = {
            "model": settings.haiku_model,
            "max_tokens": settings.max_tokens,
            "temperature": settings.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_blocks is not None:
            kwargs["system"] = system_blocks
        msg = client.messages.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000.0
        text = "".join(
            block.text for block in msg.content if block.type == "text"
        ).strip()
        return Completion(
            text=text,
            provider="haiku",
            model=settings.haiku_model,
            latency_ms=latency_ms,
        )

    return _retry(_call, attempts=settings.max_retries, what="haiku.messages")


def generate(
    prompt: str,
    *,
    system: str | None = None,
    provider: Provider | None = None,
    cache_system: bool = False,
    settings: Settings | None = None,
) -> Completion:
    """Generate a completion from the configured provider.

    Args:
        prompt: The user prompt.
        system: Optional system instruction.
        provider: Force a backend; defaults to ``settings.model_provider``.
            The judge always passes ``provider="haiku"`` explicitly.
        cache_system: Cache the system block (Haiku only; used by the judge).
        settings: Injectable settings (defaults to the cached singleton).

    Returns:
        A :class:`Completion` with text, provider, model, and latency.
    """
    settings = settings or get_settings()
    chosen: Provider = provider or settings.model_provider
    if chosen == "ollama":
        return _ollama_chat(system, prompt, settings)
    return _haiku_chat(system, prompt, settings, cache_system=cache_system)
