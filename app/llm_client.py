import asyncio
import logging
import time
from collections.abc import AsyncIterator
from openai import AsyncOpenAI, APIStatusError, APIConnectionError
from app.config import (
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLM_TIMEOUT_SECONDS,
    CIRCUIT_BREAKER_THRESHOLD, CIRCUIT_BREAKER_WINDOW_SECONDS,
    CIRCUIT_BREAKER_OPEN_DURATION_SECONDS,
)

logger = logging.getLogger(__name__)

_RETRYABLE_CODES = {429, 500, 502, 503, 504}
_NON_RETRYABLE_CODES = {400, 401, 403, 422}

_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )
    return _client


class CircuitBreaker:
    def __init__(self) -> None:
        self._state: dict[str, dict] = {}

    def record_error(self, model: str) -> None:
        now = time.time()
        if model not in self._state:
            self._state[model] = {"errors": 0, "window_start": now, "open_until": 0.0}
        s = self._state[model]
        if now - s["window_start"] > CIRCUIT_BREAKER_WINDOW_SECONDS:
            s["errors"] = 0
            s["window_start"] = now
        s["errors"] += 1
        if s["errors"] >= CIRCUIT_BREAKER_THRESHOLD:
            s["open_until"] = now + CIRCUIT_BREAKER_OPEN_DURATION_SECONDS
            logger.warning("Circuit breaker OPEN for model=%s", model)

    def record_success(self, model: str) -> None:
        if model in self._state:
            self._state[model]["errors"] = 0

    def is_open(self, model: str) -> bool:
        if model not in self._state:
            return False
        return time.time() < self._state[model]["open_until"]


circuit_breaker = CircuitBreaker()


async def stream_with_fallback(
    messages: list[dict],
    models: list[str],
    result_meta: dict,
) -> AsyncIterator[str]:
    """
    Async generator that yields token strings.
    Populates `result_meta` with: model, fallback_used, usage (dict).
    """
    client = get_client()
    fallback_used = False
    last_error: Exception | None = None

    for i, model in enumerate(models):
        if i == 0 and circuit_breaker.is_open(model):
            logger.info("Circuit open for %s — skipping to fallback", model)
            fallback_used = True
            continue
        if i > 0:
            fallback_used = True

        try:
            stream = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    max_tokens=2048,
                ),
                timeout=LLM_TIMEOUT_SECONDS,
            )

            usage_data: dict | None = None
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_data = {
                        "input_tokens": chunk.usage.prompt_tokens or 0,
                        "output_tokens": chunk.usage.completion_tokens or 0,
                    }

            circuit_breaker.record_success(model)
            result_meta["model"] = model
            result_meta["fallback_used"] = fallback_used
            result_meta["usage"] = usage_data or {"input_tokens": 0, "output_tokens": 0}
            return

        except asyncio.TimeoutError as e:
            logger.warning("Timeout on model=%s", model)
            circuit_breaker.record_error(model)
            last_error = e

        except APIStatusError as e:
            # 400 from provider (e.g. max_tokens exceeded) is retryable — try next model
            # Only hard-fail on auth/validation errors we sent ourselves
            if e.status_code in _NON_RETRYABLE_CODES and "provider" not in str(e).lower():
                raise
            logger.warning("APIStatusError status=%s model=%s: %s", e.status_code, model, e.message)
            circuit_breaker.record_error(model)
            last_error = e

        except APIConnectionError as e:
            logger.warning("Connection error model=%s: %s", model, e)
            circuit_breaker.record_error(model)
            last_error = e

    raise RuntimeError(f"All models exhausted. Last error: {last_error}")
