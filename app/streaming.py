import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import Request

import tiktoken

from app import cost_tracker, observability, rate_limiter, semantic_cache
from app.config import MAX_CONCURRENT_LLM_CALLS
from app.embedder import embed
from app.llm_client import stream_with_fallback
from app.pricing import calculate_cost
from app.security import check_output
from app.vector_store import search as vector_search

_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))

logger = logging.getLogger(__name__)

# Module-level counters (safe in single-process asyncio)
active_streams: int = 0
aborted_streams: int = 0

_semaphore: asyncio.Semaphore | None = None


def get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)
    return _semaphore


def _build_messages(user_query: str, chunks: list[dict]) -> list[dict]:
    context = "\n\n".join(
        f"[Source {i + 1} | id={c['chunk_id']}]\n{c['text']}"
        for i, c in enumerate(chunks)
    )
    system = (
        "You are a helpful assistant that answers questions based strictly on the "
        "provided document context. If the answer is not contained in the context, "
        "say so clearly. Do not make up information."
    )
    user_content = (
        f"<context>\n{context}\n</context>\n\n"
        f"<user_query>{user_query}</user_query>"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


async def chat_stream_generator(
    request: Request,
    message: str,
    api_key_info: dict,
) -> AsyncIterator[str]:
    global active_streams, aborted_streams

    request_id = str(uuid.uuid4())
    api_key = api_key_info["key"]
    tier = api_key_info["tier"]
    models: list[str] = api_key_info["models"]
    tier_limit: int = api_key_info["token_limit_per_min"]

    start_time = time.time()
    trace = observability.start_trace(request_id, api_key, tier, message)

    # ── Embed query (single call, reused for cache + RAG) ─────────────────────
    embed_span = trace.span(name="embed_query")
    embedding = embed(message)
    embed_span.end()

    # ── Semantic cache check ──────────────────────────────────────────────────
    cache_span = trace.span(name="cache_check")
    cached = await semantic_cache.get(embedding)
    cache_span.end(metadata={"hit": cached is not None})

    if cached:
        words = cached["response"].split(" ")
        for word in words:
            if await request.is_disconnected():
                aborted_streams += 1
                return
            yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"
            await asyncio.sleep(0.005)

        latency_ms = int((time.time() - start_time) * 1000)

        await cost_tracker.log_request(
            request_id=request_id,
            api_key=api_key,
            model=cached.get("model", "cache"),
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
            ttft_ms=0,
            cache_hit=True,
            fallback_used=False,
        )

        done_event = {
            "type": "done",
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "cost_usd": 0.0,
            "cache_hit": True,
            "sources": [],
            "latency_ms": latency_ms,
        }
        yield f"data: {json.dumps(done_event)}\n\n"
        observability.flush()
        return

    # ── Vector search ─────────────────────────────────────────────────────────
    search_span = trace.span(name="vector_search")
    chunks = await vector_search(embedding)
    sources = [c["chunk_id"] for c in chunks]
    search_span.end(metadata={"sources": sources})

    messages = _build_messages(message, chunks)

    # ── Reserve input tokens before LLM call (atomic pre-deduction) ──────────
    estimated_input = _count_tokens(message) + sum(_count_tokens(c["text"]) for c in chunks)
    allowed, retry_after = await rate_limiter.reserve(api_key, estimated_input, tier_limit)
    if not allowed:
        yield f"data: {json.dumps({'type': 'error', 'code': 429, 'retry_after': retry_after})}\n\n"
        return

    # ── LLM call under semaphore ──────────────────────────────────────────────
    sem = get_semaphore()
    async with sem:
        if await request.is_disconnected():
            aborted_streams += 1
            await rate_limiter.consume(api_key, -estimated_input)  # refund reservation
            return

        active_streams += 1
        result_meta: dict = {}
        full_response: list[str] = []
        ttft_ms = 0
        first_token = True

        llm_span = trace.span(name="llm_call")

        try:
            async for token in stream_with_fallback(messages, models, result_meta):
                if await request.is_disconnected():
                    aborted_streams += 1
                    active_streams -= 1
                    llm_span.end()
                    return

                if first_token:
                    ttft_ms = int((time.time() - start_time) * 1000)
                    first_token = False

                full_response.append(token)
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        except asyncio.CancelledError:
            aborted_streams += 1
            active_streams -= 1
            llm_span.end()
            return

        except Exception as e:
            active_streams -= 1
            llm_span.end()
            logger.error("LLM error request_id=%s: %s", request_id, e)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return

        active_streams -= 1
        llm_span.end(
            metadata={
                "model": result_meta.get("model"),
                "fallback_used": result_meta.get("fallback_used"),
            }
        )

    # ── Post-stream processing ────────────────────────────────────────────────
    model_used: str = result_meta.get("model", models[0])
    fallback_used: bool = result_meta.get("fallback_used", False)
    latency_ms = int((time.time() - start_time) * 1000)
    accumulated = "".join(full_response)

    # Use reported usage if available; otherwise count locally from actual text
    reported = result_meta.get("usage") or {}
    input_tokens = reported.get("input_tokens") or _count_tokens(
        " ".join(m["content"] for m in messages)
    )
    output_tokens = reported.get("output_tokens") or _count_tokens(accumulated)
    usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}

    output_filtered = check_output(accumulated, request_id)

    await semantic_cache.set(embedding, message, accumulated, model_used)

    await cost_tracker.log_request(
        request_id=request_id,
        api_key=api_key,
        model=model_used,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        ttft_ms=ttft_ms,
        cache_hit=False,
        fallback_used=fallback_used,
        output_filtered=output_filtered,
    )

    # Input was already reserved; add output tokens and reconcile any estimate delta
    await rate_limiter.consume(api_key, output_tokens + (input_tokens - estimated_input))

    cost_usd = calculate_cost(model_used, usage["input_tokens"], usage["output_tokens"])

    done_event = {
        "type": "done",
        "usage": usage,
        "cost_usd": cost_usd,
        "cache_hit": False,
        "sources": sources,
        "latency_ms": latency_ms,
        "model": model_used,
        "fallback_used": fallback_used,
    }
    yield f"data: {json.dumps(done_event)}\n\n"

    trace.update(
        output=accumulated,
        metadata={
            "model": model_used,
            "fallback_used": fallback_used,
            "cache_hit": False,
            "cost_usd": cost_usd,
        },
    )
    observability.flush()
