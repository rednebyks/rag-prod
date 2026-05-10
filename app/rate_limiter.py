import time
import logging
import redis.asyncio as aioredis
from app.config import REDIS_URL

logger = logging.getLogger(__name__)

_redis: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    return _redis


def _window_key(api_key: str) -> str:
    window = int(time.time() // 60)
    return f"rl:{api_key}:{window}"


async def check_limit(api_key: str, tier_limit: int) -> tuple[bool, int]:
    """Read-only check — use reserve() for actual enforcement."""
    r = get_redis()
    key = _window_key(api_key)
    try:
        current = await r.get(key)
        used = int(current) if current else 0
        if used >= tier_limit:
            seconds_elapsed = int(time.time() % 60)
            return False, 60 - seconds_elapsed
        return True, 0
    except Exception as e:
        logger.warning("Redis check_limit failed (fail-open): %s", e)
        return True, 0


async def reserve(api_key: str, tokens: int, tier_limit: int) -> tuple[bool, int]:
    """
    Atomically pre-deduct tokens before the LLM call.
    If the bucket would overflow, rolls back and returns (False, retry_after).
    This prevents concurrent requests from all passing a read-only check.
    """
    r = get_redis()
    key = _window_key(api_key)
    try:
        new_total = await r.incrby(key, tokens)
        await r.expire(key, 120)
        if new_total > tier_limit:
            await r.decrby(key, tokens)  # roll back
            seconds_elapsed = int(time.time() % 60)
            return False, 60 - seconds_elapsed
        return True, 0
    except Exception as e:
        logger.warning("Redis reserve failed (fail-open): %s", e)
        return True, 0


async def consume(api_key: str, extra_tokens: int) -> None:
    """Add remaining tokens after LLM call (output tokens not known until after)."""
    if extra_tokens <= 0:
        return
    r = get_redis()
    key = _window_key(api_key)
    try:
        pipe = r.pipeline()
        pipe.incrby(key, extra_tokens)
        pipe.expire(key, 120)
        await pipe.execute()
    except Exception as e:
        logger.warning("Redis consume failed: %s", e)
