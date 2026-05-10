import hashlib
import time
import logging
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.config import (
    QDRANT_URL, QDRANT_API_KEY, CACHE_COLLECTION,
    CACHE_SIMILARITY_THRESHOLD, CACHE_TTL_SECONDS, EMBEDDING_DIM,
)

logger = logging.getLogger(__name__)

_client: AsyncQdrantClient | None = None


def get_client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        _client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _client


def _point_id(query: str) -> int:
    # Deterministic 63-bit integer from query text
    digest = hashlib.sha256(query.encode()).hexdigest()
    return int(digest[:15], 16)


async def ensure_collection_exists() -> None:
    client = get_client()
    try:
        await client.get_collection(CACHE_COLLECTION)
    except Exception:
        await client.create_collection(
            collection_name=CACHE_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        logger.info("Created cache collection: %s", CACHE_COLLECTION)


async def get(embedding: list[float]) -> dict | None:
    client = get_client()
    now = time.time()

    try:
        results = await client.query_points(
            collection_name=CACHE_COLLECTION,
            query=embedding,
            limit=1,
            with_payload=True,
            score_threshold=CACHE_SIMILARITY_THRESHOLD,
        )
    except Exception as e:
        logger.warning("Cache lookup failed: %s", e)
        return None

    if not results.points:
        return None

    hit = results.points[0]
    payload = hit.payload or {}

    expire_at = payload.get("expire_at", 0)
    if expire_at and now > expire_at:
        return None

    return {
        "response": payload.get("response", ""),
        "model": payload.get("model", ""),
        "score": hit.score,
    }


async def set(embedding: list[float], query: str, response: str, model: str) -> None:
    client = get_client()
    try:
        await client.upsert(
            collection_name=CACHE_COLLECTION,
            points=[
                PointStruct(
                    id=_point_id(query),
                    vector=embedding,
                    payload={
                        "query": query,
                        "response": response,
                        "model": model,
                        "expire_at": time.time() + CACHE_TTL_SECONDS,
                        "created_at": time.time(),
                    },
                )
            ],
        )
    except Exception as e:
        logger.warning("Cache write failed: %s", e)
