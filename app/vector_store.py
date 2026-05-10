from qdrant_client import AsyncQdrantClient
from app.config import QDRANT_URL, QDRANT_API_KEY, CHUNKS_COLLECTION, TOP_K

_client: AsyncQdrantClient | None = None


def get_client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        _client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _client


async def search(embedding: list[float], top_k: int = TOP_K) -> list[dict]:
    client = get_client()
    results = await client.query_points(
        collection_name=CHUNKS_COLLECTION,
        query=embedding,
        limit=top_k,
        with_payload=True,
    )
    return [
        {
            "chunk_id": r.payload.get("chunk_id", f"chunk_{r.id}"),
            "text": r.payload.get("text", ""),
            "score": r.score,
        }
        for r in results.points
    ]
