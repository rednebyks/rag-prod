import subprocess
import sys
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from app import cost_tracker, observability, semantic_cache
from app import streaming as streaming_module
from app.auth import get_api_key
from app.rate_limiter import check_limit, get_redis
from app.security import validate_input
from app.streaming import chat_stream_generator
from app.vector_store import get_client as get_qdrant_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await cost_tracker.init_db()
    await semantic_cache.ensure_collection_exists()
    observability.init_langfuse()
    logger.info("RAG API started")
    yield
    logger.info("RAG API shutting down")


app = FastAPI(title="RAG Production API", version="1.0.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str


@app.post("/chat/stream")
async def chat_stream(
    request: Request,
    body: ChatRequest,
    api_key_info: dict = Depends(get_api_key),
):
    validate_input(body.message, api_key_info["key"])

    allowed, retry_after = await check_limit(
        api_key_info["key"], api_key_info["token_limit_per_min"]
    )
    if not allowed:
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": str(retry_after)},
            content={"detail": "Rate limit exceeded", "retry_after": retry_after},
        )

    return StreamingResponse(
        chat_stream_generator(request, body.message, api_key_info),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/usage/today")
async def usage_today(api_key_info: dict = Depends(get_api_key)):
    return await cost_tracker.get_today(api_key_info["key"])


@app.get("/usage/breakdown")
async def usage_breakdown(api_key_info: dict = Depends(get_api_key)):
    return await cost_tracker.get_breakdown(api_key_info["key"])


@app.get("/health")
async def health():
    qdrant_status = "ok"
    redis_status = "ok"

    try:
        client = get_qdrant_client()
        await client.get_collections()
    except Exception as e:
        qdrant_status = f"error: {e}"

    try:
        r = get_redis()
        await r.ping()
    except Exception as e:
        redis_status = f"error: {e}"

    return {
        "status": "ok",
        "active_streams": streaming_module.active_streams,
        "aborted_streams": streaming_module.aborted_streams,
        "qdrant": qdrant_status,
        "redis": redis_status,
    }


@app.post("/index/rebuild")
async def index_rebuild(api_key_info: dict = Depends(get_api_key)):
    if api_key_info["tier"] != "enterprise":
        raise HTTPException(status_code=403, detail="Enterprise tier required")

    try:
        result = subprocess.run(
            [sys.executable, "scripts/index.py"],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Indexing timed out")

    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"Indexing failed: {result.stderr[:500]}",
        )

    return {"status": "ok", "output": result.stdout.strip()}
