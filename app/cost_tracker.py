import aiosqlite
from datetime import datetime, timezone
from app.config import DATABASE_PATH
from app.pricing import calculate_cost

DB_PATH = DATABASE_PATH

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS requests (
    request_id    TEXT PRIMARY KEY,
    api_key       TEXT NOT NULL,
    model         TEXT NOT NULL,
    input_tokens  INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cost_usd      REAL    DEFAULT 0.0,
    latency_ms    INTEGER DEFAULT 0,
    ttft_ms       INTEGER DEFAULT 0,
    cache_hit     INTEGER DEFAULT 0,
    fallback_used INTEGER DEFAULT 0,
    output_filtered INTEGER DEFAULT 0,
    created_at    TEXT DEFAULT CURRENT_TIMESTAMP
)
"""


async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(_CREATE_TABLE)
        await db.commit()


async def log_request(
    *,
    request_id: str,
    api_key: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
    ttft_ms: int,
    cache_hit: bool,
    fallback_used: bool,
    output_filtered: bool = False,
) -> None:
    cost_usd = calculate_cost(model, input_tokens, output_tokens)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO requests
              (request_id, api_key, model, input_tokens, output_tokens,
               cost_usd, latency_ms, ttft_ms, cache_hit, fallback_used, output_filtered)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                request_id, api_key, model,
                input_tokens, output_tokens, cost_usd,
                latency_ms, ttft_ms,
                int(cache_hit), int(fallback_used), int(output_filtered),
            ),
        )
        await db.commit()


async def get_today(api_key: str) -> dict:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            """
            SELECT COUNT(*),
                   COALESCE(SUM(input_tokens + output_tokens), 0),
                   COALESCE(SUM(cost_usd), 0)
            FROM requests
            WHERE api_key = ? AND created_at LIKE ?
            """,
            (api_key, f"{today}%"),
        ) as cur:
            row = await cur.fetchone()
    return {
        "date": today,
        "requests": row[0],
        "tokens": row[1],
        "cost_usd": round(row[2], 6),
    }


async def get_breakdown(api_key: str) -> dict:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            """
            SELECT model,
                   COUNT(*) as requests,
                   COALESCE(SUM(input_tokens + output_tokens), 0) as tokens,
                   COALESCE(SUM(cost_usd), 0) as cost_usd,
                   AVG(CASE WHEN cache_hit = 0 THEN latency_ms END) as avg_latency,
                   AVG(CASE WHEN cache_hit = 0 THEN ttft_ms END) as avg_ttft
            FROM requests
            WHERE api_key = ?
            GROUP BY model
            """,
            (api_key,),
        ) as cur:
            model_rows = await cur.fetchall()

        async with db.execute(
            """
            SELECT COUNT(*) as total,
                   COALESCE(SUM(cache_hit), 0)     as cache_hits,
                   COALESCE(SUM(fallback_used), 0) as fallbacks,
                   AVG(CASE WHEN cache_hit = 0 THEN latency_ms END) as avg_latency_ms,
                   MAX(CASE WHEN cache_hit = 0 THEN latency_ms END) as max_latency_ms
            FROM requests
            WHERE api_key = ?
              AND created_at >= strftime('%Y-%m-%d %H:%M:%S', datetime('now', '-1 hour'))
            """,
            (api_key,),
        ) as cur:
            stats = await cur.fetchone()

    total = stats[0] or 1
    by_model = {
        row[0]: {
            "requests": row[1],
            "tokens": row[2],
            "cost_usd": round(row[3], 6),
            "avg_latency_ms": round(row[4] or 0),
            "avg_ttft_ms": round(row[5] or 0),
        }
        for row in model_rows
    }

    return {
        "by_model": by_model,
        "cache_hit_rate": round((stats[1] or 0) / total, 3),
        "fallback_rate": round((stats[2] or 0) / total, 3),
        "avg_latency_ms": round(stats[3] or 0),
        "p95_latency_ms": round((stats[4] or 0) * 0.95),
    }
