import os
from dotenv import load_dotenv

load_dotenv()

API_KEYS: dict[str, dict] = {
    "demo-free": {
        "tier": "free",
        "token_limit_per_min": 5000,
        "models": [
            "meta-llama/llama-3.2-3b-instruct:free",
            "openrouter/owl-alpha",
            "inclusionai/ring-2.6-1t:free",
        ],
    },
    "demo-pro": {
        "tier": "pro",
        "token_limit_per_min": 20000,
        "models": [
            "mistralai/mistral-7b-instruct",
            "openai/gpt-4o-mini",
            "meta-llama/llama-3.1-8b-instruct:free",
        ],
    },
    "demo-enterprise": {
        "tier": "enterprise",
        "token_limit_per_min": 100000,
        "models": [
            "openai/gpt-4o",
            "anthropic/claude-3.5-sonnet",
            "mistralai/mistral-large",
        ],
    },
}

OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY")
CHUNKS_COLLECTION = "chunks_collection"
CACHE_COLLECTION = "cache_collection"
EMBEDDING_DIM = 384
CACHE_SIMILARITY_THRESHOLD = 0.92
CACHE_TTL_SECONDS = 3600
TOP_K = 3

REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")

LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_BASE_URL: str = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

DATABASE_PATH: str = os.getenv("DATABASE_PATH", "cost_tracking.db")

LLM_TIMEOUT_SECONDS = 15
MAX_CONCURRENT_LLM_CALLS = 20
MAX_INPUT_LENGTH = 4000

CIRCUIT_BREAKER_THRESHOLD = 5
CIRCUIT_BREAKER_WINDOW_SECONDS = 60
CIRCUIT_BREAKER_OPEN_DURATION_SECONDS = 60
