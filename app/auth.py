from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from app.config import API_KEYS

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key(api_key: str = Security(_api_key_header)) -> dict:
    if not api_key:
        raise HTTPException(status_code=401, detail="X-API-Key header required")
    tier_info = API_KEYS.get(api_key)
    if not tier_info:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return {"key": api_key, **tier_info}
