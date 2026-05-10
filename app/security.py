import re
import logging
from fastapi import HTTPException
from app.config import MAX_INPUT_LENGTH

_req_logger = logging.getLogger("suspicious_requests")
_req_handler = logging.FileHandler("suspicious_requests.log")
_req_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
_req_logger.addHandler(_req_handler)
_req_logger.setLevel(logging.WARNING)

_PATTERNS = [
    r"ignore\s+(?:previous|all|prior)\s+instructions?",
    r"\bsystem\s*:",
    r"<\|im_start\|>",
    r"</s>",
    r"disregard\s+(?:your|all|previous)\s+instructions?",
    r"you\s+are\s+now\s+(?:a|an|the)\s+\w",
    r"act\s+as\s+if\s+you",
    r"new\s+instructions?\s*:",
    r"override\s+(?:your|the)\s+(?:previous\s+)?instructions?",
    r"forget\s+(?:everything|all\s+previous|your\s+previous)",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _PATTERNS]

_OUT_LOGGER = logging.getLogger("suspicious_responses")
_out_handler = logging.FileHandler("suspicious_responses.log")
_out_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
_OUT_LOGGER.addHandler(_out_handler)
_OUT_LOGGER.setLevel(logging.WARNING)

# Fragments that would indicate system prompt leakage in the output
_SYSTEM_FRAGMENTS = [
    "you are a helpful assistant that answers questions based strictly",
    "do not make up information",
    "<context>",
    "<user_query>",
]


def validate_input(message: str, api_key: str = "") -> None:
    if len(message) > MAX_INPUT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Input exceeds maximum length of {MAX_INPUT_LENGTH} characters",
        )
    for pattern in _COMPILED:
        if pattern.search(message):
            _req_logger.warning(
                "INJECTION_ATTEMPT api_key=%s pattern=%r message=%r",
                api_key,
                pattern.pattern,
                message[:300],
            )
            raise HTTPException(
                status_code=400,
                detail="Suspicious input detected: potential prompt injection",
            )


def check_output(response: str, request_id: str = "") -> bool:
    lower = response.lower()
    for fragment in _SYSTEM_FRAGMENTS:
        if fragment in lower:
            _OUT_LOGGER.warning(
                "OUTPUT_FILTERED request_id=%s fragment=%r response=%r",
                request_id,
                fragment,
                response[:300],
            )
            return True
    return False
