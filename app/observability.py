import logging

logger = logging.getLogger(__name__)

_langfuse = None
_enabled = False


def init_langfuse() -> None:
    global _langfuse, _enabled
    try:
        from langfuse import Langfuse
        from app.config import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL

        if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
            logger.info("Langfuse keys not set — observability disabled")
            return

        _langfuse = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_BASE_URL,
        )
        _enabled = True
        logger.info("Langfuse observability enabled")
    except ImportError:
        logger.warning("langfuse package not installed")
    except Exception as e:
        logger.warning("Langfuse init failed: %s", e)


class _NoOpObj:
    def span(self, *a, **kw):
        return _NoOpSpan()

    def generation(self, *a, **kw):
        return _NoOpSpan()

    def update(self, **kw):
        pass


class _NoOpSpan:
    def end(self, **kw):
        pass

    def update(self, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


def start_trace(request_id: str, api_key: str, tier: str, user_query: str):
    if not _enabled or not _langfuse:
        return _NoOpObj()
    try:
        return _langfuse.trace(
            id=request_id,
            name="rag_request",
            metadata={"api_key": api_key, "tier": tier},
            input=user_query,
        )
    except Exception as e:
        logger.warning("Langfuse trace creation failed: %s", e)
        return _NoOpObj()


def flush() -> None:
    if _enabled and _langfuse:
        try:
            _langfuse.flush()
        except Exception as e:
            logger.warning("Langfuse flush failed: %s", e)
