import logging
import sys
from collections.abc import Callable, Coroutine
from contextvars import ContextVar
from functools import wraps
from typing import Any, ParamSpec, TypeVar

# Context variables populated per-request by FastAPI [request_id] middleware and [get_current_user dependency]
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")
user_id_ctx: ContextVar[str] = ContextVar("user_id", default="-")


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get()
        record.user_id = user_id_ctx.get()
        return True


def setup_logging(log_level: str = "INFO") -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level.upper())

    # Prevent duplicate handlers if reloaded by Uvicorn
    if root_logger.handlers:
        root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | req=%(request_id)s | user=%(user_id)s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)
    handler.addFilter(ContextFilter())
    root_logger.addHandler(handler)

    # Quiet third-party libraries
    logging.getLogger("chromadb").setLevel(
        logging.WARNING
    )  # we are using chromadb for vector store


P = ParamSpec("P")
R = TypeVar("R")


def traceable(
    logger: logging.Logger,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            logger.info(f"Running : {func.__name__}")
            try:
                return func(*args, **kwargs)
            except Exception:
                logger.exception(f"Error in : {func.__name__}")
                raise

        return wrapper

    return decorator


def atraceable(
    logger: logging.Logger,
) -> Callable[
    [Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]
]:
    def decorator(
        func: Callable[P, Coroutine[Any, Any, R]],
    ) -> Callable[P, Coroutine[Any, Any, R]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            logger.info(f"Running : {func.__name__}")
            try:
                return await func(*args, **kwargs)
            except Exception:
                logger.exception(f"Error in : {func.__name__}")
                raise

        return wrapper

    return decorator
