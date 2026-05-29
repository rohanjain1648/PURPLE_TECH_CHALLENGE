"""
Structured logging configuration using structlog.
Every request emits: trace_id, store_id, endpoint, latency_ms, event_count, status_code.
"""
import logging
import sys
import uuid
from contextvars import ContextVar

import structlog

# Context variable shared across the request lifecycle
_request_context: ContextVar[dict] = ContextVar("request_context", default={})


def get_context() -> dict:
    return _request_context.get()


def set_context(**kwargs) -> None:
    ctx = dict(_request_context.get())
    ctx.update(kwargs)
    _request_context.set(ctx)


def new_trace_id() -> str:
    return str(uuid.uuid4())[:8]


def configure_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer() if not debug else structlog.dev.ConsoleRenderer(),
        ],
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
