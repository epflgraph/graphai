"""
Shared structured logging configuration for GraphAI.

Uses structlog to emit clean, emoji-friendly logs. The output format is controlled
by environment variables:

    GRAPHAI_LOG_FORMAT   - "console" (default, human-readable colours) or "json"
    GRAPHAI_LOG_LEVEL    - "DEBUG", "INFO" (default), "WARNING", "ERROR", "CRITICAL"

For Grafana/Loki set GRAPHAI_LOG_FORMAT=json; the system journal / Promtail already
provides the timestamp, so the JSON payload does not duplicate it.
"""

import logging
import os
import sys
from typing import Optional

import structlog


_DEFAULT_LOG_FORMAT = "console"
_DEFAULT_LOG_LEVEL = "INFO"


def _log_level_from_env() -> int:
    level_name = os.getenv("GRAPHAI_LOG_LEVEL", _DEFAULT_LOG_LEVEL).upper()
    return getattr(logging, level_name, logging.INFO)


def _renderer_from_env():
    fmt = os.getenv("GRAPHAI_LOG_FORMAT", _DEFAULT_LOG_FORMAT).lower()
    if fmt == "json":
        return structlog.processors.JSONRenderer()
    # Force colors even when stdout is not a TTY (e.g. systemd journal) so that
    # journalctl shows them, while Promtail's decolorize stage keeps Grafana clean.
    return structlog.dev.ConsoleRenderer(colors=True, force_colors=True)


def _shared_processors():
    """Processors shared between structlog and stdlib ProcessorFormatter."""
    return [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]


def _final_processor():
    """Return the processor that hands the event dict off to stdlib formatting.

    Because we route structlog through stdlib logging (LoggerFactory), the
    structlog processor chain must end with ``wrap_for_formatter``.  The actual
    console/JSON rendering is then done by ``structlog.stdlib.ProcessorFormatter``
    attached to the stdlib handlers.  This avoids double-rendering.
    """
    return structlog.stdlib.ProcessorFormatter.wrap_for_formatter


def configure_structlog(
    level: Optional[int] = None,
    renderer=None,
    force: bool = False,
) -> None:
    """
    Configure structlog for GraphAI.

    Args:
        level: Logging level. Defaults to GRAPHAI_LOG_LEVEL env var or INFO.
        renderer: Rendering processor used by stdlib ProcessorFormatter. Defaults
            to console (colours) or JSON based on GRAPHAI_LOG_FORMAT env var.
        force: If True, reconfigure even if already configured.
    """
    if not force and getattr(configure_structlog, "_configured", False):
        return

    if level is None:
        level = _log_level_from_env()

    structlog.configure(
        processors=_shared_processors() + [_final_processor()],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )
    configure_structlog._configured = True


def configure_stdlib_logging(
    level: Optional[int] = None,
    renderer=None,
    force: bool = False,
) -> None:
    """Configure stdlib logging to route through structlog's renderer.

    This removes Celery's default bracketed format and gives all libraries the same
    clean output as our application loggers. It is safe to call multiple times.
    """
    if not force and getattr(configure_stdlib_logging, "_configured", False):
        return

    configure_structlog(level=level, renderer=renderer, force=force)

    if level is None:
        level = _log_level_from_env()
    if renderer is None:
        renderer = _renderer_from_env()

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=_shared_processors(),
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Replace existing handlers on the root logger with our formatted stream handler.
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # Propagate our formatter to any already-instantiated loggers (e.g. uvicorn,
    # celery) so they do not keep their own bracketed formatters.
    for logger_name in list(logging.root.manager.loggerDict.keys()):
        logger = logging.getLogger(logger_name)
        if isinstance(logger, logging.PlaceHolder):
            continue
        for handler in list(logger.handlers):
            handler.setFormatter(formatter)

    configure_stdlib_logging._configured = True


def get_logger(name: Optional[str] = None):
    """Return a structlog logger, configuring structlog first if needed."""
    configure_structlog()
    return structlog.get_logger(name)
