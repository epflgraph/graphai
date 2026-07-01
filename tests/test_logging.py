import json
import logging
import re
import uuid

import pytest
import structlog

from graphai.core.common.logging import configure_stdlib_logging


def _strip_ansi(text: str) -> str:
    return re.sub(r'\x1b\[[0-9;]*m', '', text)


class TestLoggingConfig:
    """Verify GraphAI's shared logging configuration renders clean, concise output."""

    @pytest.fixture(autouse=True)
    def _isolate_root_logger(self):
        """Save and restore root logger handlers so tests don't leak formatters."""
        root = logging.getLogger()
        saved_handlers = list(root.handlers)
        saved_level = root.level
        yield
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)

    def _fresh_logger(self):
        # structlog caches loggers by name; use a unique name per call to avoid
        # stale processor caches across force reconfigurations.
        return structlog.get_logger(f'test_logger_{uuid.uuid4().hex}')

    def test_console_mode_renders_event_and_fields(self, monkeypatch, capsys):
        monkeypatch.setenv('GRAPHAI_LOG_FORMAT', 'console')
        monkeypatch.setenv('GRAPHAI_LOG_LEVEL', 'INFO')
        configure_stdlib_logging(force=True)

        logger = self._fresh_logger()
        logger.info('🚀 hello', key='value')

        output = _strip_ansi(capsys.readouterr().out)
        assert '🚀 hello' in output
        assert 'key=value' in output

    def test_json_mode_drops_duplicate_and_noisy_fields(self, monkeypatch, capsys):
        monkeypatch.setenv('GRAPHAI_LOG_FORMAT', 'json')
        monkeypatch.setenv('GRAPHAI_LOG_LEVEL', 'INFO')
        configure_stdlib_logging(force=True)

        logger = self._fresh_logger()
        logger.info('✅ event', count=42)

        output = capsys.readouterr().out.strip()
        record = json.loads(output.splitlines()[-1])

        # Grafana / Loki already timestamp the log line; duplicating it inside the
        # JSON payload is noisy.
        assert 'timestamp' not in record
        # Logger name and numeric level are redundant for operational dashboards.
        assert 'logger' not in record
        assert 'level_number' not in record

        assert record['event'] == '✅ event'
        assert record['count'] == 42
        assert record['level'] == 'info'

    def test_stdlib_logging_uses_structlog_formatter(self, monkeypatch, capsys):
        monkeypatch.setenv('GRAPHAI_LOG_FORMAT', 'console')
        monkeypatch.setenv('GRAPHAI_LOG_LEVEL', 'INFO')
        configure_stdlib_logging(force=True)

        stdlib_logger = logging.getLogger('foreign_library')
        stdlib_logger.info('plain stdlib message')

        output = _strip_ansi(capsys.readouterr().out)
        assert 'plain stdlib message' in output
