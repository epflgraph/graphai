import logging

import pytest

from graphai.celery.common.celery_config import SuppressTaskSuccessFilter


class TestSuppressTaskSuccessFilter:
    """Unit tests for the Celery task-success log filter.

    These tests verify that the filter drops Celery's built-in task-success
    messages (which dump task return values such as DataFrames) while keeping
    task failures, retries and other warnings.
    """

    @pytest.fixture
    def filter_instance(self):
        return SuppressTaskSuccessFilter()

    def _make_record(self, msg):
        record = logging.LogRecord(
            name='celery.app.trace',
            level=logging.INFO,
            pathname='',
            lineno=0,
            msg=msg,
            args=(),
            exc_info=None,
        )
        return record

    @pytest.mark.parametrize(
        'msg',
        [
            'Task text.compute_scores[64d0a3e8-a61a-4cf3-850f-242f87ba01fa] succeeded in 1.77s: [10 rows x 13 columns]',
            'Task text.wikisearch[99fec644-f8c3-4d32-8301-2a8c2dd4fad6] succeeded in 1.70s: Empty DataFrame',
            'Task foo.bar succeeded in 0.01s: hello',
        ],
    )
    def test_filter_drops_task_success(self, filter_instance, msg):
        record = self._make_record(msg)
        assert filter_instance.filter(record) is False

    @pytest.mark.parametrize(
        'msg',
        [
            'Task text.compute_scores[64d0a3e8-a61a-4cf3-850f-242f87ba01fa] raised unexpected: ValueError(\"boom\")',
            'Task text.compute_scores[64d0a3e8-a61a-4cf3-850f-242f87ba01fa] retry: Retry in 10s',
            'Task text.compute_scores[64d0a3e8-a61a-4cf3-850f-242f87ba01fa] received',
            'Something unrelated succeeded',
            'plain log message',
        ],
    )
    def test_filter_keeps_failures_and_other_messages(self, filter_instance, msg):
        record = self._make_record(msg)
        assert filter_instance.filter(record) is True
