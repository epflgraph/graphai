from graphai.celery.common.tasks import text_dummy_task, video_dummy_task


def test_text_dummy_task_accepts_request_id():
    assert text_dummy_task("results", request_id="abc") == "results"


def test_video_dummy_task_accepts_request_id():
    assert video_dummy_task("results", request_id="abc") == "results"


def test_video_dummy_task_returns_input():
    assert video_dummy_task({"key": "value"}) == {"key": "value"}
