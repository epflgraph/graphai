import json

import pytest
from unittest.mock import patch
from time import sleep

from graphai.api.celery_tasks.summarization import summarize_text_task, compute_summarization_text_fingerprint_task

################################################################
# /completion/summary                                       #
# /completion/title                                         #
################################################################


@patch('graphai.api.celery_tasks.summarization.summarize_text_task.run')
@pytest.mark.usefixtures('transcript_text')
def test__summarization_summary__summarize_text__mock_task(mock_run, transcript_text):
    # Mock calling the task
    summarize_text_task.run(transcript_text)

    # Assert that the task has been called
    assert summarize_text_task.run.call_count == 1


################################################################


@pytest.mark.usefixtures('transcript_text')
def test__summarization_summary__summarize_text__run_task(transcript_text):
    # Call the task
    token_and_text = {
        'existing_results': None,
        'token': 'mock_token_transcript',
        'text': transcript_text,
        'original_text': transcript_text
    }
    summary_transcript = summarize_text_task.run(token_and_text, 'lecture', 'summary')

    # Assert that the results are correct
    assert isinstance(summary_transcript, dict)
    assert 'summary' in summary_transcript
    assert summary_transcript['successful'] is True
    summary_text = summary_transcript['summary'].lower()
    assert 'lecture' in summary_text
    assert 'digital circuit' in summary_text
    assert 'simulation' in summary_text


#############################################################


@pytest.mark.usefixtures('ocr_text')
def test__summarization_title__summarize_text__run_task(ocr_text):
    # Call the task
    token_and_text = {
        'existing_results': None,
        'token': 'mock_token_ocr',
        'text': ocr_text,
        'original_text': ocr_text
    }
    title_ocr = summarize_text_task.run(token_and_text, 'lecture', 'title')

    # Assert that the results are correct
    assert isinstance(title_ocr, dict)
    assert 'summary' in title_ocr
    assert title_ocr['successful'] is True
    title_text = title_ocr['summary'].lower()
    assert 'simulation' in title_text
    assert 'digital' in title_text or 'discrete' in title_text or 'circuit' in title_text


################################################################


@pytest.mark.celery(accept_content=['pickle', 'json'], result_serializer='pickle', task_serializer='pickle')
@pytest.mark.usefixtures('transcript_text')
def test__summarization_summary__summarize_text__integration(fixture_app, celery_worker, transcript_text, timeout=30):
    # The celery_worker object is necessary for async tasks, otherwise the status will be permanently stuck on
    # PENDING.

    # First, we call the summary endpoint with force=True to test the full task pipeline working
    response = fixture_app.post('/completion/summary',
                                data=json.dumps({"text": transcript_text, "text_type": "lecture",
                                                 "force": True}),
                                timeout=timeout)
    # Check status code is successful
    assert response.status_code == 200
    # Parse resulting task id
    task_id = response.json()['task_id']

    # Waiting for task chain to succeed
    current_status = 'PENDING'
    n_tries = 0
    while current_status == 'PENDING' and n_tries < 10:
        # Wait a few seconds
        sleep(3)
        # Now get status
        response = fixture_app.get(f'/completion/summary/status/{task_id}',
                                   timeout=timeout)
        current_status = response.json()['task_status']
        n_tries += 1

    # Now get status
    response = fixture_app.get(f'/completion/summary/status/{task_id}',
                               timeout=timeout)
    # Parse result
    summary_results = response.json()
    # Check returned value
    assert isinstance(summary_results, dict)
    assert 'task_result' in summary_results
    assert summary_results['task_status'] == 'SUCCESS'
    assert summary_results['task_result']['successful'] is True
    assert summary_results['task_result']['fresh'] is True
    summary_text = summary_results['task_result']['summary'].lower()
    assert 'lecture' in summary_text
    assert 'digital circuit' in summary_text or 'discrete event' in summary_text
    assert 'simulation' in summary_text
    assert summary_results['task_result']['summary_type'] == 'summary'
    original_summary = summary_text

    ################################################

    # Now, we call the summary endpoint again with the same input to make sure the caching works correctly
    response = fixture_app.post('/completion/summary',
                                data=json.dumps({"text": transcript_text, "text_type": "lecture",
                                                 "force": False}),
                                timeout=timeout)
    # Check status code is successful
    assert response.status_code == 200
    # Parse resulting task id
    task_id = response.json()['task_id']

    # Waiting for task chain to succeed
    current_status = 'PENDING'
    n_tries = 0
    while current_status == 'PENDING' and n_tries < 10:
        # Wait a few seconds
        sleep(3)
        # Now get status
        response = fixture_app.get(f'/completion/summary/status/{task_id}',
                                   timeout=timeout)
        current_status = response.json()['task_status']
        n_tries += 1

    # Now get status
    response = fixture_app.get(f'/completion/summary/status/{task_id}',
                               timeout=timeout)
    # Parse result
    summary_results = response.json()
    # Check values
    assert isinstance(summary_results, dict)
    # results must be successful but not fresh, since they were already cached
    assert summary_results['task_status'] == 'SUCCESS'
    assert summary_results['task_result']['successful'] is True
    assert summary_results['task_result']['fresh'] is False
    assert summary_results['task_result']['summary'].lower() == original_summary


################################################################
# /completion/calculate_fingerprint                           #
################################################################


@pytest.mark.usefixtures('transcript_text')
def test__summarization_calculate_fingerprint__compute_text_fingerprint__run_task(transcript_text):
    # Call the task
    fp = compute_summarization_text_fingerprint_task.run('mock_token_transcript', transcript_text, True)

    # Assert that the results are correct
    assert isinstance(fp, dict)
    assert 'result' in fp
    assert fp['fresh']
    assert fp['fp_token'] == 'mock_token_transcript'
    assert fp['result'] == '020c000800021008020208000e0a120008000408080a30060208020c00000804'


################################################################


@pytest.mark.celery(accept_content=['pickle', 'json'], result_serializer='pickle', task_serializer='pickle')
@pytest.mark.usefixtures('transcript_text')
def test__summarization_calculate_fingerprint__compute_text_fingerprint__integration(
        fixture_app, celery_worker, transcript_text, timeout=30):
    # The celery_worker object is necessary for async tasks, otherwise the status will be permanently stuck on
    # PENDING.

    # Call the calculate_fingerprint endpoint with force=True
    response = fixture_app.post('/completion/calculate_fingerprint',
                                data=json.dumps({"text": transcript_text, "text_type": "lecture",
                                                 "force": True}),
                                timeout=timeout)
    # Check status code is successful
    assert response.status_code == 200
    # Parse resulting task id
    task_id = response.json()['task_id']

    # Wait a few seconds

    # Waiting for task chain to succeed
    current_status = 'PENDING'
    n_tries = 0
    while current_status == 'PENDING' and n_tries < 10:
        # Wait a few seconds
        sleep(3)
        # Now get status
        response = fixture_app.get(f'/completion/calculate_fingerprint/status/{task_id}',
                                   timeout=timeout)
        current_status = response.json()['task_status']
        n_tries += 1

    # Now get status
    response = fixture_app.get(f'/completion/calculate_fingerprint/status/{task_id}',
                               timeout=timeout)
    # Parse result
    content = response.json()
    # Check returned value
    assert isinstance(content, dict)
    assert 'task_result' in content
    assert content['task_status'] == 'SUCCESS'
    assert content['task_result']['successful'] is True
    assert content['task_result']['fresh'] is True
    assert content['task_result']['result'] == '020c000800021008020208000e0a120008000408080a30060208020c00000804'
