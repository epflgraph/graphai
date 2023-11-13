import json

import pytest
from unittest.mock import patch
from time import sleep

from graphai.api.celery_tasks.completion import request_text_completion_task, \
    compute_summarization_text_fingerprint_task

################################################################
# /completion/summary                                       #
# /completion/title                                         #
################################################################


@patch('graphai.api.celery_tasks.completion.request_text_completion_task.run')
@pytest.mark.usefixtures('transcript_text')
def test__summarization_summary__summarize_text__mock_task(mock_run, transcript_text):
    # Mock calling the task
    request_text_completion_task.run(transcript_text)

    # Assert that the task has been called
    assert request_text_completion_task.run.call_count == 1


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
    summary_transcript = request_text_completion_task.run(token_and_text, 'text', 'summary')

    # Assert that the results are correct
    assert isinstance(summary_transcript, dict)
    assert 'result' in summary_transcript
    assert 'summary_long' in summary_transcript['result'] and 'summary_short' in summary_transcript['result'] and \
           'title' in summary_transcript['result']
    assert summary_transcript['successful'] is True
    summary_text = summary_transcript['result']['summary_long'].lower()
    assert 'digital circuit' in summary_text
    assert 'simulation' in summary_text or 'simulator' in summary_text


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
    title_ocr = request_text_completion_task.run(token_and_text, 'slide', 'summary')

    # Assert that the results are correct
    assert isinstance(title_ocr, dict)
    assert 'result' in title_ocr
    assert 'summary_long' in title_ocr['result'] and 'summary_short' in title_ocr['result'] and \
           'title' in title_ocr['result']
    assert title_ocr['successful'] is True
    title_text = title_ocr['result']['summary_long'].lower()
    assert 'simulation' in title_text or 'simulator' in title_text
    assert 'digital' in title_text or 'discrete' in title_text or 'circuit' in title_text


################################################################


@pytest.mark.celery(accept_content=['pickle', 'json'], result_serializer='pickle', task_serializer='pickle')
@pytest.mark.usefixtures('slides_and_clean_concepts')
def test__summarization_summary_lecture__summarize_text__integration(
        fixture_app, celery_worker, slides_and_clean_concepts, timeout=30):
    # The celery_worker object is necessary for async tasks, otherwise the status will be permanently stuck on
    # PENDING.

    # First, we call the summary endpoint with force=True to test the full task pipeline working
    response = fixture_app.post('/completion/summary/lecture',
                                data=json.dumps({"slides": slides_and_clean_concepts,
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
        response = fixture_app.get(f'/completion/summary/lecture/status/{task_id}',
                                   timeout=timeout)
        current_status = response.json()['task_status']
        n_tries += 1

    # Now get status
    response = fixture_app.get(f'/completion/summary/lecture/status/{task_id}',
                               timeout=timeout)
    # Parse result
    summary_results = response.json()
    # Check returned value
    assert isinstance(summary_results, dict)
    assert 'task_result' in summary_results
    assert summary_results['task_status'] == 'SUCCESS'
    assert summary_results['task_result']['successful'] is True
    assert summary_results['task_result']['fresh'] is True
    summary_text = summary_results['task_result']['result']['summary_long'].lower()
    assert 'lecture' in summary_text
    assert 'climate' in summary_text and 'ice' in summary_text \
           and 'model' in summary_text and 'antarctic' in summary_text
    assert summary_results['task_result']['result_type'] == 'summary'
    original_summary = summary_text

    ################################################

    # Now, we call the summary endpoint again with the same input to make sure the caching works correctly
    response = fixture_app.post('/completion/summary/lecture',
                                data=json.dumps({"slides": slides_and_clean_concepts,
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
        response = fixture_app.get(f'/completion/summary/lecture/status/{task_id}',
                                   timeout=timeout)
        current_status = response.json()['task_status']
        n_tries += 1

    # Now get status
    response = fixture_app.get(f'/completion/summary/lecture/status/{task_id}',
                               timeout=timeout)
    # Parse result
    summary_results = response.json()
    # Check values
    assert isinstance(summary_results, dict)
    # results must be successful but not fresh, since they were already cached
    assert summary_results['task_status'] == 'SUCCESS'
    assert summary_results['task_result']['successful'] is True
    assert summary_results['task_result']['fresh'] is False
    assert summary_results['task_result']['result']['summary_long'].lower() == original_summary


@pytest.mark.celery(accept_content=['pickle', 'json'], result_serializer='pickle', task_serializer='pickle')
@pytest.mark.usefixtures('unit_info')
def test__summarization_summary_academic_entity__summarize_text__integration(
        fixture_app, celery_worker, unit_info, timeout=30):
    # The celery_worker object is necessary for async tasks, otherwise the status will be permanently stuck on
    # PENDING.

    data_dict = {"entity": unit_info['entity'],
                 "name": unit_info['name'],
                 "subtype": unit_info['subtype'],
                 "possible_subtypes": unit_info['possible_subtypes'],
                 "categories": unit_info['categories'],
                 "text": unit_info['text'],
                 "force": True}

    # First, we call the summary endpoint with force=True to test the full task pipeline working
    response = fixture_app.post('/completion/summary/academic_entity',
                                data=json.dumps(data_dict),
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
        response = fixture_app.get(f'/completion/summary/academic_entity/status/{task_id}',
                                   timeout=timeout)
        current_status = response.json()['task_status']
        n_tries += 1

    # Now get status
    response = fixture_app.get(f'/completion/summary/academic_entity/status/{task_id}',
                               timeout=timeout)
    # Parse result
    summary_results = response.json()
    # Check returned value
    assert isinstance(summary_results, dict)
    assert 'task_result' in summary_results
    assert summary_results['task_status'] == 'SUCCESS'
    assert summary_results['task_result']['successful'] is True
    assert summary_results['task_result']['fresh'] is True
    assert 'top_3_categories' in summary_results['task_result']['result'] and \
           'inferred_subtype' in summary_results['task_result']['result']
    summary_text = summary_results['task_result']['result']['summary_long'].lower()
    assert 'signal processing' in summary_text and 'audiovisual' in summary_text and 'research' in summary_text
    assert 'school of computer and communication sciences' in summary_text
    assert summary_results['task_result']['result_type'] == 'summary'
    original_summary = summary_text

    ################################################

    data_dict['force'] = False
    # Now, we call the summary endpoint again with the same input to make sure the caching works correctly
    response = fixture_app.post('/completion/summary/academic_entity',
                                data=json.dumps(data_dict),
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
        response = fixture_app.get(f'/completion/summary/academic_entity/status/{task_id}',
                                   timeout=timeout)
        current_status = response.json()['task_status']
        n_tries += 1

    # Now get status
    response = fixture_app.get(f'/completion/summary/academic_entity/status/{task_id}',
                               timeout=timeout)
    # Parse result
    summary_results = response.json()
    # Check values
    assert isinstance(summary_results, dict)
    # results must be successful but not fresh, since they were already cached
    assert summary_results['task_status'] == 'SUCCESS'
    assert summary_results['task_result']['successful'] is True
    assert summary_results['task_result']['fresh'] is False
    assert summary_results['task_result']['result']['summary_long'].lower() == original_summary


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
    assert fp['result'] == '020c000800021008020208000e0a120008000408080a30060208020c00000804_50_25'


################################################################


@pytest.mark.celery(accept_content=['pickle', 'json'], result_serializer='pickle', task_serializer='pickle')
@pytest.mark.usefixtures('transcript_text')
def test__summarization_calculate_fingerprint__compute_text_fingerprint__integration(
        fixture_app, celery_worker, transcript_text, timeout=30):
    # The celery_worker object is necessary for async tasks, otherwise the status will be permanently stuck on
    # PENDING.

    # Call the calculate_fingerprint endpoint with force=True
    response = fixture_app.post('/completion/calculate_fingerprint',
                                data=json.dumps({"text": transcript_text, "text_type": "text",
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
    assert content['task_result']['result'] == '020c000800021008020208000e0a120008000408080a30060208020c00000804_50_25'


@pytest.mark.celery(accept_content=['pickle', 'json'], result_serializer='pickle', task_serializer='pickle')
@pytest.mark.usefixtures('dirty_ocr_text')
def test__summarization_cleanup__cleanup_text__integration(fixture_app, celery_worker, dirty_ocr_text, timeout=30):
    # The celery_worker object is necessary for async tasks, otherwise the status will be permanently stuck on
    # PENDING.

    # First, we call the summary endpoint with force=True to test the full task pipeline working
    response = fixture_app.post('/completion/cleanup',
                                data=json.dumps({"text": dirty_ocr_text, "force": True}),
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
        response = fixture_app.get(f'/completion/cleanup/status/{task_id}',
                                   timeout=timeout)
        current_status = response.json()['task_status']
        n_tries += 1

    # Now get status
    response = fixture_app.get(f'/completion/cleanup/status/{task_id}',
                               timeout=timeout)
    # Parse result
    cleanup_results = response.json()
    # Check returned value
    assert isinstance(cleanup_results, dict)
    assert 'task_result' in cleanup_results
    assert cleanup_results['task_status'] == 'SUCCESS'
    assert cleanup_results['task_result']['successful'] is True
    assert cleanup_results['task_result']['fresh'] is True
    cleaned_up_text = cleanup_results['task_result']['result']['text'].lower()
    cleaned_up_subject = cleanup_results['task_result']['result']['subject'].lower()
    assert 'formalisme de hamilton' in cleaned_up_text
    assert 'legendre' in cleaned_up_text
    assert 'fonction inverse' in cleaned_up_text
    assert 'trouver' in cleaned_up_text
    assert 'hamilton' in cleaned_up_subject or 'legendre' in cleaned_up_subject
    assert cleanup_results['task_result']['result_type'] == 'cleanup'


@pytest.mark.celery(accept_content=['pickle', 'json'], result_serializer='pickle', task_serializer='pickle')
@pytest.mark.usefixtures('slides_and_raw_concepts')
def test__summarization__slide_subset__integration(fixture_app, celery_worker, slides_and_raw_concepts,
                                                             timeout=30):
    # The celery_worker object is necessary for async tasks, otherwise the status will be permanently stuck on
    # PENDING.

    # First, we call the summary endpoint with force=True to test the full task pipeline working
    response = fixture_app.post('/completion/slide_subset',
                                data=json.dumps({"slides": slides_and_raw_concepts}),
                                timeout=timeout)
    # Check status code is successful
    assert response.status_code == 200
    # Parse result
    results = response.json()
    # Check returned value
    assert isinstance(results, dict)
    assert 'subset' in results
    assert sorted(results['subset']) == [2, 9, 16, 24, 31, 34, 38]
