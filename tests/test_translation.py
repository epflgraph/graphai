import json

import pytest
from unittest.mock import patch
from time import sleep

from graphai.api.celery_tasks.translation import translate_text_task, compute_text_fingerprint_task


################################################################
# /translation/translate                                       #
################################################################


@patch('graphai.api.celery_tasks.translation.translate_text_task.run')
@pytest.mark.usefixtures('en_to_fr_text')
def test__translation_translate__translate_text__mock_task(mock_run, en_to_fr_text):
    # Mock calling the task
    translate_text_task.run(en_to_fr_text)

    # Assert that the task has been called
    assert translate_text_task.run.call_count == 1


################################################################


@pytest.mark.usefixtures('en_to_fr_text', 'fr_to_en_text')
def test__translation_translate__translate_text__run_task(en_to_fr_text, fr_to_en_text):
    # Call the task
    en_fr_translated = translate_text_task.run('mock_token_en_fr', en_to_fr_text, "en", "fr", True)

    # Assert that the results are correct
    assert isinstance(en_fr_translated, dict)
    assert 'result' in en_fr_translated
    assert en_fr_translated['successful'] is True
    assert 'salut les gars' in en_fr_translated['result'].lower()
    assert 'comment ça va' in en_fr_translated['result'].lower()

    #############################################################

    # Call the task
    fr_en_translated = translate_text_task.run('mock_token_fr_en', fr_to_en_text, "fr", "en", True)

    # Assert that the results are correct
    assert isinstance(fr_en_translated, dict)
    assert 'result' in fr_en_translated
    assert fr_en_translated['successful'] is True
    assert 'ladies and gentlemen' in fr_en_translated['result'].lower()
    assert 'welcome' in fr_en_translated['result'].lower()


################################################################


@pytest.mark.celery(accept_content=['pickle', 'json'], result_serializer='pickle', task_serializer='pickle')
@pytest.mark.usefixtures('en_to_fr_text')
def test__translation_translate__translate_text__integration(fixture_app, celery_worker, en_to_fr_text, timeout=30):
    # The celery_worker object is necessary for async tasks, otherwise the status will be permanently stuck on
    # PENDING.

    # First, we call the translate endpoint with force=True to test the full task pipeline working
    response = fixture_app.post('/translation/translate',
                                data=json.dumps({"text": en_to_fr_text, "source": "en", "target": "fr",
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
        response = fixture_app.get(f'/translation/translate/status/{task_id}',
                                   timeout=timeout)
        current_status = response.json()['task_status']
        n_tries += 1

    # Now get status
    response = fixture_app.get(f'/translation/translate/status/{task_id}',
                               timeout=timeout)
    # Parse result
    en_fr_translated = response.json()
    # Check returned value
    assert isinstance(en_fr_translated, dict)
    assert 'task_result' in en_fr_translated
    assert en_fr_translated['task_status'] == 'SUCCESS'
    assert en_fr_translated['task_result']['successful'] is True
    assert en_fr_translated['task_result']['fresh'] is True
    assert 'salut les gars' in en_fr_translated['task_result']['result'].lower()
    assert 'comment ça va' in en_fr_translated['task_result']['result'].lower()
    original_results = en_fr_translated['task_result']['result']

    ################################################

    # Now, we call the translate endpoint again with the same input to make sure the caching works correctly
    response = fixture_app.post('/translation/translate',
                                data=json.dumps({"text": en_to_fr_text, "source": "en", "target": "fr",
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
        response = fixture_app.get(f'/translation/translate/status/{task_id}',
                                   timeout=timeout)
        current_status = response.json()['task_status']
        n_tries += 1

    # Now get status
    response = fixture_app.get(f'/translation/translate/status/{task_id}',
                               timeout=timeout)
    # Parse result
    en_fr_translated = response.json()
    # Check values
    assert isinstance(en_fr_translated, dict)
    # results must be successful but not fresh, since they were already cached
    assert en_fr_translated['task_status'] == 'SUCCESS'
    assert en_fr_translated['task_result']['successful'] is True
    assert en_fr_translated['task_result']['fresh'] is False
    assert en_fr_translated['task_result']['result'] == original_results


################################################################
# /translation/calculate_fingerprint                           #
################################################################


@pytest.mark.usefixtures('en_to_fr_text', 'fr_to_en_text')
def test__translation_calculate_fingerprint__compute_text_fingerprint__run_task(en_to_fr_text, fr_to_en_text):
    # Call the task
    fp = compute_text_fingerprint_task.run('mock_token_en_fr', en_to_fr_text, True)

    # Assert that the results are correct
    assert isinstance(fp, dict)
    assert 'result' in fp
    assert fp['fresh']
    assert fp['fp_token'] == 'mock_token_en_fr'
    assert fp['result'] == '0600000000000000000000000000000000000000000000000000000000000000'

    #############################################################

    # Call the task
    fp = compute_text_fingerprint_task.run('mock_token_fr_en', fr_to_en_text, True)

    # Assert that the results are correct
    assert isinstance(fp, dict)
    assert 'result' in fp
    assert fp['fresh']
    assert fp['fp_token'] == 'mock_token_fr_en'
    assert fp['result'] == '263e3e409aaa6600000000000000000000000000000000000000000000000000'


################################################################


@pytest.mark.celery(accept_content=['pickle', 'json'], result_serializer='pickle', task_serializer='pickle')
@pytest.mark.usefixtures('en_to_fr_text')
def test__translation_calculate_fingerprint__compute_text_fingerprint__integration(
        fixture_app, celery_worker, en_to_fr_text, timeout=30):
    # The celery_worker object is necessary for async tasks, otherwise the status will be permanently stuck on
    # PENDING.

    # Call the calculate_fingerprint endpoint with force=True
    response = fixture_app.post('/translation/calculate_fingerprint',
                                data=json.dumps({"text": en_to_fr_text, "source": "en", "target": "fr",
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
        response = fixture_app.get(f'/translation/calculate_fingerprint/status/{task_id}',
                                   timeout=timeout)
        current_status = response.json()['task_status']
        n_tries += 1

    # Now get status
    response = fixture_app.get(f'/translation/calculate_fingerprint/status/{task_id}',
                               timeout=timeout)
    # Parse result
    content = response.json()
    # Check returned value
    assert isinstance(content, dict)
    assert 'task_result' in content
    assert content['task_status'] == 'SUCCESS'
    assert content['task_result']['successful'] is True
    assert content['task_result']['fresh'] is True
    assert content['task_result']['result'] == '0600000000000000000000000000000000000000000000000000000000000000'
