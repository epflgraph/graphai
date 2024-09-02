import json
import numpy as np

import pytest
from unittest.mock import patch
from time import sleep

from graphai.celery.embedding.tasks import (
    embed_text_task,
)
from graphai.core.common.caching import EmbeddingDBCachingManager


################################################################
# /translation/translate                                       #
################################################################


@patch('graphai.celery.embedding.tasks.embed_text_task.run')
@pytest.mark.usefixtures('example_word')
def test__embedding_embed__translate_text__mock_task(mock_run, example_word):
    # Mock calling the task
    embed_text_task.run(example_word, 'all-MiniLM-L12-v2')

    # Assert that the task has been called
    assert embed_text_task.run.call_count == 1


################################################################


@pytest.mark.usefixtures('example_word', 'very_long_text')
def test__translation_translate__translate_text__run_task(example_word, very_long_text):
    # Call the task
    embedding = embed_text_task.run(example_word, "all-MiniLM-L12-v2")

    # Assert that the results are correct
    assert isinstance(embedding, dict)
    assert 'result' in embedding
    assert embedding['successful'] is True
    assert embedding['text_too_large'] is False
    assert embedding['result'].shape[0] == 384
    assert 0.999 < np.dot(embedding['result'], embedding['result']) < 1.001

    # Call the task
    embedding = embed_text_task.run(very_long_text, "all-MiniLM-L12-v2")

    # Assert that the results are correct
    assert isinstance(embedding, dict)
    assert 'result' in embedding
    assert embedding['successful'] is False
    assert embedding['text_too_large'] is True
    assert embedding['result'] == "Text over token limit for selected model (128)."


@pytest.mark.celery(accept_content=['pickle', 'json'], result_serializer='pickle', task_serializer='pickle')
@pytest.mark.usefixtures('example_word', 'very_long_text', 'example_word_list')
def test__translation_translate__translate_text__integration(fixture_app, celery_worker,
                                                             example_word, very_long_text, example_word_list,
                                                             timeout=30):
    # The celery_worker object is necessary for async tasks, otherwise the status will be permanently stuck on
    # PENDING.

    # This line ensures the initialization of the database in case this is the first deployment
    EmbeddingDBCachingManager(initialize_database=True)

    # First, we call the embedding endpoint with force=True to test the full task pipeline working
    response = fixture_app.post('/embedding/embed',
                                data=json.dumps({"text": example_word, "model_type": "all-MiniLM-L12-v2",
                                                 "force": True}),
                                timeout=timeout)
    # Check status code is successful
    assert response.status_code == 200
    # Parse resulting task id
    task_id = response.json()['task_id']

    # Waiting for task chain to succeed
    current_status = 'PENDING'
    n_tries = 0
    while current_status == 'PENDING' and n_tries < 20:
        # Wait a few seconds
        sleep(3)
        # Now get status
        response = fixture_app.get(f'/embedding/embed/status/{task_id}',
                                   timeout=timeout)
        current_status = response.json()['task_status']
        n_tries += 1

    # Now get status
    response = fixture_app.get(f'/embedding/embed/status/{task_id}',
                               timeout=timeout)
    # Parse result
    embedding = response.json()
    # Check returned value
    assert isinstance(embedding, dict)
    assert 'task_result' in embedding
    assert embedding['task_status'] == 'SUCCESS'
    assert embedding['task_result']['successful'] is True
    assert embedding['task_result']['text_too_large'] is False
    assert embedding['task_result']['fresh'] is True
    assert len(json.loads(embedding['task_result']['result'])) == 384
    assert isinstance(json.loads(embedding['task_result']['result'])[0], float)

    original_results = embedding['task_result']['result']

    ################################################

    # Now, we call the translate endpoint again with the same input to make sure the caching works correctly
    response = fixture_app.post('/embedding/embed',
                                data=json.dumps({"text": example_word, "model_type": "all-MiniLM-L12-v2",
                                                 "force": False}),
                                timeout=timeout)
    # Check status code is successful
    assert response.status_code == 200
    # Parse resulting task id
    task_id = response.json()['task_id']

    # Waiting for task chain to succeed
    current_status = 'PENDING'
    n_tries = 0
    while current_status == 'PENDING' and n_tries < 20:
        # Wait a few seconds
        sleep(3)
        # Now get status
        response = fixture_app.get(f'/embedding/embed/status/{task_id}',
                                   timeout=timeout)
        current_status = response.json()['task_status']
        n_tries += 1

    # Now get status
    response = fixture_app.get(f'/embedding/embed/status/{task_id}',
                               timeout=timeout)
    # Parse result
    embedding = response.json()
    # Check values
    assert isinstance(embedding, dict)
    # results must be successful but not fresh, since they were already cached
    assert embedding['task_status'] == 'SUCCESS'
    assert embedding['task_result']['successful'] is True
    assert embedding['task_result']['text_too_large'] is False
    assert embedding['task_result']['fresh'] is False
    assert embedding['task_result']['result'] == original_results

    response = fixture_app.post('/embedding/embed',
                                data=json.dumps({"text": [example_word, very_long_text] + example_word_list,
                                                 "model_type": "all-MiniLM-L12-v2",
                                                 "force": True}),
                                timeout=timeout)
    # Check status code is successful
    assert response.status_code == 200
    # Parse resulting task id
    task_id = response.json()['task_id']

    # Waiting for task chain to succeed
    current_status = 'PENDING'
    n_tries = 0
    while current_status == 'PENDING' and n_tries < 20:
        # Wait a few seconds
        sleep(3)
        # Now get status
        response = fixture_app.get(f'/embedding/embed/status/{task_id}',
                                   timeout=timeout)
        current_status = response.json()['task_status']
        n_tries += 1

    # Now get status
    response = fixture_app.get(f'/embedding/embed/status/{task_id}',
                               timeout=timeout)
    # Parse result
    embedding = response.json()
    # Check returned value
    assert isinstance(embedding, dict)
    assert embedding['task_status'] == 'SUCCESS'
    assert len(embedding['task_result']) == 2 + len(example_word_list)
    assert embedding['task_result'][0]['successful'] is True
    assert embedding['task_result'][0]['result'] == original_results
    assert embedding['task_result'][1]['successful'] is False
    assert embedding['task_result'][1]['result'] == "Text over token limit for selected model (128)."
    # All except one must have been successful
    assert sum([1 if embedding['task_result'][i]['successful'] else 0
                for i in range(len(embedding['task_result']))]) == len(embedding['task_result']) - 1
