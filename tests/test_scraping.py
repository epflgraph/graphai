import json

import pytest
from unittest.mock import patch
from time import sleep

from graphai.api.celery_tasks.scraping import initialize_url_and_get_sublinks_task

from graphai.core.scraping.scraping import create_base_url_token

################################################################
# /scraping/sublinks                                           #
# /scraping/content                                            #
################################################################


@patch('graphai.api.celery_tasks.scraping.initialize_url_and_get_sublinks_task.run')
@pytest.mark.usefixtures('test_url')
def test__scraping_sublinks__initialize_url_and_get_sublinks_task__mock_task(mock_run, test_url):
    # Mock calling the task
    initialize_url_and_get_sublinks_task.run('aaaaaa', test_url)

    # Assert that the task has been called
    assert initialize_url_and_get_sublinks_task.run.call_count == 1


################################################################


@pytest.mark.usefixtures('test_url')
def test__scraping_sublinks__initialize_url_and_get_sublinks_task__run_task(test_url):
    # Call the task
    sublinks_results = initialize_url_and_get_sublinks_task.run(create_base_url_token(test_url), test_url)

    # Assert that the results are correct
    assert isinstance(sublinks_results, dict)
    assert 'data' in sublinks_results
    assert 'sublinks' in sublinks_results
    assert sublinks_results['successful'] is True
    assert len(sublinks_results['data']) > 0
    sublinks = sublinks_results['sublinks']
    assert 'https://www.epfl.ch/labs/chili/people' in sublinks
    assert 'https://www.epfl.ch/labs/chili/dualt' in sublinks


#############################################################

@pytest.mark.celery(accept_content=['pickle', 'json'], result_serializer='pickle', task_serializer='pickle')
@pytest.mark.usefixtures('test_url_2')
def test__scraping_content__process_all_sublinks__integration(fixture_app, celery_worker, test_url_2,
                                                              timeout=30):
    # The celery_worker object is necessary for async tasks, otherwise the status will be permanently stuck on
    # PENDING.
    # First, we call the summary endpoint with force=True to test the full task pipeline working
    response = fixture_app.post('/scraping/content',
                                data=json.dumps({"url": test_url_2,
                                                 "force": True, "remove_headers": False,
                                                 "remove_long_patterns": False}),
                                timeout=timeout)
    # Check status code is successful
    assert response.status_code == 200
    # Parse resulting task id
    task_id = response.json()['task_id']

    # Waiting for task chain to succeed
    current_status = 'PENDING'
    n_tries = 0
    while current_status == 'PENDING' and n_tries < 50:
        # Wait a few seconds
        sleep(3)
        # Now get status
        response = fixture_app.get(f'/scraping/content/status/{task_id}',
                                   timeout=timeout)
        current_status = response.json()['task_status']
        n_tries += 1

    # Now get status
    response = fixture_app.get(f'/scraping/content/status/{task_id}',
                               timeout=timeout)
    # Parse result
    summary_results = response.json()
    # Check returned value
    assert isinstance(summary_results, dict)
    assert 'task_result' in summary_results
    assert summary_results['task_status'] == 'SUCCESS'
    assert summary_results['task_result']['successful'] is True
    assert summary_results['task_result']['fresh'] is True
    data = summary_results['task_result']['data']
    assert 'https://www.epfl.ch/fr' in data
    assert data['https://www.epfl.ch/fr']['pagetype'] == 'homepage'
    assert len(data['https://www.epfl.ch/fr']['content']) > 0
    original_data = data

    ################################################

    # Now, we call the summary endpoint again with the same input to make sure the caching works correctly
    response = fixture_app.post('/scraping/content',
                                data=json.dumps({"url": test_url_2,
                                                 "force": False, "remove_headers": False,
                                                 "remove_long_patterns": False}),
                                timeout=timeout)
    # Check status code is successful
    assert response.status_code == 200
    # Parse resulting task id
    task_id = response.json()['task_id']

    # Waiting for task chain to succeed
    current_status = 'PENDING'
    n_tries = 0
    while current_status == 'PENDING' and n_tries < 80:
        # Wait a few seconds
        sleep(3)
        # Now get status
        response = fixture_app.get(f'/scraping/content/status/{task_id}',
                                   timeout=timeout)
        current_status = response.json()['task_status']
        n_tries += 1

    # Now get status
    response = fixture_app.get(f'/scraping/content/status/{task_id}',
                               timeout=timeout)
    # Parse result
    summary_results = response.json()
    # Check values
    assert isinstance(summary_results, dict)
    # results must be successful but not fresh, since they were already cached
    assert summary_results['task_status'] == 'SUCCESS'
    assert summary_results['task_result']['successful'] is True
    assert summary_results['task_result']['fresh'] is False
    assert summary_results['task_result']['data'] == original_data
