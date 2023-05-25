import json

import pytest
from unittest.mock import patch

import pandas as pd

from graphai.api.celery_tasks.text import extract_keywords_task, wikisearch_task, wikisearch_callback_task, compute_scores_task, aggregate_and_filter_task

################################################################
# /text/keywords                                               #
################################################################


@patch('graphai.api.celery_tasks.text.extract_keywords_task.run')
@pytest.mark.usefixtures('sultans')
def test__text_keywords__extract_keywords__mock_task(mock_run, sultans):
    # Mock calling the task
    extract_keywords_task.run(sultans)

    # Assert that the task has been called
    assert extract_keywords_task.run.call_count == 1

################################################################


@pytest.mark.usefixtures('sultans', 'wave_fields', 'schreier')
def test__text_keywords__extract_keywords__run_task(sultans, wave_fields, schreier):
    # Call task
    keywords_list = extract_keywords_task.run(sultans)

    # Check returned value
    assert isinstance(keywords_list, list)
    assert len(keywords_list) > 0
    assert 'trumpet playin' in keywords_list

    ################

    # Call task
    keywords_list = extract_keywords_task.run(wave_fields)
    print(keywords_list)

    # Check returned value
    assert isinstance(keywords_list, list)
    assert len(keywords_list) > 0
    assert 'acoustic wave fields' in keywords_list

    ################

    # Call task
    keywords_list = extract_keywords_task.run(schreier)
    print(keywords_list)

    # Check returned value
    assert isinstance(keywords_list, list)
    assert len(keywords_list) > 0
    assert 'schreier graphs' in keywords_list

################################################################


@pytest.mark.usefixtures('sultans', 'wave_fields', 'schreier')
def test__text_keywords__integration(fixture_app, sultans, wave_fields, schreier, timeout=30):
    # Make POST request to fixture fastapi app
    response = fixture_app.post('/text/keywords', data=json.dumps({'raw_text': sultans}), timeout=timeout)

    # Check status code is successful
    assert response.status_code == 200

    # Parse result
    keywords_list = response.json()

    # Check returned value
    assert isinstance(keywords_list, list)
    assert len(keywords_list) > 0
    assert 'trumpet playin' in keywords_list

    ################

    # Make POST request to fixture fastapi app
    response = fixture_app.post('/text/keywords', data=json.dumps({'raw_text': wave_fields}), timeout=timeout)

    # Check status code is successful
    assert response.status_code == 200

    # Parse result
    keywords_list = response.json()

    # Check returned value
    assert isinstance(keywords_list, list)
    assert len(keywords_list) > 0
    assert 'acoustic wave fields' in keywords_list

    ################

    # Make POST request to fixture fastapi app
    response = fixture_app.post('/text/keywords', data=json.dumps({'raw_text': schreier}), timeout=timeout)

    # Check status code is successful
    assert response.status_code == 200

    # Parse result
    keywords_list = response.json()

    # Check returned value
    assert isinstance(keywords_list, list)
    assert len(keywords_list) > 0
    assert 'schreier graphs' in keywords_list


################################################################
# /text/wikify                                                 #
################################################################


@patch('graphai.api.celery_tasks.text.wikisearch_task.run')
@pytest.mark.usefixtures('sultans')
def test__text_wikify__wikisearch__mock_task(mock_run, sultans):
    # Mock calling the task
    wikisearch_task.run(sultans)

    # Assert that the task has been called
    assert wikisearch_task.run.call_count == 1


@patch('graphai.api.celery_tasks.text.wikisearch_callback_task.run')
@pytest.mark.usefixtures('sultans')
def test__text_wikify__wikisearch_callback__mock_task(mock_run, sultans):
    # Mock calling the task
    wikisearch_callback_task.run(sultans)

    # Assert that the task has been called
    assert wikisearch_callback_task.run.call_count == 1


@patch('graphai.api.celery_tasks.text.compute_scores_task.run')
@pytest.mark.usefixtures('sultans')
def test__text_wikify__compute_scores__mock_task(mock_run, sultans):
    # Mock calling the task
    compute_scores_task.run(sultans)

    # Assert that the task has been called
    assert compute_scores_task.run.call_count == 1


@patch('graphai.api.celery_tasks.text.aggregate_and_filter_task.run')
@pytest.mark.usefixtures('sultans')
def test__text_wikify__aggregate_and_filter__mock_task(mock_run, sultans):
    # Mock calling the task
    aggregate_and_filter_task.run(sultans)

    # Assert that the task has been called
    assert aggregate_and_filter_task.run.call_count == 1

################################################################


def test__text_keywords__wikisearch__run_task(sultans, wave_fields, schreier):
    # Call task
    results = wikisearch_task.run(['trumpet playin'])

    # Check returned value
    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert not results.isna().values.any()
    assert 61460054 in results['PageID'].values     # Playin' for Keeps (Bunky Green album) wikipage

    ################

    # Call task
    results = wikisearch_task.run(['acoustic wave fields'])

    # Check returned value
    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert not results.isna().values.any()
    assert 33516 in results['PageID'].values        # Wave wikipage

    ################

    # Call task
    results = wikisearch_task.run(['schreier graphs'])

    # Check returned value
    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert not results.isna().values.any()
    assert 358277 in results['PageID'].values        # Cayley graph wikipage


@pytest.mark.usefixtures('wave_fields_wikisearch_df')
def test__text_keywords__wikisearch_callback__run_task(wave_fields_wikisearch_df):
    # Call task
    results = wikisearch_callback_task.run([wave_fields_wikisearch_df])

    # Check returned value
    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert not results.isna().values.any()
    assert results.equals(wave_fields_wikisearch_df)


@pytest.mark.usefixtures('wave_fields_wikisearch_df')
def test__text_keywords__compute_scores__run_task(wave_fields_wikisearch_df):
    # Call task
    results = compute_scores_task.run(wave_fields_wikisearch_df)

    # Check returned value
    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert not results.isna().values.any()
    assert 33516 in results['PageID'].values        # Wave wikipage
    for column in ['OntologyLocalScore', 'OntologyGlobalScore', 'GraphScore', 'KeywordsScore']:
        assert column in results.columns


@pytest.mark.usefixtures('wave_fields_scores_df')
def test__text_keywords__aggregate_and_filter__run_task(wave_fields_scores_df):
    # Call task
    results = aggregate_and_filter_task.run(wave_fields_scores_df)

    # Check returned value
    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert not results.isna().values.any()
    assert 33516 in results['PageID'].values        # Wave wikipage
    assert 'MixedScore' in results.columns


################################################################


@pytest.mark.usefixtures('wave_fields', 'schreier')
def test__text_wikify__integration(fixture_app, wave_fields, schreier, timeout=60):
    # Make POST request to fixture fastapi app
    response = fixture_app.post('/text/wikify', data=json.dumps({'raw_text': wave_fields}), timeout=timeout)

    # Check status code is successful
    assert response.status_code == 200

    # Parse result
    results = pd.DataFrame(response.json())

    # Check returned value
    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert 33516 in results['PageID'].values        # Wave wikipage
    for column in ['PageID', 'PageTitle', 'SearchScore', 'LevenshteinScore', 'OntologyLocalScore', 'OntologyGlobalScore', 'GraphScore', 'KeywordsScore', 'MixedScore']:
        assert column in results.columns
