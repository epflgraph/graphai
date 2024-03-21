import json

import pytest
from unittest.mock import patch

import pandas as pd

from graphai.api.celery_tasks.text import (
    extract_keywords_task,
    wikisearch_task,
    compute_scores_task,
    draw_ontology_task,
    draw_graph_task,
)

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
    keywords_list = extract_keywords_task.run('')

    # Check returned value
    assert isinstance(keywords_list, list)
    assert len(keywords_list) == 0

    ################

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


@pytest.mark.celery(accept_content=['pickle', 'json'], result_serializer='pickle', task_serializer='pickle')
@pytest.mark.usefixtures('sultans', 'wave_fields', 'schreier')
def test__text_keywords__integration(fixture_app, celery_worker, sultans, wave_fields, schreier, timeout=30):
    # Make POST request to fixture fastapi app
    response = fixture_app.post('/text/keywords', data=json.dumps({'raw_text': ''}), timeout=timeout)

    # Check status code is successful
    assert response.status_code == 200

    # Parse result
    keywords_list = response.json()

    # Check returned value
    assert isinstance(keywords_list, list)
    assert len(keywords_list) == 0

    ################

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


@patch('graphai.api.celery_tasks.text.compute_scores_task.run')
def test__text_wikify__compute_scores__mock_task(mock_run):
    # Mock calling the task
    compute_scores_task.run()

    # Assert that the task has been called
    assert compute_scores_task.run.call_count == 1

################################################################


def test__text_wikify__wikisearch__run_task():
    # Call task
    results = wikisearch_task.run([])

    # Check returned value
    assert isinstance(results, pd.DataFrame)
    assert len(results) == 0

    ################

    # Call task
    results = wikisearch_task.run(['acoustic wave fields'])

    # Check returned value
    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert not results.isna().values.any()
    assert 33516 in results['concept_id'].values        # Wave wikipage
    assert list(results.columns) == ['keywords', 'concept_id', 'concept_name', 'searchrank', 'search_score']

    ################

    # Call task
    results = wikisearch_task.run(['schreier graphs'])

    # Check returned value
    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert not results.isna().values.any()
    assert 358277 in results['concept_id'].values        # Cayley graph wikipage
    assert list(results.columns) == ['keywords', 'concept_id', 'concept_name', 'searchrank', 'search_score']


@pytest.mark.usefixtures('wave_fields_wikisearch_df')
def test__text_wikify__compute_scores__run_task(wave_fields_wikisearch_df):
    # Call task
    results = compute_scores_task.run([wave_fields_wikisearch_df])

    # Check returned value
    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert not results.isna().values.any()
    assert '33516' in results['concept_id'].values        # Wave wikipage
    assert list(results.columns) == ['concept_id', 'concept_name', 'search_score', 'levenshtein_score', 'graph_score', 'ontology_local_score', 'ontology_global_score', 'keywords_score', 'mixed_score']

################################################################


@pytest.mark.celery(accept_content=['pickle', 'json'], result_serializer='pickle', task_serializer='pickle')
@pytest.mark.usefixtures('wave_fields', 'schreier')
def test__text_wikify__integration(fixture_app, celery_worker, euclid, wave_fields, schreier, timeout=60):
    # Make POST request to fixture fastapi app
    response = fixture_app.post('/text/wikify', data=json.dumps({'raw_text': ''}), timeout=timeout)

    # Check status code is successful
    assert response.status_code == 200

    # Parse result
    results = pd.DataFrame(response.json())

    # Check returned value
    assert isinstance(results, pd.DataFrame)
    assert len(results) == 0

    ################

    # Make POST request to fixture fastapi app
    response = fixture_app.post('/text/wikify', data=json.dumps({'keywords': euclid}), timeout=timeout)

    # Check status code is successful
    assert response.status_code == 200

    # Parse result
    results = pd.DataFrame(response.json())

    # Check returned value
    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert 1196 in results['PageID'].values        # Angle wikipage
    assert list(results.columns) == ['PageID', 'PageTitle', 'SearchScore', 'LevenshteinScore', 'GraphScore', 'OntologyLocalScore', 'OntologyGlobalScore', 'KeywordsScore', 'MixedScore']


    ################

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
    assert list(results.columns) == ['PageID', 'PageTitle', 'SearchScore', 'LevenshteinScore', 'GraphScore', 'OntologyLocalScore', 'OntologyGlobalScore', 'KeywordsScore', 'MixedScore']


    ################

    # Make POST request to fixture fastapi app
    response = fixture_app.post('/text/wikify', data=json.dumps({'raw_text': schreier}), timeout=timeout)

    # Check status code is successful
    assert response.status_code == 200

    # Parse result
    results = pd.DataFrame(response.json())

    # Check returned value
    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert 358277 in results['PageID'].values        # Cayley graph wikipage
    assert list(results.columns) == ['PageID', 'PageTitle', 'SearchScore', 'LevenshteinScore', 'GraphScore', 'OntologyLocalScore', 'OntologyGlobalScore', 'KeywordsScore', 'MixedScore']


################################################################
# /text/wikify_ontology_svg                                    #
################################################################


@patch('graphai.api.celery_tasks.text.draw_ontology_task.run')
@pytest.mark.usefixtures('wave_fields_wikified_json')
def test__text_wikify_ontology_svg__draw_ontology__mock_task(mock_run, wave_fields_wikified_json):
    # Mock calling the task
    draw_ontology_task.run(wave_fields_wikified_json)

    # Assert that the task has been called
    assert draw_ontology_task.run.call_count == 1

################################################################


@pytest.mark.usefixtures('wave_fields_wikified_json')
def test__text_wikify_ontology_svg__draw_ontology__run_task(wave_fields_wikified_json):
    # Call task
    result = draw_ontology_task.run(wave_fields_wikified_json)

    # Check returned value
    assert result is None

################################################################


@pytest.mark.celery(accept_content=['pickle', 'json'], result_serializer='pickle', task_serializer='pickle')
@pytest.mark.usefixtures('wave_fields_wikified_json')
def test__text_wikify_ontology_svg__integration(fixture_app, celery_worker, wave_fields_wikified_json, timeout=30):
    # FIXME reactivate this test when wikify returns lowercase keys
    return

    # Make POST request to fixture fastapi app
    response = fixture_app.post('/text/wikify_ontology_svg', data=json.dumps([]), timeout=timeout)

    # Check status code is successful
    assert response.status_code == 200

    # Check returned value
    assert isinstance(response.content, bytes)
    svg = response.content.decode()
    assert len(svg) > 0
    assert '<svg' in svg

    ################

    # Make POST request to fixture fastapi app
    response = fixture_app.post('/text/wikify_ontology_svg', data=json.dumps(wave_fields_wikified_json), timeout=timeout)

    # Check status code is successful
    assert response.status_code == 200

    # Check returned value
    assert isinstance(response.content, bytes)
    svg = response.content.decode()
    assert len(svg) > 0
    assert '<svg' in svg


################################################################
# /text/wikify_graph_svg                                    #
################################################################


@patch('graphai.api.celery_tasks.text.draw_graph_task.run')
@pytest.mark.usefixtures('wave_fields_wikified_json')
def test__text_wikify_graph_svg__draw_graph__mock_task(mock_run, wave_fields_wikified_json):
    # Mock calling the task
    draw_graph_task.run(wave_fields_wikified_json)

    # Assert that the task has been called
    assert draw_graph_task.run.call_count == 1

################################################################


@pytest.mark.usefixtures('wave_fields_wikified_json')
def test__text_wikify_graph_svg__draw_graph__run_task(wave_fields_wikified_json):
    # Call task
    result = draw_graph_task.run(wave_fields_wikified_json)

    # Check returned value
    assert result is None

################################################################


@pytest.mark.celery(accept_content=['pickle', 'json'], result_serializer='pickle', task_serializer='pickle')
@pytest.mark.usefixtures('wave_fields_wikified_json')
def test__text_wikify_graph_svg__integration(fixture_app, celery_worker, wave_fields_wikified_json, timeout=30):
    # FIXME reactivate this test when wikify returns lowercase keys
    return

    # Make POST request to fixture fastapi app
    response = fixture_app.post('/text/wikify_graph_svg', data=json.dumps([]), timeout=timeout)

    # Check status code is successful
    assert response.status_code == 200

    # Check returned value
    assert isinstance(response.content, bytes)
    svg = response.content.decode()
    assert len(svg) > 0
    assert '<svg' in svg

    ################

    # Make POST request to fixture fastapi app
    response = fixture_app.post('/text/wikify_graph_svg', data=json.dumps(wave_fields_wikified_json), timeout=timeout)

    # Check status code is successful
    assert response.status_code == 200

    # Check returned value
    assert isinstance(response.content, bytes)
    svg = response.content.decode()
    assert len(svg) > 0
    assert '<svg' in svg
