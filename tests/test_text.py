import json

import pytest
from unittest.mock import patch

import pandas as pd

from graphai.core.common.config import config
from graphai.celery.text.tasks import (
    extract_keywords_task,
    text_init_task,
    wikisearch_task,
    wiki_search_task,
    compute_scores_task,
    draw_ontology_task,
    draw_graph_task,
)
from graphai.core.text.wikisearch import ElasticsearchSearchError, search_elasticsearch_http

################################################################
# /text/keywords                                               #
################################################################


@patch('graphai.celery.text.tasks.extract_keywords_task.run')
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
# /text/wiki_search                                            #
################################################################


@patch('graphai.celery.text.tasks.search_elasticsearch_http')
def test__text_wiki_search__run_task(mock_search):
    mock_search.return_value = [
        {'concept_id': 5786179, 'concept_name': 'Acoustic wave', 'score': 42.0},
        {'concept_id': 33516, 'concept_name': 'Wave', 'score': 21.0},
    ]

    results = wiki_search_task.run('acoustic wave fields', limit=2)

    assert results == mock_search.return_value
    mock_search.assert_called_once_with(
        'acoustic wave fields',
        config['elasticsearch'],
        index=config['elasticsearch'].get('concept_detection_index', 'concepts_detection'),
        limit=2,
        timeout=config['elasticsearch'].get('request_timeout', 10),
        timeout_retries=config['elasticsearch'].get('request_timeout_retries', 2),
    )


def test__text_wiki_search__integration(fixture_app):
    with patch('graphai.celery.text.jobs.wiki_search') as mock_wiki_search:
        mock_wiki_search.return_value = [
            {'concept_id': 5786179, 'concept_name': 'Acoustic wave', 'score': 42.0},
            {'concept_id': 33516, 'concept_name': 'Wave', 'score': 21.0},
        ]

        response = fixture_app.post(
            '/text/wiki_search?limit=2',
            json={'search_term': 'acoustic wave fields'},
            timeout=30,
        )

    assert response.status_code == 200

    results = response.json()

    assert isinstance(results, list)
    assert results == [
        {'concept_id': 5786179, 'concept_name': 'Acoustic wave', 'score': 42.0},
        {'concept_id': 33516, 'concept_name': 'Wave', 'score': 21.0},
    ]
    mock_wiki_search.assert_called_once()
    assert mock_wiki_search.call_args.args == ('acoustic wave fields', 2)
    assert 'request_id' in mock_wiki_search.call_args.kwargs


def test__text_wiki_search__integration__es_error_returns_503(fixture_app):
    with patch('graphai.celery.text.jobs.wiki_search') as mock_wiki_search:
        mock_wiki_search.side_effect = ElasticsearchSearchError(
            'Elasticsearch index "concepts_detection" is unavailable.',
            api_status_code=503,
            upstream_status=404,
        )

        response = fixture_app.post(
            '/text/wiki_search?limit=2',
            json={'search_term': 'acoustic wave fields'},
            timeout=30,
        )

    assert response.status_code == 503
    assert response.json() == {'detail': 'Elasticsearch index "concepts_detection" is unavailable.'}



def test__text_wikify__integration__es_error_returns_503_from_raw_text(fixture_app):
    with patch('graphai.celery.text.jobs.wikify_text') as mock_wikify_text:
        mock_wikify_text.side_effect = ElasticsearchSearchError(
            'Elasticsearch index "concepts_detection" is unavailable.',
            api_status_code=503,
            upstream_status=404,
        )

        response = fixture_app.post(
            '/text/wikify',
            json={'raw_text': 'acoustic wave fields'},
            timeout=30,
        )

    assert response.status_code == 503
    assert response.json() == {'detail': 'Elasticsearch index "concepts_detection" is unavailable.'}


def test__text_wikify__integration__es_error_returns_503_from_keywords(fixture_app):
    with patch('graphai.celery.text.jobs.wikify_keywords') as mock_wikify_keywords:
        mock_wikify_keywords.side_effect = ElasticsearchSearchError(
            'Elasticsearch index "concepts_detection" is unavailable.',
            api_status_code=503,
            upstream_status=404,
        )

        response = fixture_app.post(
            '/text/wikify',
            json={'keywords': ['acoustic wave fields']},
            timeout=30,
        )

    assert response.status_code == 503
    assert response.json() == {'detail': 'Elasticsearch index "concepts_detection" is unavailable.'}

class _DummyHTTPResponse:
    def __init__(self, status, payload):
        self.status = status
        self.data = payload.encode('utf-8')


class _DummyHTTPPool:
    def __init__(self, response):
        self.response = response

    def request(self, *args, **kwargs):
        return self.response


def test__text_wiki_search__http_error_raises():
    es_config = {
        'host': 'es01',
        'port': '9200',
        'username': 'elastic',
        'password': 'secret',
        'cafile': '/tmp/dummy.crt',
    }
    response = _DummyHTTPResponse(
        404,
        '{"error":{"type":"index_not_found_exception","reason":"no such index [concepts_detection]"},"status":404}',
    )

    with patch('graphai.core.text.wikisearch._es_http_pool', return_value=_DummyHTTPPool(response)):
        with pytest.raises(ElasticsearchSearchError) as exc_info:
            search_elasticsearch_http('acoustic wave fields', es_config, 'concepts_detection', limit=5, timeout=1, timeout_retries=1)

    assert exc_info.value.api_status_code == 503
    assert exc_info.value.upstream_status == 404
    assert 'concepts_detection' in str(exc_info.value)


@patch('graphai.celery.text.tasks.validate_elasticsearch_index_http', return_value=1579904)
@patch('graphai.celery.text.tasks.strtobool', return_value=False)
def test__text_init__run_task__validates_es_index(mock_preload, mock_validate):
    result = text_init_task.run()

    assert result is True
    mock_validate.assert_called_once_with(
        config['elasticsearch'],
        config['elasticsearch'].get('concept_detection_index', 'concepts_detection'),
        timeout=config['elasticsearch'].get('request_timeout', 10),
    )


################################################################
# /text/wikify                                                 #
################################################################


@patch('graphai.celery.text.tasks.wikisearch_task.run')
@pytest.mark.usefixtures('sultans')
def test__text_wikify__wikisearch__mock_task(mock_run, sultans):
    # Mock calling the task
    wikisearch_task.run(sultans)

    # Assert that the task has been called
    assert wikisearch_task.run.call_count == 1


@patch('graphai.celery.text.tasks.compute_scores_task.run')
def test__text_wikify__compute_scores__mock_task(mock_run):
    # Mock calling the task
    compute_scores_task.run()

    # Assert that the task has been called
    assert compute_scores_task.run.call_count == 1

################################################################


@pytest.mark.integration
def test__text_wikify__wikisearch__run_task():
    # Call task
    results = wikisearch_task.run([])
    df = results.to_dataframe() if hasattr(results, 'to_dataframe') else results

    # Check returned value
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0

    ################

    # Call task
    results = wikisearch_task.run(['acoustic wave fields'])
    df = results.to_dataframe() if hasattr(results, 'to_dataframe') else results

    # Check returned value
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert not df.isna().values.any()
    assert 5786179 in df['concept_id'].values        # Acoustic wave wikipage
    assert list(df.columns) == ['keywords', 'concept_id', 'concept_name', 'searchrank', 'search_score']

    ################

    # Call task
    results = wikisearch_task.run(['schreier graphs'])
    df = results.to_dataframe() if hasattr(results, 'to_dataframe') else results

    # Check returned value
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert not df.isna().values.any()
    assert 358277 in df['concept_id'].values        # Cayley graph wikipage
    assert list(df.columns) == ['keywords', 'concept_id', 'concept_name', 'searchrank', 'search_score']


@pytest.mark.usefixtures('wave_fields_wikisearch_df')
def test__text_wikify__compute_scores__run_task(wave_fields_wikisearch_df):
    # Call task
    results = compute_scores_task.run([wave_fields_wikisearch_df])
    df = results.to_dataframe() if hasattr(results, 'to_dataframe') else results

    # Check returned value
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert not df.isna().values.any()
    assert '33516' in df['concept_id'].values        # Wave wikipage
    assert list(df.columns) == ['concept_id', 'concept_name', 'search_score', 'levenshtein_score', 'embedding_local_score', 'embedding_global_score', 'graph_score', 'ontology_local_score', 'ontology_global_score', 'embedding_keywords_score', 'graph_keywords_score', 'ontology_keywords_score', 'mixed_score']

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
    assert 1196 in results['concept_id'].values        # Angle wikipage
    assert list(results.columns) == ['concept_id', 'concept_name', 'search_score', 'levenshtein_score', 'embedding_local_score', 'embedding_global_score', 'graph_score', 'ontology_local_score', 'ontology_global_score', 'embedding_keywords_score', 'graph_keywords_score', 'ontology_keywords_score', 'mixed_score']


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
    assert 33516 in results['concept_id'].values        # Wave wikipage
    assert list(results.columns) == ['concept_id', 'concept_name', 'search_score', 'levenshtein_score', 'embedding_local_score', 'embedding_global_score', 'graph_score', 'ontology_local_score', 'ontology_global_score', 'embedding_keywords_score', 'graph_keywords_score', 'ontology_keywords_score', 'mixed_score']


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
    assert 358277 in results['concept_id'].values        # Cayley graph wikipage
    assert list(results.columns) == ['concept_id', 'concept_name', 'search_score', 'levenshtein_score', 'embedding_local_score', 'embedding_global_score', 'graph_score', 'ontology_local_score', 'ontology_global_score', 'embedding_keywords_score', 'graph_keywords_score', 'ontology_keywords_score', 'mixed_score']


################################################################
# /text/wikify_ontology_svg                                    #
################################################################


@patch('graphai.celery.text.tasks.draw_ontology_task.run')
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


@patch('graphai.celery.text.tasks.draw_graph_task.run')
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
