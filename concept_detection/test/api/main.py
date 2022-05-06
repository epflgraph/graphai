import requests

from concept_detection.test.conftest import *

TEST_API_URL = 'http://86.119.27.90:28800'


def test_keywords():
    # Empty call
    response = requests.post(f'{TEST_API_URL}/keywords', json={})
    assert response.status_code == 422  # Unprocessable entity

    for fixture in fixtures:
        params = {
            'raw_text': fixture['raw_text']
        }

        # Method python-rake
        response = requests.post(f'{TEST_API_URL}/keywords', json=params)
        assert response.status_code == 200
        keyword_list = response.json()
        assert set(keyword_list) == set(fixture['keyword_list'])

        # Method nltk-rake
        response = requests.post(f'{TEST_API_URL}/keywords?method=nltk', json=params)
        assert response.status_code == 200
        keyword_list = response.json()
        assert set(keyword_list) == set(fixture['keyword_list_nltk'])


def _test_wikify_case(url, params):
    response = requests.post(url, json=params)
    assert response.status_code == 200
    results = response.json()
    assert results
    for result in results:
        assert result['mixed_score'] > 0


def test_wikify():
    # Empty call
    response = requests.post(f'{TEST_API_URL}/wikify', json={})
    assert response.status_code == 200
    results = response.json()
    assert not results

    for fixture in fixtures:
        url = f'{TEST_API_URL}/wikify'

        # From raw_text
        params = {'raw_text': fixture['raw_text']}
        _test_wikify_case(url, params)

        # From keywords
        params = {'keyword_list': fixture['keyword_list']}
        _test_wikify_case(url, params)

        # From raw_text, specifying anchor pages
        params = {'raw_text': fixture['raw_text'], 'anchor_page_ids': fixture['anchor_page_ids']}
        _test_wikify_case(url, params)

        # From keywords, specifying anchor pages
        params = {'keyword_list': fixture['keyword_list'], 'anchor_page_ids': fixture['anchor_page_ids']}
        _test_wikify_case(url, params)

        url = f'{TEST_API_URL}/wikify?method=es-base'

        # From raw_text, specifying anchor pages and elasticsearch method es-base
        params = {'raw_text': fixture['raw_text'], 'anchor_page_ids': fixture['anchor_page_ids']}
        _test_wikify_case(url, params)

        # From keywords, specifying anchor pages and elasticsearch method es-base
        params = {'keyword_list': fixture['keyword_list'], 'anchor_page_ids': fixture['anchor_page_ids']}
        _test_wikify_case(url, params)

        url = f'{TEST_API_URL}/wikify?method=es-score'

        # From raw_text, specifying anchor pages and elasticsearch method es-base
        params = {'raw_text': fixture['raw_text'], 'anchor_page_ids': fixture['anchor_page_ids']}
        _test_wikify_case(url, params)

        # From keywords, specifying anchor pages and elasticsearch method es-base
        params = {'keyword_list': fixture['keyword_list'], 'anchor_page_ids': fixture['anchor_page_ids']}
        _test_wikify_case(url, params)



