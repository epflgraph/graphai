import requests

from concept_detection.test.conftest import *

TEST_API_URL = 'http://86.119.27.90:28800'


def test_keywords():
    # call with empty params
    response = requests.post(f'{TEST_API_URL}/keywords', json={})
    assert response.status_code == 422  # Unprocessable entity

    for fixture in fixtures:
        params = {
            'raw_text': fixture['raw_text']
        }

        # python-rake method
        response = requests.post(f'{TEST_API_URL}/keywords', json=params)
        assert response.status_code == 200

        keyword_list = response.json()
        assert set(keyword_list) == set(fixture['keyword_list'])

        # nltk-rake method
        response = requests.post(f'{TEST_API_URL}/keywords?method=nltk', json=params)
        assert response.status_code == 200

        keyword_list = response.json()
        assert set(keyword_list) == set(fixture['keyword_list_nltk'])
