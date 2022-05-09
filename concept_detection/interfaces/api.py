from requests import post

from definitions import TEST_API_URL
from concept_detection.test.types import WikifyResult


class Api:

    def keywords(self, raw_text):
        params = {
            'raw_text': raw_text
        }

        return post(f'{TEST_API_URL}/keywords', json=params).json()

    def wikify(self, raw_text, anchor_page_ids=None, method=None):
        params = {
            'raw_text': raw_text
        }

        if anchor_page_ids:
            params['anchor_page_ids'] = anchor_page_ids

        url = f'{TEST_API_URL}/wikify'
        if method:
            url += f'?method={method}'

        results = post(url, json=params).json()
        return list(map(WikifyResult.from_dict, results))

    def wikify_keywords(self, keyword_list, anchor_page_ids=None):
        if anchor_page_ids is None:
            params = {
                'keyword_list': keyword_list
            }
        else:
            params = {
                'keyword_list': keyword_list,
                'anchor_page_ids': anchor_page_ids
            }

        return post(f'{TEST_API_URL}/wikify', json=params).json()
