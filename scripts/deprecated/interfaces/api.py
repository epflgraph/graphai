from requests import post

from definitions import TEST_API_URL
from models.wikify_result import WikifyResult


class Api:
    """
    Base class to communicate with the Graph AI API.
    """

    def __init__(self):
        self.url = TEST_API_URL

    def keywords(self, raw_text):
        """
        Performs an API request to the /keywords entry point with the given text.

        Args:
            raw_text (str): Text to be used as input for the request.

        Returns:
            list[str]: List of keywords extracted from the given text from the response.
        """

        params = {
            'raw_text': raw_text
        }

        return post(f'{self.url}/keywords', json=params).json()

    def wikify(self, raw_text, anchor_page_ids=None, method=None):
        """
        Performs an API request to the /wikify entry point with the given text.

        Args:
            raw_text (str): Text to be used as input for the request.
            anchor_page_ids (list[int]): List of anchor page ids to be used as input for the request.
            method (str{'wikipedia-api', 'es-base', 'es-score'}): Wikisearch method to be used as input for the request.

        Returns:
            list[dict[str]]: List of wikify results from the response.
        """

        params = {
            'raw_text': raw_text
        }

        if anchor_page_ids is not None:
            params['anchor_page_ids'] = anchor_page_ids

        url = f'{self.url}/wikify'
        if method is not None:
            url += f'?method={method}'

        results = post(url, json=params).json()
        return list(map(WikifyResult.from_dict, results))

    def wikify_keywords(self, keyword_list, anchor_page_ids=None, method=None):
        """
        Performs an API request to the /wikify entry point with the given list of keywords.

        Args:
            keyword_list (str): List of keywords to be used as input for the request.
            anchor_page_ids (list[int]): List of anchor page ids to be used as input for the request.
            method (str{'wikipedia-api', 'es-base', 'es-score'}): Wikisearch method to be used as input for the request.

        Returns:
            list[dict[str]]: List of wikify results from the response.
        """

        params = {
            'keyword_list': keyword_list
        }

        if anchor_page_ids is not None:
            params['anchor_page_ids'] = anchor_page_ids

        url = f'{self.url}/wikify'
        if method is not None:
            url += f'?method={method}'

        return post(url, json=params).json()
