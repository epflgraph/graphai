"""
Module which provides a tool to use the concept-detection API in order to wikify a text.
Wikifying a text is analyzing it and selecting a number of Wikipedia pages (concepts) which are relevant to it,
together with several scores that measure this relevance.

List of classes and functions:
"""

from requests import post
import numpy as np

WIKIFY_TEST_URL = 'http://86.119.27.90:28800/wikify'
WIKIFY_PROD_URL = 'http://86.119.30.77:28800/wikify'
WIKIFY_URL = WIKIFY_PROD_URL

# __all__ = ['Wikitext', 'combine']


def equivalent(result1, result2):
    """
    Returns whether two wikify results are equivalent, i.e. whether they have the same keywords and page id.

    Args:
        result1 (dict): A wikify result.
        result2 (dict): Another wikify result.

    Returns:
        bool: True if both results are equivalent, False otherwise.
    """
    return result1['keywords'] == result2['keywords'] and result1['page_id'] == result2['page_id']


def combine(wikitexts, f=None):
    """
    Returns all results for all wikitexts with the scores aggregated according to f.

    Args:
        wikitexts (list[Wikitext]): A list of Wikitext objects.
        f (Callable[list[Number], Number]): A function to aggregate the scores of the common pages.
            Default: None (numpy.mean)

    Returns:
        list[dict]: A list of the common results aggregated according to f.
    """

    if f is None:
        f = np.mean

    unique_pairs = set()
    for wikitext in wikitexts:
        unique_pairs |= set(wikitext.unique_pairs())

    agg_results = []
    for keywords, page_id in unique_pairs:
        pair_results = []
        for wikitext in wikitexts:
            pair_results.extend(wikitext.pair_results(keywords, page_id))

        agg_result = {}
        for key in pair_results[0]:
            if key == 'keywords' or key == 'page_id' or key == 'page_title':
                agg_result[key] = pair_results[0][key]
                continue

            agg_result[key] = f([result[key] for result in pair_results])

        agg_results.append(agg_result)

    return agg_results


class Wikitext:
    """
    A Wikitext is initialised with a text and computes its related Wikipedia concepts.

    Attributes:
        raw_text (str): A string with the text the Wikitext represents.
        anchor_page_ids (list[int]): A list containing the ids of the anchor pages to define the search space.
        results (list[dict]): A list of the results after wikifying the text.
    """

    def __init__(self, raw_text, anchor_page_ids=None):
        """
        Initialises Wikitext with a text and a list of page ids, and makes a request to the Concepts Detection API
        to compute the associated wikify results.

        Args:
            raw_text (str): A string with the text to wikify.
            anchor_page_ids (list[int]): A list containing the ids of the anchor pages to define the search space.
                If not provided, they are automatically generated. However, not providing them might lead to results
                which are not accurate enough. Default: None.
        """
        self.raw_text = raw_text
        self.anchor_page_ids = anchor_page_ids

        if anchor_page_ids is None:
            params = {
                'raw_text': raw_text
            }
        else:
            params = {
                'raw_text': raw_text,
                'anchor_page_ids': anchor_page_ids
            }

        # Get wikify results
        self.results = post(WIKIFY_URL, json=params).json()

    def keywords(self):
        """
        Returns:
            list[str]: A list of the unique keywords present in the results.
        """
        return list({result['keywords'] for result in self.results})

    def page_ids(self):
        """
        Returns:
            list[int]: A list of the unique page ids present in the results.
        """
        return list({result['page_id'] for result in self.results})

    def unique_pairs(self):
        """
        Returns:
            list[tuple(str, int)]: A list of the unique pairs (keywords, page id) present in the results.
        """
        return list({(result['keywords'], result['page_id']) for result in self.results})

    def page_titles(self):
        """
        Returns:
            list[str]: A list of the unique page titles present in the results.
        """
        return list({result['page_title'] for result in self.results})

    def keywords_results(self, keywords):
        """
        Returns:
            list[dict]: A list of the results matching the provided keywords.
        """
        return [result for result in self.results if result['keywords'] == keywords]

    def page_results(self, page_id):
        """
        Returns:
            list[dict]: A list of the results matching the provided page id.
        """
        return [result for result in self.results if result['page_id'] == page_id]

    def pair_results(self, keywords, page_id):
        """
        Returns:
            list[dict]: A list of the results matching the provided pair (keywords, page id).
        """
        return [result for result in self.results if result['keywords'] == keywords and result['page_id'] == page_id]

    def keywords_aggregated(self):
        """
        Aggregates results with the same keywords averaging their scores.

        Returns:
            list[dict]: A list of the aggregated results, unique by keywords. Their scores are averaged.
        """
        agg_results = []
        for keywords in self.keywords():
            keywords_results = self.keywords_results(keywords)

            agg_result = {}
            for key in keywords_results[0]:
                if key == 'keywords':
                    agg_result[key] = keywords_results[0][key]
                    continue

                if key == 'page_id' or key == 'page_title':
                    agg_result[f'{key}s'] = [result[key] for result in keywords_results]
                    continue

                agg_result[key] = np.mean([result[key] for result in keywords_results])

            agg_results.append(agg_result)

        return agg_results

    def page_aggregated(self):
        """
        Aggregates results with the same page id averaging their scores.

        Returns:
            list[dict]: A list of the aggregated results, unique by page id. Their scores are averaged.
        """
        agg_results = []
        for page_id in self.page_ids():
            page_results = self.page_results(page_id)

            agg_result = {}
            for key in page_results[0]:
                if key == 'keywords':
                    agg_result[f'{key}_list'] = [result[key] for result in page_results]
                    continue

                if key == 'page_id' or key == 'page_title':
                    agg_result[key] = page_results[0][key]
                    continue

                agg_result[key] = np.mean([result[key] for result in page_results])

            agg_results.append(agg_result)

        return agg_results
