import requests
import ray
# import wikipedia

from concept_detection.search.types import *
from concept_detection.text.utils import decode_url_title

API_URL = 'http://en.wikipedia.org/w/api.php'
HEADERS = {'User-Agent': 'graphai (https://github.com/epflgraph/graphai)'}

# # Set wikipedia language
# wikipedia.set_lang('en')

# Init ray
ray.init(namespace="wikisearch", include_dashboard=False, log_to_driver=True)


@ray.remote
class WikisearchActor:
    """
    Class representing a ray Actor to perform wikisearch in parallel.
    """
    def wikisearch(self, keywords):
        """
        Searches Wikipedia API for input keywords. Returns top 10 results.

        Args:
            keywords (str): Text to input in Wikipedia's search field.

        Returns:
            WikisearchResult: Object containing the given keywords and their associated list of page results.
        """

        # Send request to Wikipedia API
        params = {
            'format': 'json',
            'action': 'query',
            'list': 'search',
            'srsearch': keywords,
            'srlimit': 10,
            'srprop': ''
        }
        r = requests.get(API_URL, params=params, headers=HEADERS).json()

        if 'error' in r:
            raise Exception(r['error']['info'])

        top_pages = r['query']['search']

        pages = [
            PageResult(
                page_id=top_pages[i]['pageid'],
                page_title=top_pages[i]['title'],
                searchrank=(i + 1),
                score=(1 / (i + 1))
            )
            for i in range(len(top_pages))
        ]

        return WikisearchResult(keywords=keywords, pages=pages)


        # # Run search on Wikipedia API
        # top_page_titles = wikipedia.search(keywords)
        #
        # # Create list of PageResults with escaped titles
        # n = min(len(top_page_titles), 10)
        # pages = [
        #     PageResult(
        #         page_id=0,
        #         page_title=top_page_titles[i].lower().replace('_', ' ').replace("'", '<squote/>').replace('"', '<dquote/>'),
        #         searchrank=i + 1,
        #         score=1 / (i + 1)
        #     )
        #     for i in range(n)
        # ]
        #
        # # Return WikisearchResult with keywords and pages
        # return WikisearchResult(keywords=keywords, pages=pages)


# Instantiate ray actor list
n_actors = 16
actors = [WikisearchActor.remote() for i in range(n_actors)]


def clean(keyword_list):
    """
    Cleans all the keywords in the given keyword list by applying the decode_url_title function.

    Args:
        keyword_list (list of str): List of keywords to be cleaned.

    Returns:
        list of str: List of cleaned keywords.
    """
    return [decode_url_title(keywords) for keywords in keyword_list]


# def postprocess(results, page_title_ids):
#     """
#     Modifies the given results to include page ids in addition to page titles.
#
#     Args:
#         results (list of WikisearchResult): List of wikisearch results.
#         page_title_ids (dict of str: int): Mapping from page titles to ids.
#
#     Returns:
#         list of WikisearchResult: The list of wikisearch results, with updated page ids based on the page titles.
#     """
#
#     for result in results:
#         for i in range(len(result.pages)):
#             # Get page id for the given page title from mapping if present, None otherwise
#             page_title = result.pages[i].page_title
#             result.pages[i].page_id = page_title_ids.get(page_title, None)
#
#     return results


def extract_page_ids(results):
    """
    Iterates through the given wikisearch results and returns a list with all the page ids.

    Args:
        results (list of WikisearchResult): List of wikisearch results.

    Returns:
        list of int: List of all page ids present along all results.
    """
    return list(set(page.page_id for result in results for page in result.pages))


def extract_anchor_page_ids(results, max_n=3):
    """
    Iterates through the given wikisearch results and returns a list with the most relevant page ids.

    Args:
        results (list of WikisearchResult): List of wikisearch results.
        max_n (int): Maximum number of page ids returned. Default: 3.

    Returns:
        list of int: List of the most relevant page ids present along all results.
    """

    # Compute sum of scores for each page over all results
    page_scores = {}
    for result in results:
        for page in result.pages:
            page_id = page.page_id

            if page_id is None:
                continue

            if page_id in page_scores:
                page_scores[page_id] += page.score
            else:
                page_scores[page_id] = page.score

    # Sort by high scores and keep only max_n
    high_scores = sorted(page_scores.values(), reverse=True)[:max_n]

    return list(set(page_id for page_id in page_scores if page_scores[page_id] in high_scores))


def wikisearch(keyword_list):
    """
    Searches Wikipedia API for all keywords in the list, sequentially or in parallel.

    Args:
        keyword_list (list of str): List of keywords to perform the wikisearch.

    Returns:
        list of WikisearchResult: List of wikisearch results.
    """

    # Clean all keywords in keyword_list
    keyword_list = clean(keyword_list)

    # Execute wikisearch in parallel
    results = [actors[i % n_actors].wikisearch.remote(keyword_list[i]) for i in range(len(keyword_list))]

    # Wait for the results
    results = ray.get(results)

    # # Modify results to include page ids instead of titles
    # results = postprocess(results, page_title_ids)

    return results
