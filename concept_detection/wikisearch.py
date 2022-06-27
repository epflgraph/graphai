import ray

from models.wikisearch_result import WikisearchResult

from interfaces.wp import WP
from interfaces.es import ES

API_URL = 'http://en.wikipedia.org/w/api.php'
HEADERS = {'User-Agent': 'graphai (https://github.com/epflgraph/graphai)'}

# Init ray
ray.init(namespace="wikisearch", include_dashboard=False, log_to_driver=True)


@ray.remote
class WikisearchActor:
    """
    Class representing a ray Actor to perform wikisearch in parallel.
    """
    def __init__(self):
        # Instantiate wikipedia-api and elasticsearch interfaces
        self.wp = WP()
        self.es = ES('wikipages')

    def wikisearch(self, keywords, method):
        """
        Returns top 10 results for Wikipedia pages relevant to the keywords.

        Args:
            keywords (str): Text to search for among Wikipedia pages.
            method (str): Method to retrieve the wikipedia pages. It can be either "wikipedia-api",
            to use the Wikipedia API (default), or one of {"es-base", "es-score"}, to use elasticsearch,
            returning as score the inverse of the searchrank or the actual elasticsearch score, respectively.

        Returns:
            :class:`~models.wikisearch_result.WikisearchResult`: Object containing the given keywords and their associated list of page results.
        """

        if method == 'es-base':
            pages = self.es.search(keywords)

            # Replace score with 1/searchrank
            for page in pages:
                page.score = 1 / page.searchrank
        elif method == 'es-score':
            pages = self.es.search(keywords)
        else:
            pages = self.wp.search(keywords)

        return WikisearchResult(keywords=keywords, pages=pages)


# Instantiate ray actor list
n_actors = 16
actors = [WikisearchActor.remote() for i in range(n_actors)]


def wikisearch(keyword_list, method):
    """
    Retrieves wikipages for all keywords in the list.

    Args:
        keyword_list (list[str]): List of keywords to perform the wikisearch.
        method (str{'wikipedia-api', 'es-base', 'es-score'}): Method to retrieve the wikipedia pages.
            It can be either 'wikipedia-api', to use the Wikipedia API (default), or one of {'es-base', 'es-score'},
            to use elasticsearch, returning as score the inverse of the searchrank or the actual elasticsearch score,
            respectively.

    Returns:
        list[:class:`~models.wikisearch_result.WikisearchResult`]: List of wikisearch results.
    """

    # Execute wikisearch in parallel
    results = [actors[i % n_actors].wikisearch.remote(keyword_list[i], method) for i in range(len(keyword_list))]

    # Wait for the results
    results = ray.get(results)

    return results


def extract_page_ids(results):
    """
    Iterates through the given wikisearch results and returns a list with all the page ids.

    Args:
        results (list[:class:`~models.wikisearch_result.WikisearchResult`]): List of wikisearch results.

    Returns:
        list[int]: List of all page ids present along all results.
    """
    return list(set(page.page_id for result in results for page in result.pages))


def extract_anchor_page_ids(results, max_n=3):
    """
    Iterates through the given wikisearch results and returns a list with the most relevant page ids.

    Args:
        results (list[:class:`~models.wikisearch_result.WikisearchResult`]): List of wikisearch results.
        max_n (int): Maximum number of page ids returned. Default: 3.

    Returns:
        list[int]: List of the most relevant page ids present along all results.

    Examples:
        Examples should be written in doctest format, and should illustrate how
        to use the function.

        >>> print([i for i in example_generator(4)])
        [0, 1, 2, 3]
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

