import pandas as pd
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
        self.es = ES('concepts')

    def wikisearch(self, keywords, method):
        """
        Returns top 10 results for Wikipedia pages relevant to the keywords.

        Args:
            keywords (str): Text to search for among Wikipedia pages.
            method (str): Method to retrieve the wikipedia pages. It can be either "wikipedia-api",
            to use the Wikipedia API (default), or one of {"es-base", "es-score"}, to use elasticsearch,
            returning as score the inverse of the searchrank or the actual elasticsearch score, respectively.

        Returns:
            pd.DataFrame: A pandas DataFrame with columns ['Keywords', 'PageID', 'PageTitle', 'Searchrank', 'SearchScore'],
            constant on 'Keywords' and unique in 'PageID', with the wikisearch results the given keywords set.
        """

        if method == 'es-score':
            results = self.es.search(keywords)
        else:
            if method == 'es-base':
                results = self.es.search(keywords)
            else:
                results = self.wp.search(keywords)

            # Replace score with linear function on Searchrank
            if len(results) >= 2:
                results['SearchScore'] = 1 - (results['Searchrank'] - 1) / (len(results) - 1)
            else:
                results['SearchScore'] = 1

        # Add Keywords column at the beginning
        results['Keywords'] = keywords
        results = results[['Keywords', 'PageID', 'PageTitle', 'Searchrank', 'SearchScore']]

        return results


# Instantiate ray actor list
n_actors = 16
actors = [WikisearchActor.remote() for i in range(n_actors)]


def wikisearch(keywords, method):
    """
    Retrieves wikipages for all keywords in the list.

    Args:
        keywords (pd.DataFrame): A pandas DataFrame with one column 'Keywords'.
        method (str{'wikipedia-api', 'es-base', 'es-score'}): Method to retrieve the wikipedia pages.
            It can be either 'wikipedia-api', to use the Wikipedia API (default), or one of {'es-base', 'es-score'},
            to use elasticsearch, returning as score the inverse of the searchrank or the actual elasticsearch score,
            respectively.

    Returns:
        pd.DataFrame: A pandas DataFrame with columns ['Keywords', 'PageID', 'PageTitle', 'Searchrank', 'SearchScore'],
        unique on ('Keywords', 'PageID'), with the wikisearch results for each keywords set.
    Examples:
        >>> keywords = pd.DataFrame(pd.Series(['brown baggies', 'platform soles']), columns=['Keywords'])
        >>> wikisearch(keywords, method='wikipedia-api')
    """

    # Execute wikisearch in parallel
    results = [actors[i % n_actors].wikisearch.remote(row['Keywords'], method) for i, row in keywords.iterrows()]

    # Wait for the results
    results = ray.get(results)

    # Concatenate all results in a single DataFrame
    results = pd.concat(results, ignore_index=True)

    return results


def extract_page_ids(results):
    """
    Iterates through the given wikisearch results and returns a list with all the page ids.

    Args:
        results (list[:class:`~models.wikisearch_result.WikisearchResult`]): List of wikisearch results.

    Returns:
        list[int]: List of all page ids present along all results.
    Examples:
        >>> extract_page_ids(wikisearch(['brown baggies', 'platform soles'], method='wikipedia-api'))
        [33921, 64498690, ..., 5859444, 50334463]
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
        >>> extract_anchor_page_ids(wikisearch(['brown baggies', 'platform soles'], method='wikipedia-api'))
        [33921, 699690, 5859444]
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

    return list(set(page_id for page_id in page_scores if page_scores[page_id] in high_scores))[:max_n]

