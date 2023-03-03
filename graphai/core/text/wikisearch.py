import pandas as pd
import ray

from graphai.core.interfaces import WP
from graphai.core.interfaces import ES


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
            method (str): Method to retrieve the wikipedia pages. It can be either "wikipedia-api", to use the
            Wikipedia API, or one of {"es-base", "es-score"}, to use elasticsearch, returning as score the inverse
            of the searchrank or the actual elasticsearch score, respectively. Default: 'es-base'.
            Fallback: 'wikipedia-api'.

        Returns:
            pd.DataFrame: A pandas DataFrame with columns ['Keywords', 'PageID', 'PageTitle', 'Searchrank', 'SearchScore'],
            constant on 'Keywords' and unique in 'PageID', with the wikisearch results the given keywords set.
        """

        # Request results
        if method == 'wikipedia-api':
            results = self.wp.search(keywords)
        else:
            results = self.es.search(keywords)

            # Fallback to Wikipedia API
            if len(results) == 0:
                results = self.wp.search(keywords)

        # Replace score with linear function on Searchrank if needed
        if method != 'es-score':
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
            It can be either 'wikipedia-api', to use the Wikipedia API, or one of {'es-base', 'es-score'},
            to use elasticsearch, returning as score the inverse of the searchrank or the actual elasticsearch score,
            respectively. Default: 'es-base'.

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
