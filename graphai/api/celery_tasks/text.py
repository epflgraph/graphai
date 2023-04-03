import pandas as pd

from celery import shared_task, group

from graphai.core.interfaces.wp import WP
from graphai.core.interfaces.es import ES


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={'max_retries': 2},
             name='text.wikisearch', ignore_result=False, wp=WP(), es=ES('concepts'))
def wikisearch_task(self, keywords, method):
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


def wikisearch_master(keywords, method):
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

    # Set up job that runs in parallel a wikisearch task for each set of keywords
    job = group(wikisearch_task.s(row['Keywords'], method) for i, row in keywords.iterrows())
    results = job.apply_async(priority=10)

    # Wait for results
    results = results.get(timeout=10)

    # Concatenate all results in a single DataFrame
    results = pd.concat(results, ignore_index=True)

    return results
