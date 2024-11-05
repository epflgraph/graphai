import requests

import pandas as pd


def search_wikipedia_api(text, limit=10):
    """
    Perform search query to Wikipedia API for a given text.

    Args:
        text (str): Query text for the search.
        limit (int): Maximum number of returned results.

    Returns:
        list: A list of dictionaries with keys 'concept_id' and 'concept_name' containing the top matches for the search.
    """

    params = {
        'format': 'json',
        'action': 'query',
        'list': 'search',
        'srsearch': text,
        'srlimit': limit,
        'srprop': ''
    }
    headers = {'User-Agent': 'graphai (https://github.com/epflgraph/graphai)'}
    url = 'http://en.wikipedia.org/w/api.php'

    try:
        # Make request
        response = requests.get(url, params=params, headers=headers, timeout=6).json()

        # Extract list of results
        hits = response['query']['search']

        # Return as list of dictionaries with keys 'concept_id' and 'concept_name'
        return [{'concept_id': hit['pageid'], 'concept_name': hit['title']} for hit in hits]

    except Exception:
        # If something goes wrong, avoid crashing and return empty list
        print('[ERROR] Error connecting to Wikipedia API.')
        return []


def search_elasticsearch(text, es, limit=10):
    """
    Perform search query to elasticserch cluster for a given text.

    Args:
        text (str): Query text for the search.
        es (ES): Elasticsearch interface.
        limit (int): Maximum number of returned results.

    Returns:
        list: A list of dictionaries with keys 'concept_id', 'concept_name' and 'score' containing the top matches for the search.
    """

    try:
        # Send search request
        hits = es.search(text, limit)

        # Return as list of dictionaries with keys 'concept_id', 'concept_name' and 'score'
        return [{'concept_id': hit['_source']['id'], 'concept_name': hit['_source']['title'], 'score': hit['_score']} for hit in hits]

    except Exception:
        # If something goes wrong, avoid crashing and return empty list
        print('[ERROR] Error connecting to elasticsearch cluster.')
        return []


def wikisearch(keywords_list, es, fraction=(0, 1), method='es-base'):
    """
    Finds 10 relevant concepts (Wikipedia pages) for each set of keywords in a list.

    Args:
        keywords_list (list(str)): List containing the sets of keywords for which to search concepts.
        es (ES): Elasticsearch interface.
        fraction (tuple(int, int)): Portion of the keywords_list to be processed, e.g. (1/3, 2/3) means only
        the middle third of the list is considered.
        method (str): Method to retrieve the concepts (Wikipedia pages). It can be either "wikipedia-api", to use the
        Wikipedia API, or one of {"es-base", "es-score"}, to use elasticsearch.

    Returns:
        pd.DataFrame: A pandas DataFrame with columns ['keywords', 'concept_id', 'concept_name', 'searchrank', 'search_score'],
        unique by ('keywords', 'concept_id'). The searchrank is the position of the concept in the list of results for that set of keywords,
        starting with 1. The search score is the elasticsearch score for method "es-score" or 1 - (searchrank - 1)/n
        for the other methods. Default: 'es-base'. Fallback: 'wikipedia-api'.
    """

    # Slice keywords_list
    begin = int(fraction[0] * len(keywords_list))
    end = int(fraction[1] * len(keywords_list))
    keywords_list = keywords_list[begin:end]

    # Iterate over all keyword sets and request the results
    all_results = pd.DataFrame()
    for keywords in keywords_list:
        if method == 'wikipedia-api':
            results_list = search_wikipedia_api(keywords)
        else:
            results_list = search_elasticsearch(keywords, es)

            # Fallback to Wikipedia API if no results from elasticsearch
            if not results_list:
                print(f'[WARNING] No results from elasticsearch cluster for keywords {keywords}. Falling back to Wikipedia API.')
                results_list = search_wikipedia_api(keywords)

        # Ignore set of keywords if no pages are found
        if not results_list:
            continue

        # Build results DataFrame
        results = pd.DataFrame(
            [
                [keywords, result['concept_id'], result['concept_name'], i + 1, result.get('score', 1)]
                for i, result in enumerate(results_list)
            ],
            columns=['keywords', 'concept_id', 'concept_name', 'searchrank', 'search_score'],
        )

        # Replace search score with linear function on searchrank if needed
        if method != 'es-score':
            results['search_score'] = 1 - (results['searchrank'] - 1) / len(results)

        # Append results
        all_results = pd.concat([all_results, results], ignore_index=True)

    return all_results


if __name__ == '__main__':
    from elasticsearch_interface.es import ES

    from graphai.core.common.config import config

    es = ES(config['elasticsearch'], index=config['elasticsearch'].get('concept_detection_index', 'concepts_detection'))

    results = wikisearch(['Cayley graph', 'Lebesgue measure', 'graph spectra', 'spectral gap'], es)
    print(results)
