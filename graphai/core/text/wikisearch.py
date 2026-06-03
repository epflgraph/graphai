import requests

import pandas as pd

try:
    from elastic_transport import ConnectionError as ESConnectionError
    from elastic_transport import ConnectionTimeout as ESConnectionTimeout
except Exception:  # pragma: no cover - keep runtime resilient if transport internals change
    ESConnectionError = ()
    ESConnectionTimeout = ()


#----------------------#
# Set up sysmsg logger #
#----------------------#
from loguru import logger as sysmsg
import sys
sysmsg.remove()
sysmsg.add(
    sys.stdout,
    level="TRACE",
    colorize=True,      # FORCE ANSI colors
    enqueue=True,       # REQUIRED for Celery / multiprocessing
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
           "{message}",
)
#----------------------#




DEFAULT_WIKIPEDIA_TIMEOUT = 6
DEFAULT_ES_TIMEOUT = 10
DEFAULT_ES_TIMEOUT_RETRIES = 2


def _safe_timeout(value, default):
    try:
        timeout = float(value)
        if timeout > 0:
            return timeout
    except (TypeError, ValueError):
        pass
    return float(default)


def _safe_positive_int(value, default):
    try:
        parsed = int(value)
        if parsed > 0:
            return parsed
    except (TypeError, ValueError):
        pass
    return int(default)


def _is_es_timeout_error(exc):
    if ESConnectionTimeout and isinstance(exc, ESConnectionTimeout):
        return True

    message = str(exc).lower()
    return 'timed out' in message or 'timeout' in message


def search_wikipedia_api(text, limit=10, timeout=DEFAULT_WIKIPEDIA_TIMEOUT):
    """
    Perform search query to Wikipedia API for a given text.

    Args:
        text (str): Query text for the search.
        limit (int): Maximum number of returned results.

    Returns:
        list: A list of dictionaries with keys 'concept_id' and 'concept_name' containing the top matches for the search.
    """
    sysmsg.debug(f'Searching Wikipedia API for text: "{text}" with limit {limit} and timeout {timeout}s.')
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
        response = requests.get(url, params=params, headers=headers, timeout=timeout).json()

        # Extract list of results
        hits = response['query']['search']

        sysmsg.debug(f'Wikipedia API returned {len(hits)} hits for text: "{text}".')

        # Return as list of dictionaries with keys 'concept_id' and 'concept_name'
        return [{'concept_id': hit['pageid'], 'concept_name': hit['title']} for hit in hits]

    except Exception:
        # If something goes wrong, avoid crashing and return empty list
        sysmsg.error('Error connecting to Wikipedia API.')
        return []


def search_elasticsearch(text, es, limit=10, timeout=DEFAULT_ES_TIMEOUT, timeout_retries=DEFAULT_ES_TIMEOUT_RETRIES):
    """
    Perform search query to elasticserch cluster for a given text.

    Args:
        text (str): Query text for the search.
        es (ESConceptDetection): Elasticsearch interface.
        limit (int): Maximum number of returned results.

    Returns:
        list: A list of dictionaries with keys 'concept_id', 'concept_name' and 'score' containing the top matches for the search.
    """
    
    
    sysmsg.info(f'⚡️ Searching Elasticsearch cluster for text: "{text}" with limit {limit} and timeout {timeout}s.')

    original_client = None
    timeout_retries = _safe_positive_int(timeout_retries, DEFAULT_ES_TIMEOUT_RETRIES)
    try:
        # Override ES timeout for this call only.
        original_client = getattr(es, 'client', None)
        if original_client is not None and hasattr(original_client, 'options'):
            es.client = original_client.options(request_timeout=timeout)

        for attempt in range(1, timeout_retries + 1):
            try:
                # Send search request
                hits = es.search(text, limit)

                if hits is None:
                    sysmsg.warning(f'⚠️ Elasticsearch search returned None for text: "{text}".')
                    return []

                sysmsg.success(f'✅ Elasticsearch search returned {len(hits)} hits for text: "{text}".')

                # Return as list of dictionaries with keys 'concept_id', 'concept_name' and 'score'
                return [{'concept_id': hit['_source']['id'], 'concept_name': hit['_source']['title'], 'score': hit['_score']} for hit in hits]
            except Exception as exc:
                error_kind = type(exc).__name__
                error_message = str(exc)
                timed_out = _is_es_timeout_error(exc)
                network_error = ESConnectionError and isinstance(exc, ESConnectionError)

                if timed_out and attempt < timeout_retries:
                    sysmsg.warning(
                        f'⚠️ Elasticsearch timeout for text "{text}" (attempt {attempt}/{timeout_retries}): '
                        f'{error_kind}: {error_message}. Retrying...'
                    )
                    continue

                if timed_out:
                    sysmsg.warning(
                        f'⚠️ Elasticsearch timeout for text "{text}" after {attempt} attempt(s): '
                        f'{error_kind}: {error_message}'
                    )
                elif network_error:
                    sysmsg.critical(
                        f'Elasticsearch network error for text "{text}": {error_kind}: {error_message}'
                    )
                else:
                    sysmsg.critical(
                        f'Elasticsearch request failed for text "{text}": {error_kind}: {error_message}'
                    )
                return []
    finally:
        # Restore original client to avoid side effects across calls.
        if original_client is not None:
            es.client = original_client
            sysmsg.debug('Restored original Elasticsearch client after search.')


def wikisearch(
    keywords_list,
    es,
    fraction=(0, 1),
    method='es-base',
    es_timeout=DEFAULT_ES_TIMEOUT,
    wikipedia_timeout=DEFAULT_WIKIPEDIA_TIMEOUT,
    es_timeout_retries=DEFAULT_ES_TIMEOUT_RETRIES,
):
    """
    Finds 10 relevant concepts (Wikipedia pages) for each set of keywords in a list.

    Args:
        keywords_list (list(str)): List containing the sets of keywords for which to search concepts.
        es (ESConceptDetection): Elasticsearch interface.
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
    sysmsg.info(f'Starting wikisearch with method "{method}" on {len(keywords_list)} sets of keywords, processing fraction {fraction}.')

    # Slice keywords_list
    begin = int(fraction[0] * len(keywords_list))
    end = int(fraction[1] * len(keywords_list))
    keywords_list = keywords_list[begin:end]

    # Normalise timeout values once per call.
    es_timeout = _safe_timeout(es_timeout, DEFAULT_ES_TIMEOUT)
    wikipedia_timeout = _safe_timeout(wikipedia_timeout, DEFAULT_WIKIPEDIA_TIMEOUT)
    es_timeout_retries = _safe_positive_int(es_timeout_retries, DEFAULT_ES_TIMEOUT_RETRIES)

    # Iterate over all keyword sets and request the results
    all_results = pd.DataFrame()
    for keywords in keywords_list:
        if method == 'wikipedia-api':
            sysmsg.warning(f'⚠️ Using Wikipedia API for keywords: "{keywords}".')
            results_list = search_wikipedia_api(keywords, timeout=wikipedia_timeout)
        else:
            sysmsg.debug(f'⚡️ Using Elasticsearch cluster for keywords: "{keywords}".')
            results_list = search_elasticsearch(
                keywords,
                es,
                timeout=es_timeout,
                timeout_retries=es_timeout_retries,
            )

            # Fallback to Wikipedia API if no results from elasticsearch
            if not results_list:
                sysmsg.warning(f'⚠️ No results from elasticsearch cluster for keywords {keywords}. Falling back to Wikipedia API.')
                results_list = search_wikipedia_api(keywords, timeout=wikipedia_timeout)
            else:
                sysmsg.success(f'✅ Found {len(results_list)} results for keywords: "{keywords}".')

        # Ignore set of keywords if no pages are found
        if not results_list:
            sysmsg.warning(f'⚠️ No results found for keywords: "{keywords}". Skipping.')
            continue

        # Build results DataFrame
        results = pd.DataFrame(
            [
                [keywords, result['concept_id'], result['concept_name'], i + 1, result.get('score', 1)]
                for i, result in enumerate(results_list)
            ],
            columns=['keywords', 'concept_id', 'concept_name', 'searchrank', 'search_score'],
        )

        sysmsg.debug(f'Constructed results DataFrame for keywords: "{keywords}". Sample:\n{results.head()}')

        # Replace search score with linear function on searchrank if needed
        if method != 'es-score':
            results['search_score'] = 1 - (results['searchrank'] - 1) / len(results)
            sysmsg.debug(f'Updated search scores based on search rank for keywords: "{keywords}". Sample:\n{results.head()}')

        # Append results
        all_results = pd.concat([all_results, results], ignore_index=True)

        sysmsg.debug(f'Appended results for keywords: "{keywords}". Total results so far: {len(all_results)}.')

    return all_results


if __name__ == '__main__':
    from elasticsearch_interface.es import ESConceptDetection

    from graphai.core.common.config import config

    es = ESConceptDetection(config['elasticsearch'],
                            index=config['elasticsearch'].get('concept_detection_index', 'concepts_detection'))

    results = wikisearch(['Cayley graph', 'Lebesgue measure', 'graph spectra', 'spectral gap'], es)
    print(results)
