import json
import requests
from ssl import create_default_context

import pandas as pd
import urllib3

from elasticsearch_interface.utils import (
    bool_query,
    match_query,
    multi_match_query,
    dis_max_query,
)

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


class ElasticsearchSearchError(RuntimeError):
    def __init__(self, message, api_status_code=503, upstream_status=None):
        super().__init__(message)
        self.api_status_code = int(api_status_code)
        self.upstream_status = upstream_status


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


def _es_base_url(es_config):
    return f"https://{es_config['host']}:{es_config['port']}"


def _es_headers(es_config):
    return {
        **urllib3.make_headers(basic_auth=f"{es_config['username']}:{es_config['password']}"),
        'Content-Type': 'application/json',
    }


def _es_http_pool(es_config):
    return urllib3.PoolManager(
        num_pools=1,
        maxsize=1,
        block=True,
        ssl_context=create_default_context(cafile=es_config['cafile']),
    )


def _extract_es_error_reason(payload):
    if not isinstance(payload, dict):
        return None

    error = payload.get('error')
    if isinstance(error, dict):
        reason = error.get('reason')
        if reason:
            return str(reason)

        root_cause = error.get('root_cause')
        if isinstance(root_cause, list) and root_cause:
            first_cause = root_cause[0]
            if isinstance(first_cause, dict) and first_cause.get('reason'):
                return str(first_cause['reason'])
    elif error:
        return str(error)

    return None


def _build_es_http_error(response, es_config, index, operation):
    body_text = response.data.decode('utf-8', errors='replace')
    payload = None
    try:
        payload = json.loads(body_text)
    except json.JSONDecodeError:
        payload = None

    reason = _extract_es_error_reason(payload)
    target = f"{_es_base_url(es_config)}/{index}"

    if response.status == 404:
        message = f'Elasticsearch index "{index}" is unavailable for {operation} on {target}.'
        if reason:
            message = f'{message} {reason}'
        return ElasticsearchSearchError(message, api_status_code=503, upstream_status=response.status)

    message = f'Elasticsearch {operation} failed on {target} with HTTP {response.status}.'
    if reason:
        message = f'{message} {reason}'
    return ElasticsearchSearchError(message, api_status_code=503, upstream_status=response.status)


def validate_elasticsearch_index_http(es_config, index, timeout=DEFAULT_ES_TIMEOUT):
    timeout = _safe_timeout(timeout, DEFAULT_ES_TIMEOUT)
    http = _es_http_pool(es_config)

    try:
        response = http.request(
            'GET',
            f"{_es_base_url(es_config)}/{index}/_count",
            headers=_es_headers(es_config),
            timeout=urllib3.Timeout(connect=timeout, read=timeout),
            retries=False,
        )
        if response.status >= 400:
            raise _build_es_http_error(response, es_config, index, operation='index validation')

        payload = json.loads(response.data.decode('utf-8'))
        return int(payload.get('count', 0))
    except urllib3.exceptions.TimeoutError as exc:
        raise ElasticsearchSearchError(
            f'Elasticsearch index validation timed out for "{index}" on {_es_base_url(es_config)}: {exc}',
            api_status_code=503,
        ) from exc
    except urllib3.exceptions.HTTPError as exc:
        raise ElasticsearchSearchError(
            f'Elasticsearch index validation failed for "{index}" on {_es_base_url(es_config)}: {type(exc).__name__}: {exc}',
            api_status_code=503,
        ) from exc
    except json.JSONDecodeError as exc:
        raise ElasticsearchSearchError(
            f'Elasticsearch index validation returned invalid JSON for "{index}" on {_es_base_url(es_config)}.',
            api_status_code=503,
        ) from exc


def _build_wikipedia_search_query(text):
    return bool_query(
        should=[
            multi_match_query(fields=['all_near_match^10', 'all_near_match_asciifolding^7.5'], text=text),
            bool_query(
                filter=[
                    bool_query(
                        should=[
                            match_query('all', text=text, operator='and'),
                            match_query('all.plain', text=text, operator='and'),
                        ]
                    )
                ],
                should=[
                    multi_match_query(fields=['title^3', 'title.plain^1'], text=text, type='most_fields', boost=0.3, minimum_should_match=1),
                    multi_match_query(fields=['category^3', 'category.plain^1'], text=text, type='most_fields', boost=0.05, minimum_should_match=1),
                    multi_match_query(fields=['heading^3', 'heading.plain^1'], text=text, type='most_fields', boost=0.05, minimum_should_match=1),
                    multi_match_query(fields=['auxiliary_text^3', 'auxiliary_text.plain^1'], text=text, type='most_fields', boost=0.05, minimum_should_match=1),
                    multi_match_query(fields=['file_text^3', 'file_text.plain^1'], text=text, type='most_fields', boost=0.5, minimum_should_match=1),
                    dis_max_query([
                        multi_match_query(fields=['redirect^3', 'redirect.plain^1'], text=text, type='most_fields', boost=0.27, minimum_should_match=1),
                        multi_match_query(fields=['suggest'], text=text, type='most_fields', boost=0.2, minimum_should_match=1),
                    ]),
                    dis_max_query([
                        multi_match_query(fields=['text^3', 'text.plain^1'], text=text, type='most_fields', boost=0.6, minimum_should_match=1),
                        multi_match_query(fields=['opening_text^3', 'opening_text.plain^1'], text=text, type='most_fields', boost=0.5, minimum_should_match=1),
                    ]),
                ],
            ),
        ]
    )


def search_elasticsearch_http(
    text,
    es_config,
    index,
    limit=10,
    timeout=DEFAULT_ES_TIMEOUT,
    timeout_retries=DEFAULT_ES_TIMEOUT_RETRIES,
):
    sysmsg.info(f'⚡️ Searching Elasticsearch over HTTP for text: "{text}" with limit {limit} and timeout {timeout}s.')

    timeout = _safe_timeout(timeout, DEFAULT_ES_TIMEOUT)
    timeout_retries = _safe_positive_int(timeout_retries, DEFAULT_ES_TIMEOUT_RETRIES)
    url = f"{_es_base_url(es_config)}/{index}/_search"
    body = {
        'query': _build_wikipedia_search_query(text),
        'size': limit,
        'profile': True,
    }
    request_kwargs = {
        'body': json.dumps(body).encode('utf-8'),
        'headers': _es_headers(es_config),
        'timeout': urllib3.Timeout(connect=timeout, read=timeout),
        'retries': False,
    }
    http = _es_http_pool(es_config)

    for attempt in range(1, timeout_retries + 1):
        try:
            response = http.request('POST', url, **request_kwargs)
            if response.status >= 400:
                error = _build_es_http_error(response, es_config, index, operation='search')
                sysmsg.critical(str(error))
                raise error

            hits = json.loads(response.data.decode('utf-8')).get('hits', {}).get('hits', [])
            sysmsg.success(f'✅ Elasticsearch HTTP search returned {len(hits)} hits for text: "{text}".')
            return [
                {
                    'concept_id': hit['_source']['id'],
                    'concept_name': hit['_source']['title'],
                    'score': hit['_score'],
                }
                for hit in hits
            ]
        except urllib3.exceptions.TimeoutError as exc:
            if attempt < timeout_retries:
                sysmsg.warning(
                    f'⚠️ Elasticsearch HTTP timeout for text "{text}" (attempt {attempt}/{timeout_retries}): {exc}. Retrying...'
                )
                continue
            sysmsg.warning(
                f'⚠️ Elasticsearch HTTP timeout for text "{text}" after {attempt} attempt(s): {exc}'
            )
            raise ElasticsearchSearchError(
                f'Elasticsearch HTTP search timed out for text "{text}" on {_es_base_url(es_config)}/{index}: {exc}',
                api_status_code=503,
            ) from exc
        except urllib3.exceptions.HTTPError as exc:
            sysmsg.critical(f'Elasticsearch HTTP request failed for text "{text}": {type(exc).__name__}: {exc}')
            raise ElasticsearchSearchError(
                f'Elasticsearch HTTP request failed for text "{text}" on {_es_base_url(es_config)}/{index}: '
                f'{type(exc).__name__}: {exc}',
                api_status_code=503,
            ) from exc
        except json.JSONDecodeError as exc:
            raise ElasticsearchSearchError(
                f'Elasticsearch HTTP search returned invalid JSON for text "{text}" on {_es_base_url(es_config)}/{index}.',
                api_status_code=503,
            ) from exc

    raise ElasticsearchSearchError(
        f'Elasticsearch HTTP search exhausted retries for text "{text}" on {_es_base_url(es_config)}/{index}.',
        api_status_code=503,
    )


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
                    raise ElasticsearchSearchError(
                        f'Elasticsearch search returned no payload for text "{text}" on index "{getattr(es, "index", "<unknown>")}".',
                        api_status_code=503,
                    )

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
                    raise ElasticsearchSearchError(
                        f'Elasticsearch search timed out for text "{text}" on index "{getattr(es, "index", "<unknown>")}": '
                        f'{error_kind}: {error_message}',
                        api_status_code=503,
                    ) from exc
                if network_error:
                    sysmsg.critical(
                        f'Elasticsearch network error for text "{text}": {error_kind}: {error_message}'
                    )
                    raise ElasticsearchSearchError(
                        f'Elasticsearch network error for text "{text}" on index "{getattr(es, "index", "<unknown>")}": '
                        f'{error_kind}: {error_message}',
                        api_status_code=503,
                    ) from exc

                sysmsg.critical(
                    f'Elasticsearch request failed for text "{text}": {error_kind}: {error_message}'
                )
                raise ElasticsearchSearchError(
                    f'Elasticsearch request failed for text "{text}" on index "{getattr(es, "index", "<unknown>")}": '
                    f'{error_kind}: {error_message}',
                    api_status_code=503,
                ) from exc
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
