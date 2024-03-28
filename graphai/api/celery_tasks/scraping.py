from celery import shared_task
from graphai.core.common.scraping import (
    initialize_url,
    get_sublinks,
    process_all_sublinks,
    remove_headers,
    remove_long_patterns,
    reconstruct_data_dict
)
from graphai.core.common.common_utils import get_current_datetime
from graphai.core.common.caching import ScrapingDBCachingManager


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='caching_6.cache_lookup_get_sublinks', ignore_result=False)
def cache_lookup_get_sublinks_task(self, token):
    db_manager = ScrapingDBCachingManager()
    existing = db_manager.get_details_using_origin(token, ['link'])
    if existing is not None:
        sublinks = [r['link'] for r in existing]
        tokens = [r['id_token'] for r in existing]
        # There is always ONE row where the id_token and origin_token are the same. This is the sublink whose
        # url is equal to the validated_url.
        validated_url = [x for x in existing if x['id_token'] == x['origin_token']][0]['link']
        data_dict = reconstruct_data_dict(sublinks, tokens)
        if len(data_dict) == 0:
            data_dict = None
        return {
            'token': token,
            'sublinks': sublinks,
            'data': data_dict,
            'validated_url': validated_url,
            'status_msg': "",
            'fresh': False,
            'successful': True
        }
    return None


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='scraping_6.initialize_scraping_url', ignore_result=False)
def initialize_url_and_get_sublinks_task(self, token, url):
    if token is None or url is None:
        return {
            'token': None,
            'sublinks': None,
            'data': None,
            'validated_url': None,
            'status_msg': "Error: no input provided",
            'fresh': False,
            'successful': False
        }
    # Initializing the URL by figuring out the exact correct URL and retrieving it to make sure it's accessible
    # validated_url will be None if the URL is inaccessible
    base_url, validated_url, status_msg, status_code = initialize_url(url, base_url=token)
    sublinks, data, validated_url = get_sublinks(token, validated_url)
    fresh = sublinks is not None
    return {
        'token': token if sublinks is not None else None,
        'sublinks': sublinks,
        'data': data,
        'validated_url': validated_url,
        'status_msg': status_msg,
        'fresh': fresh,
        'successful': fresh
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='scraping_6.scraping_sublinks_callback', ignore_result=False)
def scraping_sublinks_callback_task(self, results):
    if results['data'] is not None:
        db_manager = ScrapingDBCachingManager()
        # First we double-check to see if the results already exist in the cache, and if so, delete them.
        # This is because of the one-to-many relationship between a URL and its sublink cache rows.
        existing = db_manager.get_details_using_origin(results['token'], [])
        if existing is not None:
            db_manager.delete_cache_rows([x['id_token'] for x in existing])
        # Now we can safely insert the new results
        current_datetime = get_current_datetime()
        data = results['data']
        for sublink in data:
            db_manager.insert_or_update_details(data[sublink]['id'], values_to_insert={
                'origin_token': results['token'],
                'link': sublink,
                'date_added': current_datetime
            })
    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='caching_6.cache_lookup_process_all_sublinks', ignore_result=False)
def cache_lookup_process_all_sublinks_task(self, token, headers, long_patterns):
    db_manager = ScrapingDBCachingManager()
    existing = db_manager.get_details_using_origin(token, ['link', 'content', 'page_type',
                                                           'headers_removed', 'long_patterns_removed'])
    if existing is not None:
        existing_headers = existing[0]['headers_removed'] == 1
        existing_long_patterns = existing[0]['long_patterns_removed'] == 1
        headers_match = existing_headers == headers
        long_patterns_match = existing_long_patterns == long_patterns
        base_url_content = [x for x in existing if x['id_token'] == x['origin_token']][0]['content']
        # We only return the cached results if their header and pattern removal flags are the same as the request
        # and if the cached results are not null
        if base_url_content is not None and headers_match and long_patterns_match:
            sublinks = [r['link'] for r in existing]
            tokens = [r['id_token'] for r in existing]
            contents = [r['content'] for r in existing]
            page_types = [r['page_type'] for r in existing]
            data_dict = reconstruct_data_dict(sublinks, tokens, contents, page_types)
            return data_dict
    return None


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='scraping_6.process_all_scraping_sublinks_parallel', ignore_result=False)
def process_all_scraping_sublinks_parallel_task(self, results, i, n_total):
    if results['data'] is None:
        return results
    data = results['data']
    sublinks = results['sublinks']
    base_url = results['token']
    validated_url = results['validated_url']
    start_index = int(i * len(sublinks) / n_total)
    end_index = int((i + 1) * len(sublinks) / n_total)
    sublink_sublist = sublinks[start_index:end_index]
    data = {k: data[k] for k in sublink_sublist}
    data = process_all_sublinks(data, base_url, validated_url)
    return {
        'token': results['token'],
        'sublinks': results['sublinks'],
        'data': data,
        'validated_url': results['validated_url'],
        'status_msg': results['status_msg']
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='scraping_6.sublink_parallel_processing_merge_callback', ignore_result=False)
def sublink_parallel_processing_merge_callback_task(self, results):
    # If the results are null (e.g. unreachable url)
    if results[0]['data'] is None:
        return {
            'token': None,
            'sublinks': None,
            'validated_url': None,
            'status_msg': results[0]['status_msg'],
            'data': None,
            'fresh': False,
            'successful': False
        }
    # Merging fresh results
    joint_data = dict()
    for r in results:
        joint_data.update(r['data'])
    return {
        'token': results[0]['token'],
        'sublinks': results[0]['sublinks'],
        'validated_url': results[0]['validated_url'],
        'status_msg': results[0]['status_msg'],
        'data': joint_data,
        'fresh': True,
        'successful': True
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='scraping_6.remove_junk_scraping_parallel', ignore_result=False)
def remove_junk_scraping_parallel_task(self, results, i, n_total, headers, long_patterns):
    # If the results are null, again there's nothing to do and no merging
    if results['data'] is None:
        return results
    data = results['data']
    sublinks = results['sublinks']
    start_index = int(i * len(sublinks) / n_total)
    end_index = int((i + 1) * len(sublinks) / n_total)
    sublink_sublist = sublinks[start_index:end_index]
    data = {k: data[k] for k in sublink_sublist}
    # If headers and long_patterns are both False, nothing happens, we just pass on the data slice.
    if headers:
        data = remove_headers(data)
    if long_patterns:
        data = remove_long_patterns(data)
    return {
        'token': results['token'],
        'sublinks': results['sublinks'],
        'data': data,
        'validated_url': results['validated_url'],
        'status_msg': results['status_msg'],
        'fresh': True,
        'successful': True
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='scraping_6.extract_scraping_content_callback', ignore_result=False)
def extract_scraping_content_callback_task(self, results, headers, long_patterns):
    if results['data'] is not None:
        current_datetime = get_current_datetime()
        db_manager = ScrapingDBCachingManager()
        for sublink in results['data']:
            db_manager.insert_or_update_details(results['data'][sublink]['id'], values_to_insert={
                'origin_token': results['token'],
                'link': sublink,
                'content': results['data'][sublink]['content'],
                'page_type': results['data'][sublink]['pagetype'],
                'headers_removed': 1 if headers else 0,
                'long_patterns_removed': 1 if long_patterns else 0,
                'date_added': current_datetime
            })
    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='scraping_6.scraping_dummy_task', ignore_result=False)
def scraping_dummy_task(self, results):
    # This task is required for chaining groups due to the peculiarities of celery
    # Whenever there are two groups in one chain of tasks, there need to be at least
    # TWO tasks between them, and this dummy task is simply an f(x)=x function.
    return results
