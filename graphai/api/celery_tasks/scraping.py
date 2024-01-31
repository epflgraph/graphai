from celery import shared_task
from graphai.core.common.scraping import initialize_url, get_sublinks, process_all_sublinks, \
    remove_headers, remove_long_patterns, reconstruct_data_dict
from graphai.core.common.common_utils import get_current_datetime
from graphai.core.common.caching import ScrapingDBCachingManager


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='scraping_6.initialize_scraping_url', ignore_result=False)
def initialize_url_and_get_sublinks_task(self, token, url, force=False):
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
    if not force:
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
    if results['data'] is not None and results['fresh']:
        current_datetime = get_current_datetime()
        db_manager = ScrapingDBCachingManager()
        data = results['data']
        for sublink in data:
            db_manager.insert_or_update_details(data[sublink]['id'], values_to_insert={
                'origin_token': results['token'],
                'link': sublink,
                'date_added': current_datetime
            })
    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='scraping_6.process_all_scraping_sublinks_preprocess', ignore_result=False)
def process_all_scraping_sublinks_preprocess_task(self, results, headers, long_patterns, force=False):
    results['cached_results'] = None
    # If there are no results to begin with, we pass them on as-is
    if results['token'] is None or results['data'] is None:
        return results
    # Now we check to see if there are any cached results, and if so, whether they conform to the request that was made
    if not force and not results['fresh']:
        token = results['token']
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
                results['cached_results'] = data_dict
                return results
    # If the results are to be retrieved from scratch, pass them on with no cached results
    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='scraping_6.process_all_scraping_sublinks_parallel', ignore_result=False)
def process_all_scraping_sublinks_parallel_task(self, results, i, n_total):
    if results['data'] is None or results.get('cached_results', None) is not None:
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
        'status_msg': results['status_msg'],
        'cached_results': None
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='scraping_6.process_all_scraping_sublinks_callback', ignore_result=False)
def process_all_scraping_sublinks_callback_task(self, results):
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
    # If the results came from a cache hit
    if results[0]['cached_results'] is not None:
        return {
            'token': results[0]['token'],
            'sublinks': results[0]['sublinks'],
            'validated_url': results[0]['validated_url'],
            'status_msg': results[0]['status_msg'],
            'data': results[0]['cached_results'],
            'fresh': False,
            'successful': True
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
    # If there is no junk removal, there's nothing to merge in the callback and nothing to do here
    if not headers and not long_patterns:
        results['do_merge'] = False
        return results
    # If the results are null or the results are from the cache, again there's nothing to do and no merging
    if results['data'] is None or not results['fresh']:
        results['do_merge'] = False
        return results
    data = results['data']
    sublinks = results['sublinks']
    start_index = int(i * len(sublinks) / n_total)
    end_index = int((i + 1) * len(sublinks) / n_total)
    sublink_sublist = sublinks[start_index:end_index]
    data = {k: data[k] for k in sublink_sublist}
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
        'successful': True,
        'do_merge': True
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='scraping_6.remove_junk_scraping_callback', ignore_result=False)
def remove_junk_scraping_callback_task(self, results):
    # If the results aren't fresh (whether cache hit or unsuccessful), there is nothing to merge and nothing to
    # insert into the database
    if not results[0]['fresh']:
        del results[0]['do_merge']
        return results[0]
    if results[0]['do_merge']:
        # If 'do_merge' is True, the full data is the combination of all 'data' values and needs merging
        joint_data = dict()
        for r in results:
            joint_data.update(r['data'])
    else:
        # If 'do_merge' is False (but 'fresh' is True, since we're here!), then all 'data' values are identical
        joint_data = results[0]['data']

    new_results = results[0]
    new_results['data'] = joint_data
    return new_results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='scraping_6.extract_scraping_content_callback', ignore_result=False)
def extract_scraping_content_callback_task(self, results, headers, long_patterns):
    if results['fresh']:
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
