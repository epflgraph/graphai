from celery import shared_task
from graphai.core.common.scraping import initialize_url, get_sublinks, process_all_sublinks, \
    remove_headers, remove_long_patterns


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.initialize_scraping_url', ignore_result=False)
def initialize_scraping_url_task(self, token, url, force=False):
    if not force:

        pass
    # Initializing the URL by figuring out the exact correct URL and retrieving it to make sure it's accessible
    # validated_url will be None if the URL is inaccessible
    base_url, validated_url, status_msg, status_code = initialize_url(url, base_url=token)
    validation_results = {
        'token': token,
        'validated_url': validated_url,
        'status_msg': status_msg,
        'sublink_results': None,
        'fresh': True
    }
    return validation_results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.get_scraping_sublinks', ignore_result=False)
def get_scraping_sublinks_task(self, results):
    # This condition is only met if force=False and there is a cache hit
    if results['sublink_results'] is not None:
        sublinks = results['sublink_results']['sublinks']
        data = results['sublink_results']['data']
        validated_url = results['sublink_results']['data']
        fresh = False
    else:
        sublinks, data, validated_url = get_sublinks(results.get('validated_url', None))
        fresh = sublinks is not None
    return {
        'token': results['token'] if sublinks is not None else None,
        'sublinks': sublinks,
        'data': data,
        'validated_url': validated_url,
        'status_msg': results['status_msg'],
        'fresh': fresh,
        'successful': sublinks is not None
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.scraping_sublinks_callback', ignore_result=False)
def scraping_sublinks_callback_task(self, results, token):
    if results['token'] is not None and len(results['sublinks']) > 0:
        # TODO do the db callback
        pass
    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.process_all_scraping_sublinks_preprocess', ignore_result=False)
def process_all_scraping_sublinks_preprocess_task(self, results, token, force=False):
    # TODO do db lookup and return cached results if they exist
    #  otherwise clean up the results dict and pass it on
    #  The cached results should be in a `cached_results` attr
    #  Also make sure sublinks are sorted

    del results['successful']
    del results['fresh']
    results['cached_results'] = None
    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.process_all_scraping_sublinks_parallel', ignore_result=False)
def process_all_scraping_sublinks_parallel_task(self, results, i, n_total):
    if results['data'] is None or len(results['data']) == 0 or results['cached_results'] is not None:
        return results
    data = results['data']
    sublinks = results['sublinks']
    base_url = results['token']
    validated_url = results['validated_url']
    start_index = int(i * len(sublinks) / n_total)
    end_index = int((i + 1) * len(sublinks) / n_total)
    sublink_sublist = sublinks[start_index:end_index]
    # TODO handle cache hits by finding the ones that exist in the cache, retrieving their results, and
    #  keeping them in a separate dict and then deleting their sublinks from the dict that is given as arg
    #  to processing function.
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
             name='text_6.process_all_scraping_sublinks_callback', ignore_result=False)
def process_all_scraping_sublinks_callback_task(self, results):
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
             name='text_6.remove_junk_scraping_parallel', ignore_result=False)
def remove_junk_scraping_parallel_task(self, results, i, n_total, headers=True, long_patterns=True):
    if not headers and not long_patterns:
        results['do_merge'] = False
        return results
    if results['data'] is None or len(results['data']) == 0 or not results['fresh']:
        results['do_merge'] = False
        return results
    data = results['data']
    sublinks = results['sublinks']
    start_index = int(i * len(sublinks) / n_total)
    end_index = int((i + 1) * len(sublinks) / n_total)
    sublink_sublist = sublinks[start_index:end_index]
    # TODO handle cache hits by finding the ones that exist in the cache, retrieving their results, and
    #  keeping them in a separate dict and then deleting their sublinks from the dict that is given as arg
    #  to processing function.
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
             name='text_6.extract_scraping_content_callback', ignore_result=False)
def extract_scraping_content_callback_task(self, results):
    if not results[0]['do_merge']:
        del results[0]['do_merge']
        return results[0]
    # TODO do the database callback
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
