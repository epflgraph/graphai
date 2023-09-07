from celery import shared_task
from graphai.core.common.scraping import initialize_url, get_sublinks, process_all_sublinks, \
    remove_headers, remove_long_patterns
from graphai.api.celery_tasks.common import compute_text_fingerprint_common, fingerprint_lookup_retrieve_from_db, \
    fingerprint_lookup_parallel, fingerprint_lookup_direct, fingerprint_lookup_callback


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.initialize_scraping_url', ignore_result=False)
def initialize_scraping_url_task(self, token, url, force=False):
    if not force:
        # TODO database lookup
        pass
    base_url, validated_url, status_msg, status_code = initialize_url(url, base_url=token)
    validation_results = {
        'token': token,
        'validated_url': validated_url,
        'status_msg': status_msg,
        'sublink_results': None
    }
    return validation_results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.get_scraping_sublinks', ignore_result=False)
def get_scraping_sublinks_task(self, results):
    # This condition is only met if force=False and there is a cache hit
    if results['sublink_results'] is not None:
        return results['sublink_results']
    sublinks, data, validated_url = get_sublinks(results.get('validated_url', None))
    return {
        'token': results['token'],
        'sublinks': sublinks,
        'data': data,
        'validated_url': validated_url,
        'status_msg': results['status_msg']
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.scraping_sublinks_callback', ignore_result=False)
def scraping_sublinks_callback_task(self, results, token):
    if results['sublinks'] is not None and len(results['sublinks']) > 0:
        # TODO do the db callback
        pass
    return results
