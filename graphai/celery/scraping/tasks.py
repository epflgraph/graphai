from celery import shared_task
from graphai.core.scraping.scraping import (
    cache_lookup_get_sublinks,
    initialize_url_and_get_sublinks,
    scraping_sublinks_callback,
    cache_lookup_process_all_sublinks,
    process_all_scraping_sublinks_parallel,
    sublink_parallel_processing_merge_callback,
    remove_junk_scraping_parallel,
    extract_scraping_content_callback
)
from graphai.core.common.caching import ScrapingDBCachingManager


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='scraping_6.init_scraping', ignore_result=False)
def scraping_init_task(self):
    print('Initializing db caching managers...')
    ScrapingDBCachingManager(initialize_database=True)

    return True


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='caching_6.cache_lookup_get_sublinks', ignore_result=False)
def cache_lookup_get_sublinks_task(self, token):
    return cache_lookup_get_sublinks(token)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='scraping_6.initialize_scraping_url', ignore_result=False)
def initialize_url_and_get_sublinks_task(self, token, url):
    return initialize_url_and_get_sublinks(token, url)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='scraping_6.scraping_sublinks_callback', ignore_result=False)
def scraping_sublinks_callback_task(self, results):
    return scraping_sublinks_callback(results)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='caching_6.cache_lookup_process_all_sublinks', ignore_result=False)
def cache_lookup_process_all_sublinks_task(self, token, headers, long_patterns):
    return cache_lookup_process_all_sublinks(token, headers, long_patterns)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='scraping_6.process_all_scraping_sublinks_parallel', ignore_result=False)
def process_all_scraping_sublinks_parallel_task(self, results, i, n_total):
    return process_all_scraping_sublinks_parallel(results, i, n_total)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='scraping_6.sublink_parallel_processing_merge_callback', ignore_result=False)
def sublink_parallel_processing_merge_callback_task(self, results):
    return sublink_parallel_processing_merge_callback(results)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='scraping_6.remove_junk_scraping_parallel', ignore_result=False)
def remove_junk_scraping_parallel_task(self, results, i, n_total, headers, long_patterns):
    return remove_junk_scraping_parallel(results, i, n_total, headers, long_patterns)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='scraping_6.extract_scraping_content_callback', ignore_result=False)
def extract_scraping_content_callback_task(self, results, headers, long_patterns):
    return extract_scraping_content_callback(results, headers, long_patterns)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='scraping_6.scraping_dummy_task', ignore_result=False)
def scraping_dummy_task(self, results):
    # This task is required for chaining groups due to the peculiarities of celery
    # Whenever there are two groups in one chain of tasks, there need to be at least
    # TWO tasks between them, and this dummy task is simply an f(x)=x function.
    return results
