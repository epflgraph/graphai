from celery import shared_task

from graphai.core.image.image import (
    cache_lookup_retrieve_image_from_url,
    retrieve_image_file_from_url,
    retrieve_image_file_from_url_callback,
    cache_lookup_extract_slide_text,
    extract_slide_text,
    extract_slide_text_callback
)
from graphai.core.common.caching import (
    SlideDBCachingManager,
    VideoConfig
)
from graphai.core.common.lookup import fingerprint_cache_lookup_with_most_similar

file_management_config = VideoConfig()


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='caching_6.cache_lookup_retrieve_image', ignore_result=False,
             file_manager=file_management_config)
def cache_lookup_retrieve_image_from_url_task(self, url):
    return cache_lookup_retrieve_image_from_url(url, self.file_manager)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.retrieve_image', ignore_result=False,
             file_manager=file_management_config)
def retrieve_image_from_url_task(self, url, force_token=None):
    return retrieve_image_file_from_url(url, self.file_manager, force_token)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.retrieve_image_callback', ignore_result=False,
             file_manager=file_management_config)
def retrieve_image_from_url_callback_task(self, results, url):
    return retrieve_image_file_from_url_callback(results, url)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='caching_6.cache_lookup_fingerprint_slide', ignore_result=False)
def cache_lookup_slide_fingerprint_task(self, token):
    return fingerprint_cache_lookup_with_most_similar(token, SlideDBCachingManager(), None)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='caching_6.cache_lookup_extract_slide_text', ignore_result=False)
def cache_lookup_extract_slide_text_task(self, token, method='tesseract'):
    return cache_lookup_extract_slide_text(token, method)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.extract_slide_text', ignore_result=False,
             file_manager=file_management_config)
def extract_slide_text_task(self, token, method='google', api_token=None, openai_token=None):
    return extract_slide_text(token, self.file_manager, method, api_token, openai_token)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.extract_slide_text_callback', ignore_result=False)
def extract_slide_text_callback_task(self, results, token, force=False):
    return extract_slide_text_callback(results, token, force)
