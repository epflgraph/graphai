# graphai/celery/caching/tasks.py
# Lightweight cache-lookup tasks intended for the dedicated caching worker.
# All heavy core modules are imported lazily inside task functions so that the
# worker master process does not load whisper/torch, presidio, transformers,
# or other ML stacks at startup.
from celery import shared_task

# Public Function: Look up a previously-retrieved video by source URL.
@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2}, name='caching.cache_lookup_retrieve_url', ignore_result=False)
def cache_lookup_retrieve_file_from_url_task(self, url):
    from graphai.core.common.caching import VideoConfig
    from graphai.core.video.video import cache_lookup_retrieve_file_from_url
    return cache_lookup_retrieve_file_from_url(url, VideoConfig())

# Public Function: Look up a cached video fingerprint.
@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2}, name='caching.cache_lookup_fingerprint_video', ignore_result=False)
def cache_lookup_fingerprint_video_task(self, token):
    from graphai.core.common.caching import VideoDBCachingManager
    from graphai.core.common.lookup import fingerprint_cache_lookup
    return fingerprint_cache_lookup(token, VideoDBCachingManager())

# Public Function: Look up cached extracted audio for a video token.
@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2}, name='caching.cache_lookup_extract_audio', ignore_result=False)
def cache_lookup_extract_audio_task(self, token):
    from graphai.core.video.video import cache_lookup_extract_audio
    return cache_lookup_extract_audio(token)

# Public Function: Look up cached slide detection results for a video token.
@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2}, name='caching.cache_lookup_detect_slides', ignore_result=False)
def cache_lookup_detect_slides_task(self, token):
    from graphai.core.video.video import cache_lookup_detect_slides
    return cache_lookup_detect_slides(token)

# Public Function: Look up a cached audio fingerprint.
@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2}, name='caching.cache_lookup_fingerprint_audio', ignore_result=False)
def cache_lookup_audio_fingerprint_task(self, token):
    from graphai.core.common.caching import AudioDBCachingManager
    from graphai.core.common.lookup import fingerprint_cache_lookup
    return fingerprint_cache_lookup(token, AudioDBCachingManager())

# Public Function: Look up cached audio language detection.
@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2}, name='caching.cache_lookup_detect_language_audio', ignore_result=False)
def cache_lookup_audio_language_task(self, token):
    from graphai.core.common.caching import AudioDBCachingManager
    from graphai.core.common.lookup import cache_lookup_generic
    return cache_lookup_generic(token, AudioDBCachingManager(), ['language'])

# Public Function: Look up cached audio transcript / subtitles.
@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2}, name='caching.cache_lookup_transcribe_audio', ignore_result=False)
def cache_lookup_audio_transcript_task(self, token):
    from graphai.core.common.caching import AudioDBCachingManager
    from graphai.core.common.lookup import cache_lookup_generic
    return cache_lookup_generic(token, AudioDBCachingManager(),
                                ['transcript_results', 'subtitle_results', 'language'])

# Public Function: Look up a cached translation text fingerprint.
@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2}, name='caching.cache_lookup_fingerprint_translation_text', ignore_result=False)
def cache_lookup_translation_text_fingerprint_task(self, token):
    from graphai.core.common.caching import TextDBCachingManager
    from graphai.core.common.lookup import fingerprint_cache_lookup
    return fingerprint_cache_lookup(token, TextDBCachingManager())

# Public Function: Look up a cached translation by fingerprint.
@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2}, name='caching.translation_text_lookup_using_fingerprint', ignore_result=False)
def cache_lookup_translation_text_using_fingerprint_task(self, token, fp, src, tgt, return_list=False):
    from graphai.core.translation.translation import cache_lookup_translation_text_using_fingerprint
    return cache_lookup_translation_text_using_fingerprint(token, fp, src, tgt, return_list)

# Public Function: Look up a cached translation by token.
@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2}, name='caching.cache_lookup_translate_text', ignore_result=False)
def cache_lookup_translate_text_task(self, token, return_list=False):
    from graphai.core.translation.translation import cache_lookup_translate_text
    return cache_lookup_translate_text(token, return_list)

# Public Function: Look up a cached embedding text fingerprint.
@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2}, name='caching.cache_lookup_fingerprint_embedding_text', ignore_result=False)
def cache_lookup_embedding_text_fingerprint_task(self, token):
    from graphai.core.common.caching import EmbeddingDBCachingManager
    from graphai.core.common.lookup import fingerprint_cache_lookup
    return fingerprint_cache_lookup(token, EmbeddingDBCachingManager())

# Public Function: Look up a cached embedding by fingerprint.
@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2}, name='caching.embedding_text_lookup_using_fingerprint', ignore_result=False)
def cache_lookup_embedding_text_using_fingerprint_task(self, token, fp, model_type):
    from graphai.core.embedding.embedding import fingerprint_based_embedding_lookup
    return fingerprint_based_embedding_lookup(token, fp, model_type)

# Public Function: Look up a cached embedding by token.
@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2}, name='caching.cache_lookup_embedding_text', ignore_result=False)
def cache_lookup_embedding_text_task(self, token, model_type):
    from graphai.core.embedding.embedding import token_based_embedding_lookup
    return token_based_embedding_lookup(token, model_type)

# Public Function: Look up a previously-retrieved image by source URL.
@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2}, name='caching.cache_lookup_retrieve_image', ignore_result=False)
def cache_lookup_retrieve_image_from_url_task(self, url):
    from graphai.core.common.caching import VideoConfig
    from graphai.core.image.image import cache_lookup_retrieve_image_from_url
    return cache_lookup_retrieve_image_from_url(url, VideoConfig())

# Public Function: Look up a cached slide/image fingerprint.
@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2}, name='caching.cache_lookup_fingerprint_slide', ignore_result=False)
def cache_lookup_slide_fingerprint_task(self, token):
    from graphai.core.common.caching import SlideDBCachingManager
    from graphai.core.common.lookup import fingerprint_cache_lookup_with_most_similar
    return fingerprint_cache_lookup_with_most_similar(token, SlideDBCachingManager(), None)

# Public Function: Look up cached slide text extraction.
@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2}, name='caching.cache_lookup_extract_slide_text', ignore_result=False)
def cache_lookup_extract_slide_text_task(self, token, method="tesseract"):
    from graphai.core.image.image import cache_lookup_extract_slide_text
    return cache_lookup_extract_slide_text(token, method)

# Public Function: Look up cached sublinks for a scraping token.
@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2}, name='caching.cache_lookup_get_sublinks', ignore_result=False)
def cache_lookup_get_sublinks_task(self, token):
    from graphai.core.scraping.scraping import cache_lookup_get_sublinks
    return cache_lookup_get_sublinks(token)

# Public Function: Look up cached processed sublinks for a scraping token.
@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2}, name='caching.cache_lookup_process_all_sublinks', ignore_result=False)
def cache_lookup_process_all_sublinks_task(self, token, headers, long_patterns):
    from graphai.core.scraping.scraping import cache_lookup_process_all_sublinks
    return cache_lookup_process_all_sublinks(token, headers, long_patterns)
