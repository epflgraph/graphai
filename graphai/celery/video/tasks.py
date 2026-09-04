import functools
import time

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars

from celery import shared_task

from graphai.core.common.logging import get_logger

from graphai.core.video.video import (
    retrieve_file_from_url,
    retrieve_file_from_url_callback,
    compute_video_fingerprint,
    compute_video_fingerprint_callback,
    video_id_and_duration_fp_lookup,
    cache_lookup_retrieve_file_from_url,
    cache_lookup_extract_audio,
    extract_audio,
    extract_audio_callback,
    reextract_cached_audio,
    compute_audio_fingerprint,
    compute_audio_fingerprint_callback,
    cache_lookup_detect_slides,
    extract_and_sample_frames,
    compute_noise_level_parallel,
    compute_noise_threshold_callback,
    compute_slide_transitions_parallel,
    compute_slide_transitions_callback,
    detect_slides_callback,
    reextract_cached_slides,
    compute_single_image_fingerprint,
    compute_slide_set_fingerprint,
    compute_slide_fingerprint_callback,
    retrieve_slide_fingerprint_callback,
    ignore_slide_fingerprint_results_callback,
    ignore_audio_fingerprint_results_callback,
    retrieve_audio_fingerprint_callback,
    retrieve_video_fingerprint_callback,
    ignore_video_fingerprint_results_callback,
    ignore_single_image_fingerprint_results_callback,
    add_token_status_to_single_image
)
from graphai.core.video.video_utils import NLPModels, NLP_UNLOAD_WAITING_PERIOD
from graphai.core.common.caching import (
    AudioDBCachingManager,
    SlideDBCachingManager,
    VideoDBCachingManager,
    VideoConfig
)
from graphai.core.common.common_utils import (
    strtobool
)

from graphai.core.common.lookup import (
    fingerprint_lookup_retrieve_from_db,
    fingerprint_lookup_parallel,
    fingerprint_lookup_direct,
    fingerprint_lookup_callback,
    fingerprint_cache_lookup
)
from graphai.core.common.config import config

logger = get_logger('graphai.celery.video')

file_management_config = VideoConfig()

local_ocr_nlp_models = NLPModels()


def bind_request_id(task_fn):
    """Decorator that binds request_id and logs task start/completion.

    This lets all downstream library logs (video.py, fingerprints, etc.) carry the
    same request_id without every task needing to manually thread it through, and
    gives every video task an automatic lifecycle frame.
    """
    @functools.wraps(task_fn)
    def wrapper(self, *args, **kwargs):
        request_id = kwargs.pop('request_id', None)
        clear_contextvars()
        if request_id:
            bind_contextvars(request_id=request_id)
        start = time.perf_counter()
        task_name = getattr(task_fn, '__name__', 'unknown_task')
        # Celery task names are "queue.taskname"; the first segment is the queue/service.
        full_name = getattr(self, 'name', '')
        queue = full_name.split('.')[0] if full_name else 'unknown'
        logger.debug(
            f'▶️ Starting {task_name}',
            task_id=self.request.id,
            task_name=task_name,
            queue=queue,
        )
        try:
            result = task_fn(self, *args, **kwargs)
            logger.debug(
                f'✅ Completed {task_name}',
                task_id=self.request.id,
                task_name=task_name,
                queue=queue,
                duration_ms=int((time.perf_counter() - start) * 1000),
            )
            return result
        except Exception as exc:
            logger.error(
                f'❌ Failed {task_name}',
                task_id=self.request.id,
                task_name=task_name,
                queue=queue,
                error_type=type(exc).__name__,
                error=str(exc),
                duration_ms=int((time.perf_counter() - start) * 1000),
            )
            raise
    return wrapper


def shared_video_task(*args, **kwargs):
    """Celery shared_task wrapper that also binds request_id contextvars."""
    def decorator(fn):
        return shared_task(*args, **kwargs)(bind_request_id(fn))
    return decorator


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.init_slide_detection', ignore_result=False,
             nlp_obj=local_ocr_nlp_models)
def slide_detection_init_task(self):
    start = time.perf_counter()
    logger.info('🚀 Start video init_slide_detection task')

    if strtobool(config['preload'].get('video', 'no')):
        logger.debug('⏳ Loading NLP models for slide detection')
        self.nlp_obj.load_nlp_models()
    else:
        logger.debug('⏭️ Skipping preloading for slide detection endpoint')

    logger.debug('🗄️ Initializing video and slide database caching managers')
    VideoDBCachingManager(initialize_database=True)
    SlideDBCachingManager(initialize_database=True)

    logger.info(
        '✅ Video init_slide_detection task complete',
        duration_ms=int((time.perf_counter() - start) * 1000),
    )
    return True


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.clean_up_nlp_objects', ignore_result=False,
             nlp_model=local_ocr_nlp_models)
def cleanup_nlp_objects_task(self):
    """Periodic task that releases the large fasttext models from memory."""
    unloaded = self.nlp_model.unload_model(NLP_UNLOAD_WAITING_PERIOD)
    logger.info('🧹 Cleaned up NLP objects', unloaded_languages=unloaded)
    return unloaded


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='caching.cache_lookup_retrieve_url', ignore_result=False,
             file_manager=file_management_config)
def cache_lookup_retrieve_file_from_url_task(self, url):
    return cache_lookup_retrieve_file_from_url(url, self.file_manager)


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.retrieve_url', ignore_result=False,
             file_manager=file_management_config)
def retrieve_file_from_url_task(self, url, is_kaltura=True, force_token=None):
    logger.debug('📥 Retrieving video from URL', url=url, is_kaltura=is_kaltura, force_token=force_token)
    result = retrieve_file_from_url(url, self.file_manager, is_kaltura, force_token)
    token = result.get('token') if isinstance(result, dict) else None
    logger.info('✅ Video retrieved from URL', url=url, token=token)
    return result


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.retrieve_url_callback', ignore_result=False,
             file_manager=file_management_config)
def retrieve_file_from_url_callback_task(self, results, url):
    return retrieve_file_from_url_callback(results, url)


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='caching.cache_lookup_fingerprint_video', ignore_result=False,
             file_manager=file_management_config)
def cache_lookup_fingerprint_video_task(self, token):
    return fingerprint_cache_lookup(token, VideoDBCachingManager())


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.fingerprint_video', ignore_result=False,
             file_manager=file_management_config)
def compute_video_fingerprint_task(self, results, force=False):
    logger.debug('🎬 Computing video fingerprint', force=force)
    result = compute_video_fingerprint(results, self.file_manager, force)
    logger.info('✅ Video fingerprint computed', force=force)
    return result


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.fingerprint_video_callback', ignore_result=False,
             file_manager=file_management_config)
def compute_video_fingerprint_callback_task(self, results):
    return compute_video_fingerprint_callback(results)


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.video_id_and_duration_fp_lookup', ignore_result=False,
             file_manager=file_management_config)
def video_id_and_duration_fp_lookup_task(self, results):
    return video_id_and_duration_fp_lookup(results)


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.video_fingerprint_find_closest_retrieve_from_db', ignore_result=False)
def video_fingerprint_find_closest_retrieve_from_db_task(self, results):
    return fingerprint_lookup_retrieve_from_db(results, VideoDBCachingManager())


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.video_fingerprint_find_closest_parallel', ignore_result=False)
def video_fingerprint_find_closest_parallel_task(self, input_dict, i, n_total,
                                                 min_similarity=1):
    return fingerprint_lookup_parallel(input_dict, i, n_total, min_similarity, VideoDBCachingManager(), data_type='video')


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.video_fingerprint_find_closest_callback', ignore_result=False)
def video_fingerprint_find_closest_callback_task(self, results_list):
    return fingerprint_lookup_callback(results_list, VideoDBCachingManager())


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.retrieve_video_fingerprint_final_callback', ignore_result=False)
def retrieve_video_fingerprint_callback_task(self, results):
    return retrieve_video_fingerprint_callback(results)


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.ignore_video_fingerprint_results_callback', ignore_result=False)
def ignore_video_fingerprint_results_callback_task(self, results):
    return ignore_video_fingerprint_results_callback(results)


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.get_file', ignore_result=False,
             file_manager=file_management_config)
def get_file_task(self, filename):
    file_path = self.file_manager.generate_filepath(filename)
    logger.info('📁 Resolved video file path', filename=filename, file_path=file_path)
    return file_path


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='caching.cache_lookup_extract_audio', ignore_result=False)
def cache_lookup_extract_audio_task(self, token):
    return cache_lookup_extract_audio(token)


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.extract_audio', ignore_result=False,
             file_manager=file_management_config)
def extract_audio_task(self, token):
    logger.debug('🎵 Extracting audio from video', token=token)
    result = extract_audio(token, self.file_manager)
    logger.info('✅ Audio extraction complete', token=token, has_audio=result is not None)
    return result


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.extract_audio_callback', ignore_result=False,
             file_manager=file_management_config)
def extract_audio_callback_task(self, results, origin_token, force=False):
    logger.debug('🎵 Processing extracted audio callback', origin_token=origin_token, force=force)
    result = extract_audio_callback(results, origin_token, self.file_manager, force)
    logger.info('✅ Audio callback processed', origin_token=origin_token)
    return result


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.reextract_cached_audio', ignore_result=False,
             file_manager=file_management_config)
def reextract_cached_audio_task(self, token):
    logger.debug('🔄 Re-extracting cached audio', token=token)
    result = reextract_cached_audio(token, self.file_manager)
    logger.info('✅ Cached audio re-extracted', token=token)
    return result


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='voice.audio_fingerprint', ignore_result=False,
             file_manager=file_management_config)
def compute_audio_fingerprint_task(self, results, force=False):
    logger.debug('🎵 Computing audio fingerprint', force=force)
    result = compute_audio_fingerprint(results, self.file_manager, force)
    logger.info('✅ Audio fingerprint computed', force=force)
    return result


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='voice.audio_fingerprint_callback', ignore_result=False)
def compute_audio_fingerprint_callback_task(self, results, force=False):
    return compute_audio_fingerprint_callback(results, force)


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='voice.audio_fingerprint_find_closest_retrieve_from_db', ignore_result=False)
def audio_fingerprint_find_closest_retrieve_from_db_task(self, results):
    return fingerprint_lookup_retrieve_from_db(results, AudioDBCachingManager())


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='voice.audio_fingerprint_find_closest_parallel', ignore_result=False)
def audio_fingerprint_find_closest_parallel_task(self, input_dict, i, n_total, min_similarity=1):
    return fingerprint_lookup_parallel(input_dict, i, n_total, min_similarity, AudioDBCachingManager(), data_type='audio')


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='voice.audio_fingerprint_find_closest_direct', ignore_result=False)
def audio_fingerprint_find_closest_direct_task(self, results):
    return fingerprint_lookup_direct(results, AudioDBCachingManager())


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='voice.audio_fingerprint_find_closest_callback', ignore_result=False)
def audio_fingerprint_find_closest_callback_task(self, results_list):
    return fingerprint_lookup_callback(results_list, AudioDBCachingManager())


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='voice.retrieve_audio_fingerprint_final_callback', ignore_result=False)
def retrieve_audio_fingerprint_callback_task(self, results):
    return retrieve_audio_fingerprint_callback(results)


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='voice.ignore_audio_fingerprint_results_callback', ignore_result=False)
def ignore_audio_fingerprint_results_callback_task(self, results):
    return ignore_audio_fingerprint_results_callback(results)


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='caching.cache_lookup_detect_slides', ignore_result=False,
             file_manager=file_management_config)
def cache_lookup_detect_slides_task(self, token):
    return cache_lookup_detect_slides(token)


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.extract_and_sample_frames', ignore_result=False,
             file_manager=file_management_config)
def extract_and_sample_frames_task(self, token):
    logger.debug('🖼️ Extracting and sampling video frames', token=token)
    result = extract_and_sample_frames(token, self.file_manager)
    logger.info('✅ Frames extracted and sampled', token=token)
    return result


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.noise_level_parallel', ignore_result=False,
             file_manager=file_management_config,
             nlp_model=local_ocr_nlp_models)
def compute_noise_level_parallel_task(self, results, i, n, language=None):
    logger.debug('🔊 Computing noise level for frame shard', shard=i, total_shards=n, language=language)
    result = compute_noise_level_parallel(results, i, n, language, self.file_manager, self.nlp_model)
    logger.debug('✅ Noise level computed for frame shard', shard=i, total_shards=n)
    return result


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.noise_level_callback', ignore_result=False)
def compute_noise_threshold_callback_task(self, results, hash_thresh=0.8, multiplier=5, default_threshold=0.05):
    logger.debug('📊 Computing noise threshold', hash_thresh=hash_thresh, multiplier=multiplier, default_threshold=default_threshold)
    result = compute_noise_threshold_callback(results, hash_thresh, multiplier, default_threshold)
    logger.debug('✅ Noise threshold computed')
    return result


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.slide_transitions_parallel', ignore_result=False,
             file_manager=file_management_config, nlp_model=local_ocr_nlp_models)
def compute_slide_transitions_parallel_task(self, results, i, n, language=None, include_first=True, include_last=True):
    logger.debug('🖼️ Computing slide transitions for shard', shard=i, total_shards=n, language=language)
    result = compute_slide_transitions_parallel(results, i, n, language, self.file_manager, self.nlp_model,
                                                include_first, include_last)
    logger.debug('✅ Slide transitions computed for shard', shard=i, total_shards=n)
    return result


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.slide_transitions_callback', ignore_result=False,
             file_manager=file_management_config, nlp_model=local_ocr_nlp_models)
def compute_slide_transitions_callback_task(self, results, language=None):
    logger.debug('🖼️ Aggregating slide transitions', language=language)
    result = compute_slide_transitions_callback(results, language, self.file_manager, self.nlp_model)
    logger.info('✅ Slide transitions aggregated', language=language)
    return result


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.detect_slides_callback', ignore_result=False,
             file_manager=file_management_config)
def detect_slides_callback_task(self, results, token, force=False):
    logger.debug('🖼️ Finalizing slide detection', token=token, force=force, retries=self.request.retries)
    result = detect_slides_callback(results, token, self.file_manager, force, self.request.retries)
    slide_count = len(result) if isinstance(result, (list, tuple)) else None
    logger.info('✅ Slide detection finalized', token=token, slide_count=slide_count)
    return result


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.reextract_cached_slides', ignore_result=False,
             file_manager=file_management_config)
def reextract_cached_slides_task(self, token):
    logger.debug('🔄 Re-extracting cached slides', token=token)
    result = reextract_cached_slides(token, self.file_manager)
    logger.info('✅ Cached slides re-extracted', token=token)
    return result


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='image.slide_fingerprint', ignore_result=False,
             file_manager=file_management_config)
def compute_single_image_fingerprint_task(self, results):
    logger.debug('🖼️ Computing single image fingerprint')
    result = compute_single_image_fingerprint(results, self.file_manager)
    logger.info('✅ Single image fingerprint computed')
    return result


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='image.slide_set_fingerprint', ignore_result=False,
             file_manager=file_management_config)
def compute_slide_set_fingerprint_task(self, results, origin_token):
    logger.debug('🖼️ Computing slide set fingerprint', origin_token=origin_token)
    result = compute_slide_set_fingerprint(results, origin_token, self.file_manager)
    logger.info('✅ Slide set fingerprint computed', origin_token=origin_token)
    return result


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='image.slide_fingerprint_callback', ignore_result=False)
def compute_slide_fingerprint_callback_task(self, results, force=False):
    return compute_slide_fingerprint_callback(results, force)


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='image.slide_fingerprint_find_closest_retrieve_from_db', ignore_result=False)
def slide_fingerprint_find_closest_retrieve_from_db_task(self, results):
    return fingerprint_lookup_retrieve_from_db(results, SlideDBCachingManager())


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='image.slide_fingerprint_find_closest_parallel', ignore_result=False)
def slide_fingerprint_find_closest_parallel_task(self, input_dict, i, n_total, min_similarity=1):
    return fingerprint_lookup_parallel(input_dict, i, n_total, min_similarity, SlideDBCachingManager(), data_type='image')


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='image.slide_fingerprint_find_closest_direct', ignore_result=False)
def slide_fingerprint_find_closest_direct_task(self, results):
    return fingerprint_lookup_direct(results, SlideDBCachingManager())


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='image.slide_fingerprint_find_closest_callback', ignore_result=False)
def slide_fingerprint_find_closest_callback_task(self, results_list):
    return fingerprint_lookup_callback(results_list, SlideDBCachingManager())


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='image.retrieve_slide_fingerprint_final_callback', ignore_result=False)
def retrieve_slide_fingerprint_callback_task(self, results):
    return retrieve_slide_fingerprint_callback(results)


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='image.ignore_slide_fingerprint_results_callback', ignore_result=False)
def ignore_slide_fingerprint_results_callback_task(self, results):
    return ignore_slide_fingerprint_results_callback(results)


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='image.ignore_single_image_fingerprint_results_callback', ignore_result=False)
def ignore_single_image_fingerprint_results_callback_task(self, results):
    return ignore_single_image_fingerprint_results_callback(results)


@shared_video_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='image.add_token_status_to_single_image_results_callback', ignore_result=False)
def add_token_status_to_single_image_results_callback_task(self, results):
    return add_token_status_to_single_image(results)
