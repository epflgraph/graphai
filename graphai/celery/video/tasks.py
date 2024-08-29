from celery import shared_task

from graphai.core.video.video import (
    NLPModels,
    retrieve_file_from_url,
    retrieve_file_from_url_callback,
    compute_video_fingerprint,
    compute_video_fingerprint_callback,
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
    compute_slide_fingerprint,
    compute_slide_set_fingerprint,
    compute_slide_fingerprint_callback,
    retrieve_slide_fingerprint_callback,
    ignore_slide_fingerprint_results_callback,
    ignore_audio_fingerprint_results_callback,
    retrieve_audio_fingerprint_callback,
    retrieve_video_fingerprint_callback,
    ignore_video_fingerprint_results_callback
)
from graphai.core.common.caching import (
    AudioDBCachingManager,
    SlideDBCachingManager,
    VideoDBCachingManager,
    VideoConfig,
    fingerprint_cache_lookup
)
from graphai.core.common.common_utils import (
    strtobool
)

from graphai.core.common.lookup import fingerprint_lookup_retrieve_from_db, fingerprint_lookup_parallel, \
    fingerprint_lookup_direct, fingerprint_lookup_callback
from graphai.core.common.config import config

file_management_config = VideoConfig()

local_ocr_nlp_models = NLPModels()


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.init_slide_detection', ignore_result=False,
             nlp_obj=local_ocr_nlp_models)
def slide_detection_init_task(self):
    # This task initialises the video celery worker by loading into memory the transcription and NLP models
    print('Start init_slide_detection task')

    if strtobool(config['preload'].get('video', 'no')):
        print('Loading NLP models...')
        self.nlp_obj.load_nlp_models()
    else:
        print('Skipping preloading for slide detection endpoint')

    print('Initializing db caching managers...')
    VideoDBCachingManager(initialize_database=True)
    SlideDBCachingManager(initialize_database=True)

    return True


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='caching_6.cache_lookup_retrieve_url', ignore_result=False,
             file_manager=file_management_config)
def cache_lookup_retrieve_file_from_url_task(self, url):
    return cache_lookup_retrieve_file_from_url(url, self.file_manager)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.retrieve_url', ignore_result=False,
             file_manager=file_management_config)
def retrieve_file_from_url_task(self, url, is_kaltura=True, force_token=None):
    return retrieve_file_from_url(url, self.file_manager, is_kaltura, force_token)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.retrieve_url_callback', ignore_result=False,
             file_manager=file_management_config)
def retrieve_file_from_url_callback_task(self, results, url):
    return retrieve_file_from_url_callback(results, url)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='caching_6.cache_lookup_fingerprint_video', ignore_result=False,
             file_manager=file_management_config)
def cache_lookup_fingerprint_video_task(self, token):
    return fingerprint_cache_lookup(token, VideoDBCachingManager())


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.fingerprint_video', ignore_result=False,
             file_manager=file_management_config)
def compute_video_fingerprint_task(self, results, force=False):
    return compute_video_fingerprint(results, self.file_manager, force)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.fingerprint_video_callback', ignore_result=False,
             file_manager=file_management_config)
def compute_video_fingerprint_callback_task(self, results):
    return compute_video_fingerprint_callback(results)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.video_fingerprint_find_closest_retrieve_from_db', ignore_result=False)
def video_fingerprint_find_closest_retrieve_from_db_task(self, results):
    return fingerprint_lookup_retrieve_from_db(results, VideoDBCachingManager())


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.video_fingerprint_find_closest_parallel', ignore_result=False)
def video_fingerprint_find_closest_parallel_task(self, input_dict, i, n_total,
                                                 min_similarity=1):
    return fingerprint_lookup_parallel(input_dict, i, n_total, min_similarity, VideoDBCachingManager(), data_type='video')


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.video_fingerprint_find_closest_callback', ignore_result=False)
def video_fingerprint_find_closest_callback_task(self, results_list):
    return fingerprint_lookup_callback(results_list, VideoDBCachingManager())


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.retrieve_video_fingerprint_final_callback', ignore_result=False)
def retrieve_video_fingerprint_callback_task(self, results):
    return retrieve_video_fingerprint_callback(results)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.ignore_video_fingerprint_results_callback', ignore_result=False)
def ignore_video_fingerprint_results_callback_task(self, results):
    return ignore_video_fingerprint_results_callback(results)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.get_file', ignore_result=False,
             file_manager=file_management_config)
def get_file_task(self, filename):
    return self.file_manager.generate_filepath(filename)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='caching_6.cache_lookup_extract_audio', ignore_result=False)
def cache_lookup_extract_audio_task(self, token):
    return cache_lookup_extract_audio(token)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.extract_audio', ignore_result=False,
             file_manager=file_management_config)
def extract_audio_task(self, token):
    return extract_audio(token, self.file_manager)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.extract_audio_callback', ignore_result=False,
             file_manager=file_management_config)
def extract_audio_callback_task(self, results, origin_token, force=False):
    return extract_audio_callback(results, origin_token, self.file_manager, force)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.reextract_cached_audio', ignore_result=False,
             file_manager=file_management_config)
def reextract_cached_audio_task(self, token):
    return reextract_cached_audio(token, self.file_manager)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.audio_fingerprint', ignore_result=False,
             file_manager=file_management_config)
def compute_audio_fingerprint_task(self, results, force=False):
    return compute_audio_fingerprint(results, self.file_manager, force)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.audio_fingerprint_callback', ignore_result=False)
def compute_audio_fingerprint_callback_task(self, results, force=False):
    return compute_audio_fingerprint_callback(results, force)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.audio_fingerprint_find_closest_retrieve_from_db', ignore_result=False)
def audio_fingerprint_find_closest_retrieve_from_db_task(self, results):
    return fingerprint_lookup_retrieve_from_db(results, AudioDBCachingManager())


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.audio_fingerprint_find_closest_parallel', ignore_result=False)
def audio_fingerprint_find_closest_parallel_task(self, input_dict, i, n_total, min_similarity=1):
    return fingerprint_lookup_parallel(input_dict, i, n_total, min_similarity, AudioDBCachingManager(), data_type='audio')


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.audio_fingerprint_find_closest_direct', ignore_result=False)
def audio_fingerprint_find_closest_direct_task(self, results):
    return fingerprint_lookup_direct(results, AudioDBCachingManager())


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.audio_fingerprint_find_closest_callback', ignore_result=False)
def audio_fingerprint_find_closest_callback_task(self, results_list):
    return fingerprint_lookup_callback(results_list, AudioDBCachingManager())


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.retrieve_audio_fingerprint_final_callback', ignore_result=False)
def retrieve_audio_fingerprint_callback_task(self, results):
    return retrieve_audio_fingerprint_callback(results)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.ignore_audio_fingerprint_results_callback', ignore_result=False)
def ignore_audio_fingerprint_results_callback_task(self, results):
    return ignore_audio_fingerprint_results_callback(results)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='caching_6.cache_lookup_detect_slides', ignore_result=False,
             file_manager=file_management_config)
def cache_lookup_detect_slides_task(self, token):
    return cache_lookup_detect_slides(token)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.extract_and_sample_frames', ignore_result=False,
             file_manager=file_management_config)
def extract_and_sample_frames_task(self, token):
    return extract_and_sample_frames(token, self.file_manager)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.noise_level_parallel', ignore_result=False,
             file_manager=file_management_config,
             nlp_model=local_ocr_nlp_models)
def compute_noise_level_parallel_task(self, results, i, n, language=None):
    return compute_noise_level_parallel(results, i, n, language, self.file_manager, self.nlp_model)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.noise_level_callback', ignore_result=False)
def compute_noise_threshold_callback_task(self, results, hash_thresh=0.8):
    return compute_noise_threshold_callback(results, hash_thresh)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.slide_transitions_parallel', ignore_result=False,
             file_manager=file_management_config, nlp_model=local_ocr_nlp_models)
def compute_slide_transitions_parallel_task(self, results, i, n, language=None):
    return compute_slide_transitions_parallel(results, i, n, language, self.file_manager, self.nlp_model)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.slide_transitions_callback', ignore_result=False,
             file_manager=file_management_config, nlp_model=local_ocr_nlp_models)
def compute_slide_transitions_callback_task(self, results, language=None):
    return compute_slide_transitions_callback(results, language, self.file_manager, self.nlp_model)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.detect_slides_callback', ignore_result=False,
             file_manager=file_management_config)
def detect_slides_callback_task(self, results, token, force=False):
    return detect_slides_callback(results, token, self.file_manager, force, self.request.retries)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.reextract_cached_slides', ignore_result=False,
             file_manager=file_management_config)
def reextract_cached_slides_task(self, token):
    return reextract_cached_slides(token, self.file_manager)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.slide_fingerprint', ignore_result=False,
             file_manager=file_management_config)
def compute_slide_fingerprint_task(self, token):
    return compute_slide_fingerprint(token, self.file_manager)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.slide_set_fingerprint', ignore_result=False,
             file_manager=file_management_config)
def compute_slide_set_fingerprint_task(self, results, origin_token):
    return compute_slide_set_fingerprint(results, origin_token, self.file_manager)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.slide_fingerprint_callback', ignore_result=False)
def compute_slide_fingerprint_callback_task(self, results, force=False):
    return compute_slide_fingerprint_callback(results, force)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.slide_fingerprint_find_closest_retrieve_from_db', ignore_result=False)
def slide_fingerprint_find_closest_retrieve_from_db_task(self, results):
    return fingerprint_lookup_retrieve_from_db(results, SlideDBCachingManager())


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.slide_fingerprint_find_closest_parallel', ignore_result=False)
def slide_fingerprint_find_closest_parallel_task(self, input_dict, i, n_total, min_similarity=1):
    return fingerprint_lookup_parallel(input_dict, i, n_total, min_similarity, SlideDBCachingManager(), data_type='image')


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.slide_fingerprint_find_closest_direct', ignore_result=False)
def slide_fingerprint_find_closest_direct_task(self, results):
    return fingerprint_lookup_direct(results, SlideDBCachingManager())


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.slide_fingerprint_find_closest_callback', ignore_result=False)
def slide_fingerprint_find_closest_callback_task(self, results_list):
    return fingerprint_lookup_callback(results_list, SlideDBCachingManager())


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.retrieve_slide_fingerprint_final_callback', ignore_result=False)
def retrieve_slide_fingerprint_callback_task(self, results):
    return retrieve_slide_fingerprint_callback(results)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.ignore_slide_fingerprint_results_callback', ignore_result=False)
def ignore_slide_fingerprint_results_callback_task(self, results):
    return ignore_slide_fingerprint_results_callback(results)
