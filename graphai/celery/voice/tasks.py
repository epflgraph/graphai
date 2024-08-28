from celery import shared_task

from graphai.core.voice.transcribe import (
    WHISPER_UNLOAD_WAITING_PERIOD,
    WhisperTranscriptionModel,
    detect_language_parallel,
    detect_language_retrieve_from_db_and_split,
    detect_language_callback,
    transcribe_audio_to_text, transcribe_callback
)
from graphai.core.common.caching import (
    cache_lookup_generic,
    AudioDBCachingManager,
    VideoConfig,
    fingerprint_cache_lookup_with_most_similar
)
from graphai.core.common.config import config
from graphai.core.common.common_utils import strtobool

file_management_config = VideoConfig()

transcription_model = WhisperTranscriptionModel()


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.init_transcription', ignore_result=False,
             transcription_obj=transcription_model)
def transcript_init_task(self):
    print('Start init_transcription task')

    if strtobool(config['preload'].get('video', 'no')):
        print('Loading transcription model...')
        self.transcription_obj.load_model_whisper()
    else:
        print('Skipping preloading for voice endpoints')

    print('Initializing db caching managers...')
    AudioDBCachingManager(initialize_database=True)

    return True


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='caching_6.cache_lookup_fingerprint_audio', ignore_result=False)
def cache_lookup_audio_fingerprint_task(self, token):
    return fingerprint_cache_lookup_with_most_similar(token, AudioDBCachingManager(), ['duration'])


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='caching_6.cache_lookup_detect_language_audio', ignore_result=False)
def cache_lookup_audio_language_task(self, token):
    return cache_lookup_generic(token, AudioDBCachingManager(), ['language'])


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True,
             retry_kwargs={"max_retries": 2}, name='video_2.detect_language_retrieve_from_db', ignore_result=False,
             file_manager=file_management_config)
def detect_language_retrieve_from_db_and_split_task(self, input_dict, n_divs=5, segment_length=30):
    return detect_language_retrieve_from_db_and_split(input_dict, self.file_manager, n_divs, segment_length)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True,
             retry_kwargs={"max_retries": 2}, name='video_2.detect_language_parallel', ignore_result=False,
             file_manager=file_management_config, model=transcription_model)
def detect_language_parallel_task(self, tokens_dict, i):
    return detect_language_parallel(tokens_dict, i, self.model, self.file_manager)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True,
             retry_kwargs={"max_retries": 2}, name='video_2.detect_language_callback', ignore_result=False,
             file_manager=file_management_config)
def detect_language_callback_task(self, results_list, token, force=False):
    # Here, even a single error (corresponding to a None value for the 'lang' key) will cause a failure.
    # If all the detected languages are non-null, the results are valid and are inserted into the database.
    return detect_language_callback(token, results_list, force)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='caching_6.cache_lookup_transcribe_audio', ignore_result=False)
def cache_lookup_audio_transcript_task(self, token):
    return cache_lookup_generic(token, AudioDBCachingManager(),
                                ['transcript_results', 'subtitle_results', 'language'])


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True,
             retry_kwargs={"max_retries": 2}, name='video_2.transcribe', ignore_result=False,
             file_manager=file_management_config, model=transcription_model)
def transcribe_task(self, input_dict, strict_silence=False):
    return transcribe_audio_to_text(input_dict, self.model, self.file_manager, strict_silence)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.transcribe_callback', ignore_result=False,
             file_manager=file_management_config)
def transcribe_callback_task(self, results, token, force=False):
    return transcribe_callback(token, results, force)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.clean_up_transcription_object', model=transcription_model, ignore_result=False)
def cleanup_transcription_object_task(self):
    return self.model.unload_model(WHISPER_UNLOAD_WAITING_PERIOD)
