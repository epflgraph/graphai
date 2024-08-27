import json
from collections import Counter

from celery import shared_task

from graphai.core.video.video import (
    extract_media_segment
)
from graphai.core.video.transcribe import WHISPER_UNLOAD_WAITING_PERIOD, WhisperTranscriptionModel
from graphai.core.interfaces.caching import (
    TEMP_SUBFOLDER,
    AudioDBCachingManager,
    VideoConfig
)
from graphai.core.interfaces.config import config
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
    db_manager = AudioDBCachingManager()
    existing_list = db_manager.get_details(token, cols=['fingerprint', 'duration'],
                                           using_most_similar=True)

    for existing in existing_list:
        if existing is None:
            continue
        if existing['fingerprint'] is not None:
            # We have a cache hit, now we gather all the results that should be returned
            # The closest match
            existing_closest = db_manager.get_closest_match(token)
            if existing_closest is not None:
                # If the closest match exists, we also want to return its origin token
                existing_closest_origin = db_manager.get_origin(existing_closest)
            else:
                existing_closest_origin = None
            print('Returning cached result')
            return {
                'result': existing['fingerprint'],
                'fresh': False,
                'closest': existing_closest,
                'closest_origin': existing_closest_origin,
                'duration': existing['duration']
            }

    return None


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='caching_6.cache_lookup_detect_language_audio', ignore_result=False)
def cache_lookup_audio_language_task(self, token):
    db_manager = AudioDBCachingManager()
    existing_list = db_manager.get_details(token, ['language'],
                                           using_most_similar=True)
    for existing in existing_list:
        if existing is None:
            continue
        if existing['language'] is not None:
            print('Returning cached result')
            return {
                'token': token,
                'language': existing['language'],
                'fresh': False
            }
    return None


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True,
             retry_kwargs={"max_retries": 2}, name='video_2.detect_language_retrieve_from_db', ignore_result=False,
             file_manager=file_management_config)
def detect_language_retrieve_from_db_and_split_task(self, input_dict, n_divs=5, segment_length=30):
    token = input_dict['token']

    db_manager = AudioDBCachingManager()
    existing = db_manager.get_details(token, ['duration'],
                                      using_most_similar=True)[0]
    if existing is None or existing['duration'] is None:
        # We need the duration of the file from the cache, so the task fails if the cache row doesn't exist
        return {
            'temp_tokens': None,
            'lang': None,
            'fresh': False
        }

    duration = existing['duration']
    input_filename_with_path = self.file_manager.generate_filepath(token)
    result_tokens = list()

    # Creating `n_divs` segments (of duration `length` each) of the audio file and saving them to the temp subfolder
    for i in range(n_divs):
        current_output_token = token + '_' + str(i) + '_temp.ogg'
        current_output_token_with_path = self.file_manager.generate_filepath(current_output_token,
                                                                             force_dir=TEMP_SUBFOLDER)
        current_result = extract_media_segment(
            input_filename_with_path, current_output_token_with_path, current_output_token,
            start=duration * i / n_divs, length=segment_length)
        if current_result is None:
            print('Unspecified error while creating temp files')
            return {
                'lang': None,
                'temp_tokens': None,
                'fresh': False
            }

        result_tokens.append(current_result)

    return {
        'lang': None,
        'temp_tokens': result_tokens,
        'fresh': True
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True,
             retry_kwargs={"max_retries": 2}, name='video_2.detect_language_parallel', ignore_result=False,
             file_manager=file_management_config, model=transcription_model)
def detect_language_parallel_task(self, tokens_dict, i):
    if not tokens_dict['fresh']:
        return {
            'lang': None,
            'fresh': False
        }

    current_token = tokens_dict['temp_tokens'][i]
    try:
        language = self.model.detect_audio_segment_lang_whisper(
            self.file_manager.generate_filepath(current_token, force_dir=TEMP_SUBFOLDER)
        )
    except Exception:
        return {
            'lang': None,
            'fresh': False
        }

    return {
        'lang': language,
        'fresh': True
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True,
             retry_kwargs={"max_retries": 2}, name='video_2.detect_language_callback', ignore_result=False,
             file_manager=file_management_config)
def detect_language_callback_task(self, results_list, token, force=False):
    # Here, even a single error (corresponding to a None value for the 'lang' key) will cause a failure.
    # If all the detected languages are non-null, the results are valid and are inserted into the database.
    if all([x['lang'] is not None for x in results_list]):
        # This indicates success (regardless of freshness)
        languages = [x['lang'] for x in results_list]
        most_common_lang = Counter(languages).most_common(1)[0][0]
        values_dict = {'language': most_common_lang}

        # Inserting values for original token
        db_manager = AudioDBCachingManager()
        db_manager.insert_or_update_details(
            token, values_dict
        )
        if not force:
            closest = db_manager.get_closest_match(token)
            if closest is not None and closest != token:
                # If force=False and there's a closest match, it means that the closest match has not
                # had this computation performed on it, so we insert these results for the closest match
                # as well.
                db_manager.insert_or_update_details(
                    closest, values_dict
                )

        return {
            'token': token,
            'language': most_common_lang,
            'fresh': True
        }

    return {
        'token': None,
        'language': None,
        'fresh': False
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='caching_6.cache_lookup_transcribe_audio', ignore_result=False)
def cache_lookup_audio_transcript_task(self, token):
    db_manager = AudioDBCachingManager()
    existing_list = db_manager.get_details(token, ['transcript_results', 'subtitle_results', 'language'],
                                           using_most_similar=True)
    for existing in existing_list:
        if existing is None:
            continue
        if (existing['transcript_results'] is not None
                and existing['subtitle_results'] is not None
                and existing['language'] is not None):
            print('Returning cached result')
            transcript_results = existing['transcript_results']
            subtitle_results = existing['subtitle_results']
            language_result = existing['language']

            return {
                'transcript_results': transcript_results,
                'subtitle_results': subtitle_results,
                'language': language_result,
                'fresh': False
            }
    return None


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True,
             retry_kwargs={"max_retries": 2}, name='video_2.transcribe', ignore_result=False,
             file_manager=file_management_config, model=transcription_model)
def transcribe_task(self, input_dict, strict_silence=False):
    token = input_dict['token']
    lang = input_dict['language']

    # If the token is null, it means that some error happened in the previous step (e.g. the file didn't exist
    # in language detection)
    if token is None:
        return {
            'transcript_results': None,
            'subtitle_results': None,
            'language': None,
            'fresh': False
        }

    result_dict = self.model.transcribe_audio_whisper(self.file_manager.generate_filepath(token),
                                                      force_lang=lang, verbose=True,
                                                      strict_silence=strict_silence)

    if result_dict is None:
        return {
            'transcript_results': None,
            'subtitle_results': None,
            'language': None,
            'fresh': False
        }

    transcript_results = result_dict['text']
    subtitle_results = result_dict['segments']
    subtitle_results = json.dumps(subtitle_results, ensure_ascii=False)
    language_result = result_dict['language']

    return {
        'transcript_results': transcript_results,
        'subtitle_results': subtitle_results,
        'language': language_result,
        'fresh': True
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.transcribe_callback', ignore_result=False,
             file_manager=file_management_config)
def transcribe_callback_task(self, results, token, force=False):
    if results['fresh']:
        values_dict = {
            'transcript_results': results['transcript_results'],
            'subtitle_results': results['subtitle_results'],
            'language': results['language']
        }

        # Inserting values for original token
        db_manager = AudioDBCachingManager()
        db_manager.insert_or_update_details(
            token, values_dict
        )
        if not force:
            # Inserting the same values for closest token if different than original token
            # Unless force=True, the whole computation happens when the token and its closest
            # neighbor both lack the requested value. As a result, once it's computed, we
            # update both rows with the value, for other tokens that might depend on the
            # aforementioned closest neighbor (i.e. other tokens that share their closest neighbor
            # with the original token here).
            # Happens if the other token has been fingerprinted before being transcribed, or if the two tokens
            # have identical but separate parent videos.
            closest = db_manager.get_closest_match(token)
            if closest is not None and closest != token:
                db_manager.insert_or_update_details(
                    closest, values_dict
                )
    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.clean_up_transcription_object', model=transcription_model, ignore_result=False)
def cleanup_transcription_object_task(self):
    return self.model.unload_model(WHISPER_UNLOAD_WAITING_PERIOD)
