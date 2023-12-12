import json
from time import sleep
from collections import Counter

from celery import shared_task

from graphai.api.common.video import file_management_config, transcription_model
from graphai.core.common.video import remove_silence_doublesided, perceptual_hash_audio, \
    extract_media_segment
from graphai.core.common.caching import TEMP_SUBFOLDER, AudioDBCachingManager
from graphai.api.celery_tasks.common import fingerprint_lookup_retrieve_from_db, fingerprint_lookup_parallel, \
    fingerprint_lookup_callback


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.audio_silenceremoval', ignore_result=False,
             file_manager=file_management_config)
def remove_audio_silence_task(self, token, force=False, threshold=0.0):
    input_filename_with_path = self.file_manager.generate_filepath(token)
    audio_type = token.split('.')[-1]
    output_suffix = '_nosilence.' + audio_type
    output_token = token + output_suffix
    output_filename_with_path = self.file_manager.generate_filepath(output_token)
    db_manager = AudioDBCachingManager()
    existing = db_manager.get_details(token, cols=['nosilence_token', 'nosilence_duration'])[0]

    # The information on audio file needs to have been inserted into the cache table when
    # the audio was extracted from the video. Therefore, the `existing` row needs to *exist*.
    if existing is None:
        print('Audio file not found!')
        return {
            'fp_token': None,
            'fresh': False,
            'duration': 0.0
        }

    if not force:
        if existing['nosilence_token'] is not None:
            print('Returning cached result')
            return {
                'fp_token': existing['nosilence_token'],
                'fresh': False,
                'duration': existing['nosilence_duration']
            }

    fp_token, duration = remove_silence_doublesided(input_filename_with_path, output_filename_with_path,
                                                    output_token, threshold=threshold)

    if fp_token is None:
        return {
            'fp_token': None,
            'fresh': False,
            'duration': 0.0
        }

    return {
        'fp_token': fp_token,
        'fresh': True,
        'duration': duration
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.audio_silenceremoval_callback', ignore_result=False,
             file_manager=file_management_config)
def remove_audio_silence_callback_task(self, result, audio_token):
    if result['fp_token'] is not None and result['fresh']:
        db_manager = AudioDBCachingManager()
        db_manager.insert_or_update_details(
            audio_token,
            {
                'nosilence_token': result['fp_token'],
                'nosilence_duration': result['duration']
            }
        )

    return result


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.audio_fingerprint', ignore_result=False,
             file_manager=file_management_config)
def compute_audio_fingerprint_task(self, input_dict, audio_token, force=False):
    fp_token = input_dict['fp_token']
    if fp_token is None:
        return {
            'result': None,
            'fp_token': None,
            'perform_lookup': False,
            'fresh': False,
            'duration': 0.0,
            'fp_nosilence': 0
        }

    db_manager = AudioDBCachingManager()
    existing_list = db_manager.get_details(audio_token, cols=['fingerprint', 'duration',
                                                              'nosilence_duration', 'fp_nosilence'],
                                           using_most_similar=True)
    # The information on audio file needs to have been inserted into the cache table when
    # the audio was extracted from the video. Therefore, the `existing` row needs to *exist*,
    # even if its fingerprint and many of its other fields are null.
    if existing_list[0] is None:
        print('Audio file not found!')
        return {
            'result': None,
            'fp_token': None,
            'perform_lookup': False,
            'fresh': False,
            'duration': 0.0,
            'fp_nosilence': 0
        }

    if not force:
        for existing in existing_list:
            if existing is None:
                continue
            if existing['fingerprint'] is not None:
                print('Returning cached result')
                return {
                    'result': existing['fingerprint'],
                    'fp_token': existing['id_token'],
                    'perform_lookup': False,
                    'fresh': False,
                    'duration': existing['duration'] if existing['fp_nosilence'] == 0 else existing[
                        'nosilence_duration'],
                    'fp_nosilence': existing['fp_nosilence']
                }
    fp_token_with_path = self.file_manager.generate_filepath(fp_token)
    fingerprint, decoded = perceptual_hash_audio(fp_token_with_path)
    if fingerprint is None:
        return {
            'result': None,
            'fp_token': None,
            'perform_lookup': False,
            'fresh': False,
            'duration': 0.0,
            'fp_nosilence': 0
        }

    if input_dict.get('duration', None) is None:
        duration = existing_list[0]['duration']
        fp_nosilence = 0
    else:
        duration = input_dict['duration']
        fp_nosilence = 1

    return {
        'result': fingerprint,
        'fp_token': fp_token,
        'perform_lookup': True,
        'fresh': True,
        'duration': duration,
        'fp_nosilence': fp_nosilence
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.audio_fingerprint_callback', ignore_result=False,
             file_manager=file_management_config)
def compute_audio_fingerprint_callback_task(self, results, force=False):
    if results['fresh']:
        token = results['fp_token']
        db_manager = AudioDBCachingManager()
        db_manager.insert_or_update_details(
            token,
            {
                'fingerprint': results['result'],
                'fp_nosilence': results['fp_nosilence']
            }
        )
        if not force:
            closest_token = db_manager.get_closest_match(token)
            # If this token has a closest token, it means that their relationship comes from their parent videos,
            # and that the closest token's fingerprint has not been calculated either (otherwise `fresh` wouldn't be True).
            # In that case, we insert the computed fingerprint for the closest token as well, and then we will perform the
            # fingerprint lookup for that token instead of the one we computed the fingerprint for.
            if closest_token is not None and closest_token != token:
                db_manager.insert_or_update_details(
                    closest_token,
                    {
                        'fingerprint': results['result'],
                        'fp_nosilence': results['fp_nosilence']
                    }
                )
                results['fp_token'] = closest_token
    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.audio_fingerprint_find_closest_retrieve_from_db', ignore_result=False)
def audio_fingerprint_find_closest_retrieve_from_db_task(self, results):
    db_manager = AudioDBCachingManager()
    return fingerprint_lookup_retrieve_from_db(results, db_manager)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.audio_fingerprint_find_closest_parallel', ignore_result=False)
def audio_fingerprint_find_closest_parallel_task(self, input_dict, i, n_total, min_similarity=0.8):
    db_manager = AudioDBCachingManager()
    return fingerprint_lookup_parallel(input_dict, i, n_total, min_similarity, db_manager, data_type='audio')


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.audio_fingerprint_find_closest_callback', ignore_result=False)
def audio_fingerprint_find_closest_callback_task(self, results_list):
    db_manager = AudioDBCachingManager()
    return fingerprint_lookup_callback(results_list, db_manager)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.retrieve_audio_fingerprint_final_callback', ignore_result=False)
def retrieve_audio_fingerprint_callback_task(self, results):
    # Returning the fingerprinting results, which is the part of this task whose results are sent back to the user.
    results_to_return = results['fp_results']
    results_to_return['closest'] = results['closest']
    db_manager = AudioDBCachingManager()

    if results_to_return['closest'] is not None:
        results_to_return['closest_origin'] = db_manager.get_origin(results_to_return['closest'])
    else:
        results_to_return['closest_origin'] = None

    return results_to_return


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True,
             retry_kwargs={"max_retries": 2}, name='video_2.detect_language_retrieve_from_db', ignore_result=False,
             file_manager=file_management_config)
def detect_language_retrieve_from_db_and_split_task(self, token, force=False, n_divs=5, segment_length=30):
    db_manager = AudioDBCachingManager()
    existing_list = db_manager.get_details(token, ['duration', 'language'],
                                           using_most_similar=True)
    if existing_list[0] is None:
        # The token doesn't exist in the cache, so the file doesn't exist
        return {
            'temp_tokens': None,
            'lang': None,
            'fresh': False
        }

    if not force:
        for existing in existing_list:
            if existing is None:
                continue
            if existing['language'] is not None:
                # A recomputation is not forced and a value already exists
                return {
                    'temp_tokens': None,
                    'lang': existing['language'],
                    'fresh': False
                }

    input_filename_with_path = self.file_manager.generate_filepath(token)
    result_tokens = list()

    # Creating `n_divs` segments (of duration `length` each) of the audio file and saving them to the temp subfolder
    for i in range(n_divs):
        current_output_token = token + '_' + str(i) + '_temp.ogg'
        current_output_token_with_path = self.file_manager.generate_filepath(current_output_token,
                                                                             force_dir=TEMP_SUBFOLDER)
        current_result = extract_media_segment(
            input_filename_with_path, current_output_token_with_path, current_output_token,
            start=existing_list[0]['duration'] * i / n_divs, length=segment_length)
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
            'lang': tokens_dict['lang'],
            'fresh': tokens_dict['fresh']
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
    # The logic here is twofold:
    # 1. If the results are not fresh (in which case all fresh flags are False),
    #     they'll be passed through but not reinserted into the database.
    # 2. If the computation has been performed, even a single error in the parallel task
    #     (corresponding to a False value for the fresh flag) will cause a failure.
    if all([x['lang'] is not None for x in results_list]):
        # This indicates success (regardless of freshness)
        languages = [x['lang'] for x in results_list]
        most_common_lang = Counter(languages).most_common(1)[0][0]
        fresh = False
        if all([x['fresh'] for x in results_list]):
            # This indicates freshness
            fresh = True
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
            'fresh': fresh
        }

    return {
        'token': None,
        'language': None,
        'fresh': False
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True,
             retry_kwargs={"max_retries": 2}, name='video_2.transcribe', ignore_result=False,
             file_manager=file_management_config, model=transcription_model)
def transcribe_task(self, input_dict, strict_silence=False, force=False):
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

    if not force:
        # using_most_similar is True here because if the transcript has previously been computed
        # for this token, then the results have also been inserted into the table for its closest
        # neighbor. However, it's also possible that the results have been computed for its closest
        # neighbor but not for itself.
        db_manager = AudioDBCachingManager()
        existing_list = db_manager.get_details(token, ['transcript_results', 'subtitle_results', 'language'],
                                               using_most_similar=True)
        if existing_list[0] is None:
            return {
                'transcript_results': None,
                'subtitle_results': None,
                'language': None,
                'fresh': False
            }

        for existing in existing_list:
            if existing is None:
                continue

            if existing['transcript_results'] is not None and existing['subtitle_results'] is not None and \
                    existing['language'] is not None:
                print('Returning cached result')
                transcript_results = existing['transcript_results']
                subtitle_results = existing['subtitle_results']
                language_result = existing['language']

                return {
                    'transcript_results': transcript_results,
                    'subtitle_results': json.dumps(subtitle_results),
                    'language': language_result,
                    'fresh': False
                }

    if strict_silence:
        no_speech_threshold = 0.5
        logprob_threshold = -0.5
    else:
        no_speech_threshold = 0.6
        logprob_threshold = -1
    input_filename_with_path = self.file_manager.generate_filepath(token)
    result_dict = self.model.transcribe_audio_whisper(input_filename_with_path, force_lang=lang, verbose=True,
                                                      no_speech_threshold=no_speech_threshold,
                                                      logprob_threshold=logprob_threshold)

    if result_dict is None:
        return {
            'transcript_results': None,
            'subtitle_results': None,
            'language': None,
            'fresh': False
        }

    transcript_results = result_dict['text']
    subtitle_results = result_dict['segments']
    language_result = result_dict['language']

    if strict_silence:
        subtitle_results = [
            x for x in subtitle_results
            if x['avg_logprob'] >= -1.0
        ]
        transcript_results = ''.join([x['text'] for x in subtitle_results])

    subtitle_results = json.dumps(subtitle_results)
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
             name='video_2.sleeper', ignore_result=False)
def video_test_task(self):
    sleep(30)
