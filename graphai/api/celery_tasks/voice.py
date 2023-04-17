import json

from celery import shared_task, chain, group, Task
from graphai.api.common.video import audio_db_manager, file_management_config, transcription_model
from graphai.core.common.video import remove_silence_doublesided, perceptual_hash_audio, \
    find_closest_audio_fingerprint, write_text_file, read_text_file, read_json_file, extract_audio_segment
from graphai.core.common.caching import TEMP_SUBFOLDER
from collections import Counter


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.audio_silenceremoval', ignore_result=False,
             db_manager=audio_db_manager, file_manager=file_management_config)
def remove_audio_silence_task(self, token, force=False, threshold=0.0):
    input_filename_with_path = self.file_manager.generate_filepath(token)
    audio_type = token.split('.')[-1]
    output_suffix = '_nosilence.' + audio_type
    output_token = token + output_suffix
    output_filename_with_path = self.file_manager.generate_filepath(output_token)
    existing = self.db_manager.get_details(token, cols=['nosilence_token', 'nosilence_duration'])
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
             name='video.audio_silenceremoval_callback', ignore_result=False,
             db_manager=audio_db_manager, file_manager=file_management_config)
def remove_audio_silence_callback_task(self, result, audio_token):
    if result['fp_token'] is not None and result['fresh']:
        self.db_manager.insert_or_update_details(
            audio_token,
            {
                'nosilence_token': result['fp_token'],
                'nosilence_duration': result['duration']
            }
        )
    return result


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.audio_fingerprint', ignore_result=False,
             db_manager=audio_db_manager, file_manager=file_management_config)
def compute_audio_fingerprint_task(self, input_dict, audio_token, force=False):
    fp_token = input_dict['fp_token']
    if fp_token is None:
        return {
            'result': None,
            'fresh': False,
            'duration': 0.0,
            'fp_nosilence': 0
        }
    existing = self.db_manager.get_details(audio_token, cols=['fingerprint', 'duration',
                                                                     'nosilence_duration', 'fp_nosilence'])
    # The information on audio file needs to have been inserted into the cache table when
    # the audio was extracted from the video. Therefore, the `existing` row needs to *exist*,
    # even if its fingerprint and many of its other fields are null.
    if existing is None:
        print('Audio file not found!')
        return {
            'result': None,
            'fresh': False,
            'duration': 0.0,
            'fp_nosilence': 0
        }
    if not force:
        if existing['fingerprint'] is not None:
            print('Returning cached result')
            return {
                'result': existing['fingerprint'],
                'fresh': False,
                'duration': existing['duration'] if existing['fp_nosilence'] == 0 else existing['nosilence_duration'],
                'fp_nosilence': existing['fp_nosilence']
            }
    fp_token_with_path = self.file_manager.generate_filepath(fp_token)
    fingerprint, decoded = perceptual_hash_audio(fp_token_with_path)
    if fingerprint is None:
        return {
            'result': None,
            'fresh': False,
            'duration': 0.0,
            'fp_nosilence': 0
        }
    if input_dict.get('duration', None) is None:
        duration = existing['duration']
        fp_nosilence = 0
    else:
        duration = input_dict['duration']
        fp_nosilence = 1
    return {
        'result': fingerprint,
        'fresh': True,
        'duration': duration,
        'fp_nosilence': fp_nosilence
    }

@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.audio_fingerprint_callback', ignore_result=False,
             db_manager=audio_db_manager, file_manager=file_management_config)
def compute_audio_fingerprint_callback_task(self, results, token):
    if results['result'] is not None and results['fresh']:
        self.db_manager.insert_or_update_details(
            token,
            {
                'fingerprint': results['result'],
                'fp_nosilence': results['fp_nosilence']
            }
        )
    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.audio_fingerprint_find_closest_retrieve_from_db', ignore_result=False,
             db_manager=audio_db_manager, file_manager=file_management_config)
def audio_fingerprint_find_closest_retrieve_from_db_task(self, results, token):
    target_fingerprint = results['result']
    fresh = results['fresh']
    # If the fingerprint computation has been unsuccessful or if cached results are being returned,
    # then there it is not necessary (or even possible, in the former case) to compute the closest
    # audio fingerprint, so we just pass the fingerprinting results along.
    if target_fingerprint is None or not fresh:
        return {
            'target_fp': None,
            'all_tokens': None,
            'all_fingerprints': None,
            'fp_results': results
        }
    # Retrieving all the tokens and their fingerprints. Since at least one audio has been extracted
    # (i.e. this one), this result is never null. In addition, there's at least one non-null fingerprint
    # value (again, for the present audio file).
    tokens_and_fingerprints = self.db_manager.get_all_details(['fingerprint'], using_most_similar=False)
    all_tokens = list(tokens_and_fingerprints.keys())
    all_fingerprints = [tokens_and_fingerprints[key]['fingerprint'] for key in all_tokens]
    # Now we remove the token of the current file itself, because otherwise we'd always get the video itself
    # as the most similar video.
    index_to_remove = all_tokens.index(token)
    del all_tokens[index_to_remove]
    del all_fingerprints[index_to_remove]
    return {
        'target_fp': target_fingerprint,
        'all_tokens': all_tokens,
        'all_fingerprints': all_fingerprints,
        'fp_results': results
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.audio_fingerprint_find_closest_parallel', ignore_result=False,
             db_manager=audio_db_manager, file_manager=file_management_config)
def audio_fingerprint_find_closest_parallel_task(self, input_dict, i, n_total, min_similarity=0.8):
    # This parallel task's "closest fingerprint" result is null if either
    # a) the computation has been disabled (indicated by the token list being null), or
    # b) there are no previous fingerprints (indicated by the list of all tokens being empty)
    if input_dict['all_tokens'] is None or len(input_dict['all_tokens']) == 0:
        return {
            'closest': None,
            'closest_fp': None,
            'max_score': None,
            'fp_results': input_dict['fp_results']
        }
    # Get the total number of tokens and fingerprints
    n_tokens_all = len(input_dict['all_tokens'])
    # Compute the start and end indices
    start_index = int(i / n_total * n_tokens_all)
    end_index = int((i + 1) / n_total * n_tokens_all)
    # Find the closest token for this batch
    # Note: null fingerprint values are automatically handled and don't need to be filtered out.
    closest_token, closest_fingerprint, score = find_closest_audio_fingerprint(
        input_dict['target_fp'],
        input_dict['all_fingerprints'][start_index:end_index],
        input_dict['all_tokens'][start_index:end_index],
        min_similarity=min_similarity
    )
    return {
        'closest': closest_token,
        'closest_fp': closest_fingerprint,
        'max_score': score,
        'fp_results': input_dict['fp_results']
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.audio_fingerprint_find_closest_callback', ignore_result=False,
             db_manager=audio_db_manager, file_manager=file_management_config)
def audio_fingerprint_find_closest_callback_task(self, results_list, original_token):
    # Passing fingerprinting results along if it's been unsuccessful or a cached result has been returned
    # This is essentially the same check as in all the other find_closest tasks.
    fp_results = results_list[0]['fp_results']
    if fp_results['result'] is None or not fp_results['fresh']:
        return{
            'closest': None,
            'score': None,
            'fp_results': fp_results
        }
    results = [(x['closest'], x['closest_fp'], x['max_score']) for x in results_list]
    results = [x for x in results if x[0] is not None]
    # If all results are null and the list is thus empty, then no closest fingerprint has been found,
    # and therefore, the closest token to this one is itself.
    if len(results) == 0:
        closest_token = original_token
        max_score = -1
    else:
        max_score = max([x[2] for x in results])
        closest_token = [x[0] for x in results if x[2] == max_score][0]
        closest_token = self.db_manager.resolve_most_similar_chain(closest_token)
    # Whether the closest token is itself or another token, we store the result in the database.
    self.db_manager.insert_or_update_closest_match(
        original_token,
        {
            'most_similar_token': closest_token
        }
    )
    return {'closest': closest_token, 'score': max_score, 'fp_results': fp_results}


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.retrieve_fingerprint_task_callback', ignore_result=False,
             db_manager=audio_db_manager, file_manager=file_management_config)
def retrieve_fingerprint_callback_task(self, results):
    # Returning the fingerprinting results, which is the part of this task whose results are sent back to the user.
    return results['fp_results']


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True,
             retry_kwargs={"max_retries": 2}, name='video.detect_language_retrieve_from_db', ignore_result=False,
             db_manager=audio_db_manager, file_manager=file_management_config)
def detect_language_retrieve_from_db_and_split_task(self, token, force=False, n_divs=5, segment_length=30):
    existing = self.db_manager.get_details(token, ['duration', 'language'],
                                           using_most_similar=True)
    if existing is None:
        # The token doesn't exist in the cache, so the file doesn't exist
        return {
            'temp_tokens': None,
            'lang': None,
            'fresh': False
        }
    if not force and existing['language'] is not None:
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
        current_result = extract_audio_segment(
            input_filename_with_path, current_output_token_with_path, current_output_token,
            start=existing['duration']*i/n_divs, length=segment_length)
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
             retry_kwargs={"max_retries": 2}, name='video.detect_language_parallel', ignore_result=False,
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
    except:
        return {
            'lang': None,
            'fresh': False
        }
    return {
        'lang': language,
        'fresh': True
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True,
             retry_kwargs={"max_retries": 2}, name='video.detect_language_callback', ignore_result=False,
             db_manager=audio_db_manager, file_manager=file_management_config)
def detect_language_callback_task(self, results_list, token):
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
            values_dict = {
                'language': most_common_lang
            }
            # Inserting values for original token
            self.db_manager.insert_or_update_details(
                token, values_dict
            )
            # Inserting values for the closest neighbor
            closest = self.db_manager.get_closest_match(token)
            if closest is not None and closest != token:
                self.db_manager.insert_or_update_details(
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
             retry_kwargs={"max_retries": 2}, name='video.transcribe', ignore_result=False,
             db_manager=audio_db_manager, file_manager=file_management_config, model=transcription_model)
def transcribe_task(self, input_dict, force=False):
    token = input_dict['token']
    lang = input_dict['language']
    # If the token is null, it means that some error happened in the previous step (e.g. the file didn't exist
    # in language detection)
    if token is None:
        return {
            'transcript_result': None,
            'transcript_token': None,
            'subtitle_result': None,
            'subtitle_token': None,
            'language': None,
            'fresh': False
        }
    if not force:
        # using_most_similar is True here because if the transcript has previously been computed
        # for this token, then the results have also been inserted into the table for its closest
        # neighbor. However, it's also possible that the results have been computed for its closest
        # neighbor but not for itself.
        existing = self.db_manager.get_details(token, ['transcript_token', 'subtitle_token', 'language'],
                                               using_most_similar=True)
        if existing['transcript_token'] is not None and existing['subtitle_token'] is not None and \
                existing['language'] is not None:
            print('Returning cached result')
            transcript_token = existing['transcript_token']
            transcript_result = read_text_file(self.file_manager.generate_filepath(transcript_token))
            subtitle_token = existing['subtitle_token']
            subtitle_result = read_json_file(self.file_manager.generate_filepath(subtitle_token))
            language_result = existing['language']
            return {
                'transcript_result': transcript_result,
                'transcript_token': transcript_token,
                'subtitle_result': json.dumps(subtitle_result),
                'subtitle_token': subtitle_token,
                'language': language_result,
                'fresh': False
            }

    input_filename_with_path = self.file_manager.generate_filepath(token)
    transcript_token = token + '_transcript.txt'
    subtitle_token = token + '_subtitle_segments.json'
    result_dict = \
        self.model.transcribe_audio_whisper(input_filename_with_path, force_lang=lang, verbose=True)
    if result_dict is None:
        return {
            'transcript_result': None,
            'transcript_token': None,
            'subtitle_result': None,
            'subtitle_token': None,
            'language': None,
            'fresh': False
        }
    transcript_result = result_dict['text']
    subtitle_result = json.dumps(result_dict['segments'])
    language_result = result_dict['language']
    transcript_filename_with_path = self.file_manager.generate_filepath(transcript_token)
    subtitle_filename_with_path = self.file_manager.generate_filepath(subtitle_token)
    write_text_file(transcript_filename_with_path, transcript_result)
    write_text_file(subtitle_filename_with_path, subtitle_result)
    return {
        'transcript_result': transcript_result,
        'transcript_token': transcript_token,
        'subtitle_result': subtitle_result,
        'subtitle_token': subtitle_token,
        'language': language_result,
        'fresh': True
    }



@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.transcribe_callback', ignore_result=False,
             db_manager=audio_db_manager, file_manager=file_management_config)
def transcribe_callback_task(self, results, token):
    if results['fresh']:
        values_dict = {
            'transcript_token': results['transcript_token'],
            'subtitle_token': results['subtitle_token'],
            'language': results['language']
        }
        # Inserting values for original token
        self.db_manager.insert_or_update_details(
            token, values_dict
        )
        # Inserting the same values for closest token if different than original token
        # Unless force=True, the whole computation happens when the token and its closest
        # neighbor both lack the requested value. As a result, once it's computed, we
        # update both rows with the value, for other tokens that might depend on the
        # aforementioned closest neighbor (i.e. other tokens that share their closest neighbor
        # with the original token here).
        closest = self.db_manager.get_closest_match(token)
        if closest is not None and closest != token:
            self.db_manager.insert_or_update_details(
                closest, values_dict
            )
    return results


def compute_audio_fingerprint_master(token, force=False, remove_silence=False, threshold=0.0,
                                     min_similarity=0.8, n_jobs=8):
    if not remove_silence:
        task = (compute_audio_fingerprint_task.s({'fp_token': token}, token, force) |
                compute_audio_fingerprint_callback_task.s(token) |
                audio_fingerprint_find_closest_retrieve_from_db_task.s(token) |
                group(audio_fingerprint_find_closest_parallel_task.s(i, n_jobs, min_similarity) for i in range(n_jobs)) |
                audio_fingerprint_find_closest_callback_task.s(token) |
                retrieve_fingerprint_callback_task.s()
                )
    else:
        task = (remove_audio_silence_task.s(token, force, threshold) |
                remove_audio_silence_callback_task.s(token) |
                compute_audio_fingerprint_task.s(token, force) |
                compute_audio_fingerprint_callback_task.s(token) |
                audio_fingerprint_find_closest_retrieve_from_db_task.s(token) |
                group(audio_fingerprint_find_closest_parallel_task.s(i, n_jobs, min_similarity) for i in range(n_jobs)) |
                audio_fingerprint_find_closest_callback_task.s(token) |
                retrieve_fingerprint_callback_task.s()
                )
    task = task.apply_async(priority=2)
    return {'id': task.id}


def detect_language_master(token, force=False):
    n_divs = 5
    task = (detect_language_retrieve_from_db_and_split_task.s(token, force, n_divs, 30) |
            group(detect_language_parallel_task.s(i) for i in range(n_divs)) |
            detect_language_callback_task.s(token)).apply_async(priority=2)
    return {'id': task.id}


def transcribe_master(token, lang=None, force=False):
    if lang is not None:
        task = (transcribe_task.s({'token':token, 'lang': lang}, force) |
                transcribe_callback_task.s(token))
    else:
        n_divs = 5
        task = (detect_language_retrieve_from_db_and_split_task.s(token, force, n_divs, 30) |
                group(detect_language_parallel_task.s(i) for i in range(n_divs)) |
                detect_language_callback_task.s(token) |
                transcribe_task.s(force) |
                transcribe_callback_task.s(token))
    task = task.apply_async(priority=2)
    return {'id': task.id}
