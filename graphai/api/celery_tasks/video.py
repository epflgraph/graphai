import os
import shutil
from itertools import chain

from celery import shared_task

from graphai.core.video.video import (
    retrieve_file_from_url,
    create_filename_using_url_format,
    extract_audio_from_video,
    extract_frames,
    generate_frame_sample_indices,
    compute_ocr_noise_level,
    compute_ocr_threshold,
    compute_video_ocr_transitions,
    check_ocr_and_hash_thresholds,
    generate_random_token,
    generate_symbolic_token,
    read_txt_gz_file,
    generate_audio_token,
    FRAME_FORMAT_PNG,
    TESSERACT_OCR_FORMAT,
    get_file_size
)
from graphai.core.common.fingerprinting import md5_video_or_audio, perceptual_hash_audio, perceptual_hash_image
from graphai.core.interfaces.caching import (
    AudioDBCachingManager,
    SlideDBCachingManager,
    VideoDBCachingManager,
    get_video_token_status,
    get_image_token_status,
    get_audio_token_status
)
from graphai.core.common.common_utils import (
    get_current_datetime,
    copy_file_within_folder
)

from graphai.api.common.video import (
    file_management_config,
    local_ocr_nlp_models
)
from graphai.api.celery_tasks.common import (
    fingerprint_lookup_retrieve_from_db,
    fingerprint_lookup_parallel,
    fingerprint_lookup_callback, fingerprint_lookup_direct
)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='caching_6.cache_lookup_retrieve_url', ignore_result=False,
             file_manager=file_management_config)
def cache_lookup_retrieve_file_from_url_task(self, url):
    db_manager = VideoDBCachingManager()
    existing = db_manager.get_details_using_origin(url, [])

    if existing is not None:
        token = existing[0]['id_token']
        return {
            'token': token,
            'fresh': False,
            'token_status': get_video_token_status(token),
            'token_size': get_file_size(self.file_manager.generate_filepath(token))
        }

    return None


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.retrieve_url', ignore_result=False,
             file_manager=file_management_config)
def retrieve_file_from_url_task(self, url, is_kaltura=True, force_token=None):
    if force_token is not None:
        token = force_token
    else:
        db_manager = VideoDBCachingManager()
        existing = db_manager.get_details_using_origin(url, [])
        if existing is not None:
            # If the cache row already exists, then we don't create a new token, but instead
            # use the id_token of the existing row (we remove the file extension because it will be re-added soon)
            token = existing[0]['id_token'].split('.')[0]
        else:
            # Otherwise, we generate a random token
            token = generate_random_token()
    filename = create_filename_using_url_format(token, url)
    filename_with_path = self.file_manager.generate_filepath(filename)
    results = retrieve_file_from_url(url, filename_with_path, filename, is_kaltura)
    return {
        'token': results,
        'fresh': results is not None,
        'token_size': get_file_size(filename_with_path)
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.retrieve_url_callback', ignore_result=False,
             file_manager=file_management_config)
def retrieve_file_from_url_callback_task(self, results, url):
    if results['fresh']:
        db_manager = VideoDBCachingManager()
        current_datetime = get_current_datetime()
        values = {
            'date_modified': current_datetime,
            'origin_token': url
        }
        if db_manager.get_details(results['token'], [], using_most_similar=False)[0] is None:
            # If the row doesn't already exist in the database, we also set its date_added value
            values.update(
                {
                    'date_added': current_datetime
                }
            )
        db_manager.insert_or_update_details(results['token'], values_to_insert=values)
    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='caching_6.cache_lookup_fingerprint_video', ignore_result=False,
             file_manager=file_management_config)
def cache_lookup_fingerprint_video_task(self, token):
    db_manager = VideoDBCachingManager()
    existing = db_manager.get_details(token, ['fingerprint'])[0]
    if existing is not None and existing['fingerprint'] is not None:
        return {
            'result': existing['fingerprint'],
            'fp_token': existing['id_token'],
            'perform_lookup': False,
            'fresh': False
        }

    return None


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.fingerprint_video', ignore_result=False,
             file_manager=file_management_config)
def compute_video_fingerprint_task(self, results, force=False):
    token = results['token']
    db_manager = VideoDBCachingManager()
    existing = db_manager.get_details(token, ['fingerprint'])[0]
    if not force and existing is not None and existing['fingerprint'] is not None:
        fp = existing['fingerprint']
        fresh = False
        perform_lookup = False
        fp_token = None
    else:
        fp = md5_video_or_audio(self.file_manager.generate_filepath(token), video=True)
        fresh = fp is not None
        perform_lookup = fp is not None
        fp_token = token if fp is not None else None
    return {
        'result': fp,
        'fp_token': fp_token,
        'perform_lookup': perform_lookup,
        'fresh': fresh,
        'original_results': results
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.fingerprint_video_callback', ignore_result=False,
             file_manager=file_management_config)
def compute_video_fingerprint_callback_task(self, results):
    if results['fresh']:
        token = results['fp_token']
        db_manager = VideoDBCachingManager()
        # The video might already have a row in the cache, or may be nonexistent there because it was not
        # retrieved from a URI. If the latter is the case, we add the current datetime to the cache row.
        if db_manager.get_details(token, [])[0] is not None:
            db_manager.insert_or_update_details(token,
                                                {
                                                    'fingerprint': results['result'],
                                                }
                                                )
        else:
            current_datetime = get_current_datetime()
            db_manager.insert_or_update_details(token,
                                                {
                                                    'fingerprint': results['result'],
                                                    'date_added': current_datetime
                                                }
                                                )
    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.video_fingerprint_find_closest_retrieve_from_db', ignore_result=False)
def video_fingerprint_find_closest_retrieve_from_db_task(self, results):
    db_manager = VideoDBCachingManager()
    return fingerprint_lookup_retrieve_from_db(results, db_manager)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.video_fingerprint_find_closest_parallel', ignore_result=False)
def video_fingerprint_find_closest_parallel_task(self, input_dict, i, n_total,
                                                 min_similarity=1):
    db_manager = VideoDBCachingManager()
    return fingerprint_lookup_parallel(input_dict, i, n_total, min_similarity, db_manager, data_type='video')


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.video_fingerprint_find_closest_callback', ignore_result=False)
def video_fingerprint_find_closest_callback_task(self, results_list):
    db_manager = VideoDBCachingManager()
    return fingerprint_lookup_callback(results_list, db_manager)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.retrieve_video_fingerprint_final_callback', ignore_result=False)
def retrieve_video_fingerprint_callback_task(self, results):
    # Returning the fingerprinting results, which is the part of this task whose results are sent back to the user.
    results_to_return = results['fp_results']
    results_to_return['closest'] = results['closest']
    return results_to_return


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.ignore_video_fingerprint_results_callback', ignore_result=False)
def ignore_video_fingerprint_results_callback_task(self, results):
    # Ignoring the fingerprinting results and returning the results relevant to the task chain.
    # Used in tasks like transcription and OCR, where fingerprinting is performed before the task itself, but where
    # the results of the fingerprinting are not returned.

    results_to_return = results['fp_results']['original_results']
    results_to_return['token_status'] = get_video_token_status(results_to_return['token'])
    return results_to_return


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.get_file', ignore_result=False,
             file_manager=file_management_config)
def get_file_task(self, filename):
    return self.file_manager.generate_filepath(filename)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='caching_6.cache_lookup_extract_audio', ignore_result=False,
             file_manager=file_management_config)
def cache_lookup_extract_audio_task(self, token):
    # Here, the caching logic is a bit complicated. The results of audio extraction are cached in the
    # audio tables, whereas the closest-matching video is cached in the video tables. As a result, we
    # need to look for the cached extracted audio of two videos: the provided token and its closest
    # token.
    video_db_manager = VideoDBCachingManager()
    audio_db_manager = AudioDBCachingManager()
    # Retrieving the closest match of the current video
    closest_token = video_db_manager.get_closest_match(token)
    # Looking up the cached audio result of the current video
    existing_own = audio_db_manager.get_details_using_origin(token, cols=['duration'])
    # Looking up the cached audio result of the closest match video (if it's not the same as the current video)
    if closest_token is not None and closest_token != token:
        existing_closest = audio_db_manager.get_details_using_origin(closest_token, cols=['duration'])
    else:
        existing_closest = None
    # We first look at the video's own existing audio, then at that of the closest match because the video's
    # own precomputed audio (if any) takes precedence.
    all_existing = [existing_own, existing_closest]
    for existing in all_existing:
        if existing is not None:
            print('Returning cached result')
            return {
                'token': existing[0]['id_token'],
                'fresh': False,
                'duration': existing[0]['duration'],
                'token_status': get_audio_token_status(existing[0]['id_token'])
            }

    return None


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.extract_audio', ignore_result=False,
             file_manager=file_management_config)
def extract_audio_task(self, token):
    output_token = generate_audio_token(token)
    results, input_duration = extract_audio_from_video(self.file_manager.generate_filepath(token),
                                                       self.file_manager.generate_filepath(output_token),
                                                       output_token)
    if results is None:
        return {
            'token': None,
            'fresh': False,
            'duration': 0.0
        }

    return {
        'token': results,
        'fresh': True,
        'duration': input_duration
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.extract_audio_callback', ignore_result=False,
             file_manager=file_management_config)
def extract_audio_callback_task(self, results, origin_token, force=False):
    if results['fresh']:
        current_datetime = get_current_datetime()
        db_manager = AudioDBCachingManager()
        db_manager.insert_or_update_details(
            results['token'],
            {
                'duration': results['duration'],
                'origin_token': origin_token,
                'date_added': current_datetime
            }
        )
        if not force:
            # If the force flag is False, we may need to propagate the results of this computation to its closest match.
            # The propagation happens if:
            # 1. The token has a closest match (that isn't itself)
            # 2. The closest match does NOT have cached slide results
            video_db_manager = VideoDBCachingManager()
            closest_video_match = video_db_manager.get_closest_match(origin_token)
            if (closest_video_match is not None and closest_video_match != origin_token
                    and db_manager.get_details_using_origin(closest_video_match, []) is None):
                symbolic_token = generate_symbolic_token(closest_video_match, results['token'])
                self.file_manager.create_symlink(self.file_manager.generate_filepath(results['token']), symbolic_token)
                # Everything is the same aside from the id_token, which is the symbolic token, and the origin_token,
                # which is the closest video match.
                db_manager.insert_or_update_details(
                    symbolic_token,
                    {
                        'duration': results['duration'],
                        'origin_token': closest_video_match,
                        'date_added': current_datetime
                    }
                )
                # We make the symlink file the closest match of the main file (to make sure closest match refs flow in
                # the same direction).
                db_manager.insert_or_update_closest_match(results['token'], {
                    'most_similar_token': symbolic_token
                })
    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.reextract_cached_audio', ignore_result=False,
             file_manager=file_management_config)
def reextract_cached_audio_task(self, token):
    video_db_manager = VideoDBCachingManager()
    closest_token = video_db_manager.get_closest_match(token)
    audio_db_manager = AudioDBCachingManager()
    existing_audio_own = audio_db_manager.get_details_using_origin(token, cols=['duration'])
    if closest_token is not None and closest_token != token:
        existing_audio_closest = audio_db_manager.get_details_using_origin(closest_token,
                                                                           cols=['duration'])
    else:
        existing_audio_closest = None
    if existing_audio_own is None and existing_audio_closest is None:
        return {
            'token': None,
            'fresh': False,
            'duration': 0.0
        }
    existing_audio = existing_audio_own if existing_audio_own is not None else existing_audio_closest
    token_to_use_as_name = existing_audio[0]['id_token']
    output_filename_with_path = self.file_manager.generate_filepath(token_to_use_as_name)
    input_filename_with_path = self.file_manager.generate_filepath(token)
    output_filename, _ = extract_audio_from_video(
        input_filename_with_path, output_filename_with_path, token_to_use_as_name
    )
    # If there was an error of any kind (e.g. non-existing video file), the returned token will be None
    if output_filename is None:
        return {
            'token': None,
            'fresh': False,
            'duration': 0.0
        }

    return {
        'token': output_filename,
        'fresh': True,
        'duration': existing_audio[0]['duration']
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.audio_fingerprint', ignore_result=False,
             file_manager=file_management_config)
def compute_audio_fingerprint_task(self, results, force=False):
    token = results['token']
    # Making sure that the cache row for the audio file already exists.
    # This cache row is created when the audio is extracted from its corresponding video, so it must exist!
    # We also need this cache row later in order to be able to return the duration of the audio file.
    db_manager = AudioDBCachingManager()
    existing = db_manager.get_details(token, cols=['fingerprint', 'duration'],
                                      using_most_similar=False)[0]
    if existing is None:
        return {
            'result': None,
            'fp_token': None,
            'perform_lookup': False,
            'fresh': False,
            'duration': 0.0,
            'original_results': results
        }
    if not force and existing['fingerprint'] is not None:
        fp = existing['fingerprint']
        fresh = False
        perform_lookup = False
        fp_token = None
    else:
        fp = perceptual_hash_audio(self.file_manager.generate_filepath(token))
        fresh = fp is not None
        perform_lookup = fp is not None
        fp_token = token if fp is not None else None
    return {
        'result': fp,
        'fp_token': fp_token,
        'perform_lookup': perform_lookup,
        'fresh': fresh,
        'duration': existing['duration'],
        'original_results': results
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


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.ignore_audio_fingerprint_results_callback', ignore_result=False)
def ignore_audio_fingerprint_results_callback_task(self, results):
    # Ignoring the fingerprinting results and returning the results relevant to the task chain.
    # Used in tasks like transcription and OCR, where fingerprinting is performed before the task itself, but where
    # the results of the fingerprinting are not returned.

    results_to_return = results['fp_results']['original_results']
    results_to_return['token_status'] = get_audio_token_status(results_to_return['token'])
    return results_to_return


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='caching_6.cache_lookup_detect_slides', ignore_result=False,
             file_manager=file_management_config)
def cache_lookup_detect_slides_task(self, token):
    video_db_manager = VideoDBCachingManager()
    # Retrieving the closest match of the current video
    closest_token = video_db_manager.get_closest_match(token)
    slide_db_manager = SlideDBCachingManager()
    existing_slides_own = slide_db_manager.get_details_using_origin(token, cols=['slide_number', 'timestamp'])
    if closest_token is not None and closest_token != token:
        existing_slides_closest = slide_db_manager.get_details_using_origin(closest_token,
                                                                            cols=['slide_number', 'timestamp'])
    else:
        existing_slides_closest = None
    # We first look at the video's own existing slides, then at those of the closest match because the video's
    # own precomputed slides (if any) take precedence.
    all_existing = [existing_slides_own, existing_slides_closest]
    for existing_slides in all_existing:
        if existing_slides is not None:
            print('Returning cached result')
            return {
                'fresh': False,
                'slide_tokens': {
                    x['slide_number']: {
                        'token': x['id_token'],
                        'timestamp': int(x['timestamp']),
                        'token_status': get_image_token_status(x['id_token'])
                    }
                    for x in existing_slides
                }
            }

    return None


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.extract_and_sample_frames', ignore_result=False,
             file_manager=file_management_config)
def extract_and_sample_frames_task(self, token):
    # Extracting frames
    print('Extracting frames...')
    output_folder = token + '_all_frames'
    output_folder = extract_frames(self.file_manager.generate_filepath(token),
                                   self.file_manager.generate_filepath(output_folder),
                                   output_folder)
    # If there was an error of any kind (e.g. non-existing video file), the returned token will be None
    if output_folder is None:
        return {
            'result': None,
            'sample_indices': None,
            'fresh': False
        }
    # Generating frame sample indices
    frame_indices = generate_frame_sample_indices(self.file_manager.generate_filepath(output_folder))
    return {
        'result': output_folder,
        'sample_indices': frame_indices,
        'fresh': True
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.noise_level_parallel', ignore_result=False,
             file_manager=file_management_config,
             nlp_model=local_ocr_nlp_models)
def compute_noise_level_parallel_task(self, results, i, n, language=None):
    if not results['fresh']:
        return {
            'result': None,
            'sample_indices': None,
            'noise_level': None,
            'fresh': False
        }

    all_sample_indices = results['sample_indices']
    start_index = int(i * len(all_sample_indices) / n)
    end_index = int((i + 1) * len(all_sample_indices) / n)
    current_sample_indices = all_sample_indices[start_index:end_index]
    noise_level_list = compute_ocr_noise_level(
        self.file_manager.generate_filepath(results['result']),
        current_sample_indices,
        self.nlp_model,
        language=language
    )

    return {
        'result': results['result'],
        'sample_indices': results['sample_indices'],
        'noise_level': noise_level_list,
        'fresh': True
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.noise_level_callback', ignore_result=False)
def compute_noise_threshold_callback_task(self, results, hash_thresh=0.8):
    if not results[0]['fresh']:
        return {
            'result': None,
            'sample_indices': None,
            'threshold': None,
            'fresh': False
        }

    list_of_noise_value_lists = [x['noise_level'] for x in results]
    all_noise_values = list(chain.from_iterable(list_of_noise_value_lists))
    threshold = compute_ocr_threshold(all_noise_values)

    return {
        'result': results[0]['result'],
        'sample_indices': results[0]['sample_indices'],
        'threshold': threshold,
        'hash_threshold': hash_thresh,
        'fresh': True
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.slide_transitions_parallel', ignore_result=False,
             file_manager=file_management_config, nlp_model=local_ocr_nlp_models)
def compute_slide_transitions_parallel_task(self, results, i, n, language=None):
    if not results['fresh']:
        return {
            'result': None,
            'transitions': None,
            'threshold': None,
            'hash_threshold': None,
            'fresh': False
        }

    all_sample_indices = results['sample_indices']
    start_index = int(i * len(all_sample_indices) / n)
    end_index = int((i + 1) * len(all_sample_indices) / n)
    current_sample_indices = all_sample_indices[start_index:end_index]
    slide_transition_list = compute_video_ocr_transitions(
        self.file_manager.generate_filepath(results['result']),
        current_sample_indices,
        results['threshold'],
        results['hash_threshold'],
        self.nlp_model,
        language=language,
        keep_first=True
    )

    return {
        'result': results['result'],
        'transitions': slide_transition_list,
        'threshold': results['threshold'],
        'hash_threshold': results['hash_threshold'],
        'fresh': True
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.slide_transitions_callback', ignore_result=False,
             file_manager=file_management_config, nlp_model=local_ocr_nlp_models)
def compute_slide_transitions_callback_task(self, results, language=None):
    if not results[0]['fresh']:
        return {
            'result': None,
            'slides': None,
            'fresh': False
        }

    # Cleaning up the slides in-between slices
    original_list_of_slide_transition_lists = [x['transitions'] for x in results]
    original_list_of_slide_transition_lists = [x for x in original_list_of_slide_transition_lists if len(x) > 0]
    list_of_slide_transition_lists = list()
    for i in range(len(original_list_of_slide_transition_lists) - 1):
        l1 = original_list_of_slide_transition_lists[i]
        l2 = original_list_of_slide_transition_lists[i + 1]
        t_check, d, s_hash = check_ocr_and_hash_thresholds(self.file_manager.generate_filepath(results[0]['result']),
                                                           l1[-1], l2[0],
                                                           results[0]['threshold'],
                                                           results[0]['hash_threshold'],
                                                           self.nlp_model,
                                                           language)
        if not t_check:
            l1 = l1[:-1]
        if len(l1) > 0:
            list_of_slide_transition_lists.append(l1)

    list_of_slide_transition_lists.append(original_list_of_slide_transition_lists[-1])
    all_transitions = list(chain.from_iterable(list_of_slide_transition_lists))
    # Making doubly sure there are no duplicates
    all_transitions = sorted(list(set(all_transitions)))

    return {
        'result': results[0]['result'],
        'slides': all_transitions,
        'fresh': True
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.detect_slides_callback', ignore_result=False,
             file_manager=file_management_config)
def detect_slides_callback_task(self, results, token, force=False):
    slide_tokens = None
    if results['fresh']:
        db_manager = SlideDBCachingManager()
        #####################################
        # Deleting pre-existing cached slides
        #####################################
        if force:
            # If force=True, then there's the possibility that the cache contains previously-extracted slides.
            # Since the new slides and the old slides may not be 100% identical,
            # the old cache rows need to be deleted first.
            existing_slides_own = db_manager.get_details_using_origin(token, cols=[])
            if existing_slides_own is not None:
                db_manager.delete_cache_rows([x['id_token'] for x in existing_slides_own])

        ###############################################
        # Removing non-slide frames and leftover slides
        ###############################################
        # Delete non-slide frames from the frames directory
        list_of_slides = [(FRAME_FORMAT_PNG) % (x) for x in results['slides']]
        list_of_ocr_results = [(TESSERACT_OCR_FORMAT) % (x) for x in results['slides']]
        base_folder = results['result']
        base_folder_with_path = self.file_manager.generate_filepath(base_folder)
        if self.request.retries <= 1:
            # We only do this on retry #1, because otherwise, the _all_frames folder no longer exists
            for f in os.listdir(base_folder_with_path):
                if f not in list_of_slides and f not in list_of_ocr_results:
                    os.remove(os.path.join(base_folder_with_path, f))
        # Renaming the `all_frames` directory to `slides`
        slides_folder = base_folder.replace('_all_frames', '_slides')
        slides_folder_with_path = self.file_manager.generate_filepath(slides_folder)
        # Make sure the slides and all_frames folders don't both exist. If that is the case, it means that
        # the slides folder is left over from before (because force==True), so we have to delete it (recursively)
        # before we rename _all_frames to _slides.
        if os.path.exists(slides_folder_with_path) and os.path.exists(base_folder_with_path):
            shutil.rmtree(slides_folder_with_path)
        # Now rename _all_frames to _slides
        if os.path.exists(base_folder_with_path):
            os.rename(base_folder_with_path, slides_folder_with_path)
        else:
            # If the _all_frames folder doesn't exist, assert that the slides folder does!
            assert os.path.exists(slides_folder_with_path)

        ####################################
        # Result formatting and DB insertion
        ####################################
        slide_tokens = [os.path.join(slides_folder, s) for s in list_of_slides]
        ocr_tokens = [os.path.join(slides_folder, s) for s in list_of_ocr_results]
        slide_tokens = {i + 1: {'token': slide_tokens[i], 'timestamp': results['slides'][i]}
                        for i in range(len(slide_tokens))}
        ocr_tokens = {i + 1: ocr_tokens[i] for i in range(len(ocr_tokens))}
        current_datetime = get_current_datetime()
        # Inserting fresh results into the database
        for slide_number in slide_tokens:
            db_manager.insert_or_update_details(
                slide_tokens[slide_number]['token'],
                {
                    'origin_token': token,
                    'timestamp': slide_tokens[slide_number]['timestamp'],
                    'slide_number': slide_number,
                    'ocr_tesseract_results': read_txt_gz_file(
                        file_management_config.generate_filepath(ocr_tokens[slide_number])),
                    'date_added': current_datetime
                }
            )
        if not force:
            # If the force flag is False, we may need to propagate the results of this computation to its closest match.
            # The propagation happens if:
            # 1. The token has a closest match (that isn't itself)
            # 2. The closest match does NOT have cached slide results
            video_db_manager = VideoDBCachingManager()
            closest_video_match = video_db_manager.get_closest_match(token)
            if (closest_video_match is not None and closest_video_match != token
                    and db_manager.get_details_using_origin(closest_video_match, []) is None):
                for slide_number in slide_tokens:
                    # For each slide, we get its token (which is the name of its file) and create a new file with a new
                    # token that has a symlink to the actual slide file.
                    current_token = slide_tokens[slide_number]['token']
                    symbolic_token = generate_symbolic_token(closest_video_match, current_token)
                    self.file_manager.create_symlink(self.file_manager.generate_filepath(current_token), symbolic_token)
                    # Everything is the same aside from the id_token, which is the symbolic token, and the origin_token,
                    # which is the closest video match.
                    db_manager.insert_or_update_details(
                        symbolic_token,
                        {
                            'origin_token': closest_video_match,
                            'timestamp': slide_tokens[slide_number]['timestamp'],
                            'slide_number': slide_number,
                            'ocr_tesseract_results': read_txt_gz_file(
                                file_management_config.generate_filepath(ocr_tokens[slide_number])),
                            'date_added': current_datetime
                        }
                    )
                    # We make the symlink file the closest match of the main file (to make sure
                    # closest match refs flow in the same direction).
                    db_manager.insert_or_update_closest_match(current_token, {
                        'most_similar_token': symbolic_token
                    })
    return {
        'slide_tokens': slide_tokens,
        'fresh': results['fresh']
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.reextract_cached_slides', ignore_result=False,
             file_manager=file_management_config)
def reextract_cached_slides_task(self, token):
    video_db_manager = VideoDBCachingManager()
    closest_token = video_db_manager.get_closest_match(token)
    slide_db_manager = SlideDBCachingManager()
    existing_slides_own = slide_db_manager.get_details_using_origin(token, cols=['slide_number', 'timestamp'])
    if closest_token is not None and closest_token != token:
        existing_slides_closest = slide_db_manager.get_details_using_origin(closest_token,
                                                                            cols=['slide_number', 'timestamp'])
    else:
        existing_slides_closest = None
    if existing_slides_own is None and existing_slides_closest is None:
        return {
            'slide_tokens': None,
            'fresh': False
        }
    existing_slides = existing_slides_own if existing_slides_own is not None else existing_slides_closest
    output_folder = existing_slides[0]['id_token'].split('/')[0]
    timestamps_to_keep = sorted([x['timestamp'] for x in existing_slides])
    output_folder_with_path = self.file_manager.generate_filepath(output_folder)
    # If the slides folder already exists, it needs to be deleted recursively
    if os.path.exists(output_folder_with_path):
        shutil.rmtree(output_folder_with_path)
    output_folder = extract_frames(self.file_manager.generate_filepath(token),
                                   output_folder_with_path,
                                   output_folder)
    # If there was an error of any kind (e.g. non-existing video file), the returned token will be None
    if output_folder is None:
        return {
            'slide_tokens': None,
            'fresh': False
        }
    slides_to_timestamps = {FRAME_FORMAT_PNG % x: x for x in timestamps_to_keep}
    # Fix rounding errors in frame extraction
    list_of_extracted_frames = os.listdir(output_folder_with_path)
    slides_not_in_frames = set(slides_to_timestamps.keys()) - set(list_of_extracted_frames)
    timestamps_not_in_frames = [slides_to_timestamps[x] for x in slides_not_in_frames]
    for timestamp in timestamps_not_in_frames:
        if timestamp == 0:
            copy_file_within_folder(output_folder_with_path,
                                    FRAME_FORMAT_PNG % (timestamp + 1),
                                    FRAME_FORMAT_PNG % timestamp)
        if timestamp == max(timestamps_to_keep):
            copy_file_within_folder(output_folder_with_path,
                                    max(list_of_extracted_frames),
                                    FRAME_FORMAT_PNG % timestamp)
    # Remove unused frames
    for f in list_of_extracted_frames:
        if f not in slides_to_timestamps:
            os.remove(os.path.join(output_folder_with_path, f))
    slide_tokens = [os.path.join(output_folder, s) for s in slides_to_timestamps]
    slide_tokens = {i + 1: {'token': slide_tokens[i], 'timestamp': timestamps_to_keep[i]}
                    for i in range(len(slide_tokens))}
    return {
        'slide_tokens': slide_tokens,
        'fresh': True
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.slide_fingerprint', ignore_result=False,
             file_manager=file_management_config)
def compute_slide_fingerprint_task(self, token):
    # Making sure the slide's cache row exists, because otherwise, the operation should be cancelled!
    db_manager = SlideDBCachingManager()
    existing_slide_list = db_manager.get_details(token, cols=[], using_most_similar=False)
    if existing_slide_list[0] is None:
        return {
            'result': None,
            'fp_token': None,
            'perform_lookup': False,
            'fresh': False
        }

    fingerprint = perceptual_hash_image(self.file_manager.generate_filepath(token))
    if fingerprint is None:
        return {
            'result': None,
            'fp_token': None,
            'perform_lookup': False,
            'fresh': False
        }

    return {
        'result': fingerprint,
        'fp_token': token,
        'perform_lookup': True,
        'fresh': True
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.slide_set_fingerprint', ignore_result=False,
             file_manager=file_management_config)
def compute_slide_set_fingerprint_task(self, results, origin_token):
    # Making sure the cache rows exist, because otherwise, the operation should be cancelled!
    db_manager = SlideDBCachingManager()
    existing_slide_list = db_manager.get_details_using_origin(origin_token, cols=['fingerprint'])
    if existing_slide_list is None:
        return {
            'result': None,
            'fp_token': None,
            'perform_lookup': False,
            'fresh': False,
            'original_results': results
        }
    if all(existing_slide['fingerprint'] is not None for existing_slide in existing_slide_list):
        return {
            'result': None,
            'fp_token': None,
            'perform_lookup': False,
            'fresh': False,
            'original_results': results
        }
    tokens = [existing_slide['id_token'] for existing_slide in existing_slide_list
              if existing_slide['fingerprint'] is None]
    fingerprints = [perceptual_hash_image(self.file_manager.generate_filepath(token)) for token in tokens]
    if any(fp is None for fp in fingerprints):
        return {
            'result': None,
            'fp_token': None,
            'perform_lookup': False,
            'fresh': False,
            'original_results': results
        }

    return {
        'result': fingerprints,
        'fp_token': tokens,
        'perform_lookup': True,
        'fresh': True,
        'original_results': results
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.slide_fingerprint_callback', ignore_result=False)
def compute_slide_fingerprint_callback_task(self, results, force=False):
    if results['fresh']:
        tokens = results['fp_token']
        fp_results = results['result']
        if not isinstance(tokens, list):
            tokens = [tokens]
            fp_results = [fp_results]
        fp_tokens_to_pass_on = list()
        for i in range(len(tokens)):
            token = tokens[i]
            current_fp_result = fp_results[i]
            db_manager = SlideDBCachingManager()
            db_manager.insert_or_update_details(
                token,
                {
                    'fingerprint': current_fp_result,
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
                            'fingerprint': current_fp_result,
                        }
                    )
                    fp_tokens_to_pass_on.append(closest_token)
                else:
                    fp_tokens_to_pass_on.append(token)
            else:
                fp_tokens_to_pass_on.append(token)
        # Now we add the correct fp tokens to pass to the fingerprint closest match lookups
        if isinstance(results['fp_token'], list):
            results['fp_token'] = fp_tokens_to_pass_on
        else:
            results['fp_token'] = fp_tokens_to_pass_on[0]
    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.slide_fingerprint_find_closest_retrieve_from_db', ignore_result=False)
def slide_fingerprint_find_closest_retrieve_from_db_task(self, results):
    db_manager = SlideDBCachingManager()
    return fingerprint_lookup_retrieve_from_db(results, db_manager)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.slide_fingerprint_find_closest_parallel', ignore_result=False)
def slide_fingerprint_find_closest_parallel_task(self, input_dict, i, n_total, min_similarity=1):
    db_manager = SlideDBCachingManager()
    return fingerprint_lookup_parallel(input_dict, i, n_total, min_similarity, db_manager, data_type='image')


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.slide_fingerprint_find_closest_direct', ignore_result=False)
def slide_fingerprint_find_closest_direct_task(self, results):
    db_manager = SlideDBCachingManager()
    return fingerprint_lookup_direct(results, db_manager)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.slide_fingerprint_find_closest_callback', ignore_result=False)
def slide_fingerprint_find_closest_callback_task(self, results_list):
    db_manager = SlideDBCachingManager()
    return fingerprint_lookup_callback(results_list, db_manager)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.retrieve_slide_fingerprint_final_callback', ignore_result=False)
def retrieve_slide_fingerprint_callback_task(self, results):
    # Returning the fingerprinting results, which is the part of this task whose results are sent back to the user.
    results_to_return = results['fp_results']
    results_to_return['closest'] = results['closest']
    db_manager = SlideDBCachingManager()

    if results_to_return['closest'] is not None:
        results_to_return['closest_origin'] = db_manager.get_origin(results_to_return['closest'])
    else:
        results_to_return['closest_origin'] = None

    return results_to_return


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.ignore_slide_fingerprint_results_callback', ignore_result=False)
def ignore_slide_fingerprint_results_callback_task(self, results):
    # Ignoring the fingerprinting results and returning the results relevant to the task chain.
    # Used in tasks like transcription and OCR, where fingerprinting is performed before the task itself, but where
    # the results of the fingerprinting are not returned.

    results_to_return = results['fp_results']['original_results']
    slide_tokens = results_to_return['slide_tokens']
    fresh = results_to_return['fresh']
    if slide_tokens is not None:
        for slide_number in slide_tokens:
            slide_tokens[slide_number]['token_status'] = get_image_token_status(slide_tokens[slide_number]['token'])

    return {
        'slide_tokens': slide_tokens,
        'fresh': fresh
    }
