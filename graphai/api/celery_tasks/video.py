import os
import shutil

from celery import shared_task

from graphai.api.common.video import file_management_config, local_ocr_nlp_models, \
    transcription_model, translation_models
from graphai.core.common.video import retrieve_file_from_url, retrieve_file_from_kaltura, \
    detect_audio_format_and_duration, extract_audio_from_video, extract_frames, generate_frame_sample_indices, \
    compute_ocr_noise_level, compute_ocr_threshold, compute_video_ocr_transitions, generate_random_token, \
    md5_video_or_audio, generate_symbolic_token, get_current_datetime, FRAME_FORMAT_PNG, TESSERACT_OCR_FORMAT
from graphai.core.common.caching import AudioDBCachingManager, SlideDBCachingManager, VideoDBCachingManager
from itertools import chain
from graphai.api.celery_tasks.common import fingerprint_lookup_retrieve_from_db, \
    fingerprint_lookup_parallel, fingerprint_lookup_callback


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.retrieve_url', ignore_result=False,
             file_manager=file_management_config)
def retrieve_file_from_url_task(self, url, is_kaltura=True, force=False, force_token=None):
    db_manager = VideoDBCachingManager()
    if not force:
        existing = db_manager.get_details_using_origin(url, [])
        if existing is not None:
            return {
                'token': existing[0]['id_token'],
                'fresh': False
            }
    if force_token is None:
        token = generate_random_token()
    else:
        token = force_token
    file_format = url.split('.')[-1].lower()
    if file_format not in ['mp4', 'mkv', 'flv']:
        file_format = 'mp4'
    filename = token + '.' + file_format
    filename_with_path = self.file_manager.generate_filepath(filename)
    if is_kaltura:
        results = retrieve_file_from_kaltura(url, filename_with_path, filename)
    else:
        results = retrieve_file_from_url(url, filename_with_path, filename)
    return {
        'token': results,
        'fresh': results is not None
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.retrieve_url_callback', ignore_result=False,
             file_manager=file_management_config)
def retrieve_file_from_url_callback_task(self, results, url):
    if results['fresh']:
        db_manager = VideoDBCachingManager()
        current_datetime = get_current_datetime()
        db_manager.insert_or_update_details(results['token'],
                                            {
                                                'origin_token': url,
                                                'date_added': current_datetime
        })
    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.fingerprint_video', ignore_result=False,
             file_manager=file_management_config)
def compute_video_fingerprint_task(self, token, force=False):
    db_manager = VideoDBCachingManager()
    existing = db_manager.get_details(token, ['fingerprint'])[0]
    # We don't fail the task if the video isn't already cached because some videos won't come from a URI.
    if not force and existing is not None and existing['fingerprint'] is not None:
        return {
            'result': existing['fingerprint'],
            'fp_token': existing['id_token'],
            'perform_lookup': False,
            'fresh': False
        }
    input_filename_with_path = self.file_manager.generate_filepath(token)
    fp = md5_video_or_audio(input_filename_with_path, video=True)
    return {
        'result': fp,
        'fp_token': token if fp is not None else None,
        'perform_lookup': fp is not None,
        'fresh': fp is not None
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
             name='video_2.get_file', ignore_result=False,
             file_manager=file_management_config)
def get_file_task(self, filename):
    return self.file_manager.generate_filepath(filename)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.extract_audio', ignore_result=False,
             file_manager=file_management_config)
def extract_audio_task(self, token, force=False):
    input_filename_with_path = self.file_manager.generate_filepath(token)
    output_token, input_duration = detect_audio_format_and_duration(input_filename_with_path, token)
    if output_token is None:
        return {
            'token': None,
            'fresh': False,
            'duration': 0.0
        }

    output_filename_with_path = self.file_manager.generate_filepath(output_token)

    # Here, the existing row can be None because the row is inserted into the table
    # only after extracting the audio from the video.
    if not force:
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
                    'duration': existing[0]['duration']
                }

    results = extract_audio_from_video(input_filename_with_path,
                                       output_filename_with_path,
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
        # Inserting the results for the closest match happens only if the results are fresh while the
        # force flag was False. Fresh results with force=False AND a non-null, non-identical closest-match
        # mean that there's another video identical to this one, and NEITHER has had its audio extracted
        # before. That's why we would insert the results for both videos in such a case. If force=True,
        # we don't care about the closest match at all.
        if not force:
            video_db_manager = VideoDBCachingManager()
            closest_video_match = video_db_manager.get_closest_match(origin_token)
            # This only happens if the other video has been fingerprinted before having its audio extracted.
            # A case where this happens is an old video that have had slide detection performed on it,
            # but not audio extraction. When extract_audio is called on a new identical video, the results
            # of audio extraction on the latter need to be inserted for the former as well.
            if closest_video_match is not None and closest_video_match != origin_token:
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
             name='video_2.extract_and_sample_frames', ignore_result=False,
             file_manager=file_management_config)
def extract_and_sample_frames_task(self, token, force=False):
    # Checking for existing cached results
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
            if not force:
                print('Returning cached result')
                return {
                    'result': None,
                    'sample_indices': None,
                    'fresh': False,
                    'slide_tokens': {x['slide_number']: {'token': x['id_token'], 'timestamp': int(x['timestamp'])}
                                     for x in existing_slides}
                }
            else:
                # If force==True, then we need to delete the existing rows in case the results this time are different
                # than they were before, since unlike audio endpoints, there's multiple rows per video here (although
                # the old files are not deleted because there may be symlinks to them).
                # This will require a force-recomputation of every other property that pertains to the video
                # whose slides have been force-recomputed, e.g. fingerprints and text OCRs. Obviously, this only
                # applies to the video itself, and not the closest match found.
                # In general, force is only there for debugging and will not be usable by the end-user.
                if existing_slides[0]['origin_token'] == token:
                    slide_db_manager.delete_cache_rows([x['id_token'] for x in existing_slides])
    # Extracting frames
    print('Extracting frames...')
    input_filename_with_path = self.file_manager.generate_filepath(token)
    output_folder = token + '_all_frames'
    output_folder_with_path = self.file_manager.generate_filepath(output_folder)
    output_folder = extract_frames(input_filename_with_path, output_folder_with_path, output_folder)
    # If there was an error of any kind (e.g. non-existing video file), the returned token will be None
    if output_folder is None:
        return {
            'result': None,
            'sample_indices': None,
            'fresh': False,
            'slide_tokens': None
        }
    # Generating frame sample indices
    frame_indices = generate_frame_sample_indices(self.file_manager.generate_filepath(output_folder))
    return {
        'result': output_folder,
        'sample_indices': frame_indices,
        'fresh': True,
        'slide_tokens': None
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
            'fresh': False,
            'slide_tokens': results['slide_tokens']
        }

    all_sample_indices = results['sample_indices']
    start_index = int(i * len(all_sample_indices) / n)
    end_index = int((i + 1) * len(all_sample_indices) / n)
    current_sample_indices = all_sample_indices[start_index:end_index]
    noise_level_list = compute_ocr_noise_level(
        self.file_manager.generate_filepath(results['result']),
        current_sample_indices,
        self.nlp_model.get_nlp_models(),
        language=language
    )

    return {
        'result': results['result'],
        'sample_indices': results['sample_indices'],
        'noise_level': noise_level_list,
        'fresh': True,
        'slide_tokens': None
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.noise_level_callback', ignore_result=False)
def compute_noise_threshold_callback_task(self, results, hash_thresh=0.8):
    if not results[0]['fresh']:
        return {
            'result': None,
            'sample_indices': None,
            'threshold': None,
            'fresh': False,
            'slide_tokens': results[0]['slide_tokens']
        }

    list_of_noise_value_lists = [x['noise_level'] for x in results]
    all_noise_values = list(chain.from_iterable(list_of_noise_value_lists))
    threshold = compute_ocr_threshold(all_noise_values)

    return {
        'result': results[0]['result'],
        'sample_indices': results[0]['sample_indices'],
        'threshold': threshold,
        'hash_threshold': hash_thresh,
        'fresh': True,
        'slide_tokens': None
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.slide_transitions_parallel', ignore_result=False,
             file_manager=file_management_config, nlp_model=local_ocr_nlp_models)
def compute_slide_transitions_parallel_task(self, results, i, n, language=None):
    if not results['fresh']:
        return {
            'result': None,
            'transitions': None,
            'fresh': False,
            'slide_tokens': results['slide_tokens']
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
        self.nlp_model.get_nlp_models(),
        language=language
    )

    return {
        'result': results['result'],
        'transitions': slide_transition_list,
        'fresh': True,
        'slide_tokens': None
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.slide_transitions_callback', ignore_result=False)
def compute_slide_transitions_callback_task(self, results):
    if not results[0]['fresh']:
        return {
            'result': None,
            'slides': None,
            'fresh': False,
            'slide_tokens': results[0]['slide_tokens']
        }

    list_of_slide_transition_lists = [x['transitions'] for x in results]
    all_transitions = list(chain.from_iterable(list_of_slide_transition_lists))

    return {
        'result': results[0]['result'],
        'slides': all_transitions,
        'fresh': True,
        'slide_tokens': None
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.detect_slides_callback', ignore_result=False,
             file_manager=file_management_config)
def detect_slides_callback_task(self, results, token, force=False):
    if results['fresh']:
        # Delete non-slide frames from the frames directory
        list_of_slides = [(FRAME_FORMAT_PNG) % (x) for x in results['slides']]
        list_of_ocr_results = [(TESSERACT_OCR_FORMAT) % (x) for x in results['slides']]
        base_folder = results['result']
        base_folder_with_path = self.file_manager.generate_filepath(base_folder)
        for f in os.listdir(base_folder_with_path):
            if f not in list_of_slides and f not in list_of_ocr_results:
                os.remove(os.path.join(base_folder_with_path, f))
        # Renaming the `all_frames` directory to `slides`
        slides_folder = base_folder.replace('_all_frames', '_slides')
        slides_folder_with_path = self.file_manager.generate_filepath(slides_folder)
        # Make sure the slides folder doesn't already exist, and recursively delete it if it does (force==True)
        if os.path.exists(slides_folder_with_path) and os.path.isdir(slides_folder_with_path):
            shutil.rmtree(slides_folder_with_path)
        # Now rename _all_frames to _slides
        os.rename(base_folder_with_path,
                  slides_folder_with_path)
        slide_tokens = [os.path.join(slides_folder, s) for s in list_of_slides]
        ocr_tokens = [os.path.join(slides_folder, s) for s in list_of_ocr_results]
        slide_tokens = {i + 1: {'token': slide_tokens[i], 'timestamp': results['slides'][i]}
                        for i in range(len(slide_tokens))}
        ocr_tokens = {i + 1: ocr_tokens[i] for i in range(len(ocr_tokens))}
        current_datetime = get_current_datetime()
        # Inserting fresh results into the database
        db_manager = SlideDBCachingManager()
        for slide_number in slide_tokens:
            db_manager.insert_or_update_details(
                slide_tokens[slide_number]['token'],
                {
                    'origin_token': token,
                    'timestamp': slide_tokens[slide_number]['timestamp'],
                    'slide_number': slide_number,
                    'ocr_tesseract_token': ocr_tokens[slide_number],
                    'date_added': current_datetime
                }
            )
        if not force:
            # Now we check if the video had a closest video match, and if so, insert these results for that
            # video as well, but only if force is False because otherwise we ignore the closest match.
            video_db_manager = VideoDBCachingManager()
            closest_video_match = video_db_manager.get_closest_match(token)
            # This only happens if the other video has been fingerprinted before having its slides extracted.
            # A case where this happens is an old video that have had audio extraction performed on it,
            # but not slide detection. When detect_slides is called on a new identical video, the results
            # of slide detection on the latter need to be inserted for the former as well.
            if closest_video_match is not None and closest_video_match != token:
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
                            'ocr_tesseract_token': ocr_tokens[slide_number],
                            'date_added': current_datetime
                        }
                    )
                    # We make the symlink file the closest match of the main file (to make sure closest match refs flow in
                    # the same direction).
                    db_manager.insert_or_update_closest_match(current_token, {
                        'most_similar_token': symbolic_token
                    })
    else:
        # Getting cached or null results that have been passed along the chain of tasks
        slide_tokens = results['slide_tokens']
    return {
        'slide_tokens': slide_tokens,
        'fresh': results['fresh']
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.dummy_task', ignore_result=False)
def dummy_task(self, results):
    # This task is required for chaining groups due to the peculiarities of celery
    # Whenever there are two groups in one chain of tasks, there need to be at least
    # TWO tasks between them, and this dummy task is simply an f(x)=x function.
    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.init', ignore_result=False,
             transcription_obj=transcription_model,
             nlp_obj=local_ocr_nlp_models,
             translation_obj=translation_models)
def video_init_task(self):
    # This task initialises the video celery worker by loading into memory the transcription and NLP models
    print('Start video_init task')

    print('Loading transcription model...')
    self.transcription_obj.load_model_whisper()

    print('Loading NLP models...')
    self.nlp_obj.get_nlp_models()

    print('Loading translation models...')
    self.translation_obj.load_models()

    print('All video processing objects loaded')
    return True
