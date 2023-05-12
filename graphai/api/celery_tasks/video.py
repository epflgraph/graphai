from datetime import datetime
import os
import shutil

from celery import shared_task

from graphai.api.common.video import file_management_config, local_ocr_nlp_models, \
    transcription_model, translation_models
from graphai.core.common.video import retrieve_file_from_url, retrieve_file_from_kaltura, \
    detect_audio_format_and_duration, extract_audio_from_video, extract_frames, generate_frame_sample_indices, \
    compute_ocr_noise_level, compute_ocr_threshold, compute_video_ocr_transitions, generate_random_token, \
    md5_video_or_audio, FRAME_FORMAT_PNG, TESSERACT_OCR_FORMAT
from graphai.core.common.caching import AudioDBCachingManager, SlideDBCachingManager
from itertools import chain


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.retrieve_url', ignore_result=False,
             file_manager=file_management_config)
def retrieve_file_from_url_task(self, url, is_kaltura=True, timeout=120):
    token = generate_random_token()
    file_format = url.split('.')[-1].lower()
    if file_format not in ['mp4', 'mkv', 'flv']:
        file_format = 'mp4'
    filename = token + '.' + file_format
    filename_with_path = self.file_manager.generate_filepath(filename)
    if is_kaltura:
        results = retrieve_file_from_kaltura(url, filename_with_path, filename, timeout=timeout)
    else:
        results = retrieve_file_from_url(url, filename_with_path, filename)
    return {'token': results}


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
        # we get the first element because in the audio caching table, each origin token has only one row
        db_manager = AudioDBCachingManager()
        existing = db_manager.get_details_using_origin(token, cols=['duration'])
        if existing is not None:
            existing = existing[0]
    else:
        existing = None
    if existing is not None:
        print('Returning cached result')
        return {
            'token': existing['id_token'],
            'fresh': False,
            'duration': existing['duration']
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
def extract_audio_callback_task(self, results, origin_token):
    if results['fresh']:
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        db_manager = AudioDBCachingManager()
        db_manager.insert_or_update_details(
            results['token'],
            {
                'duration': results['duration'],
                'origin_token': origin_token,
                'date_added': current_datetime
            }
        )
    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.extract_and_sample_frames', ignore_result=False,
             file_manager=file_management_config)
def extract_and_sample_frames_task(self, token, force=False):
    # Checking for existing cached results
    db_manager = SlideDBCachingManager()
    existing_slides = db_manager.get_details_using_origin(token, cols=['slide_number'])

    if existing_slides is not None:
        if not force:
            print('Returning cached result')
            return {
                'result': None,
                'sample_indices': None,
                'fresh': False,
                'slide_tokens': {x['slide_number']: x['id_token'] for x in existing_slides}
            }
        else:
            # If force==True, then we need to delete the existing rows in case the results this time are different
            # than they were before, since unlike audio endpoints, there's multiple rows per video here!
            # This will require a force-recomputation of every other property that pertains to the video whose slides
            # have been force-recomputed, e.g. fingerprints and text OCRs.
            # In general, force is only there for debugging and will not be usable by the end-user.
            db_manager.delete_cache_rows([x['id_token'] for x in existing_slides])
    # Extracting frames
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
    start_index = int(i*len(all_sample_indices)/n)
    end_index = int((i+1)*len(all_sample_indices)/n)
    current_sample_indices = all_sample_indices[start_index:end_index]
    noise_level_list = \
        compute_ocr_noise_level(self.file_manager.generate_filepath(results['result']),
                                current_sample_indices, self.nlp_model.get_nlp_models(), language=language)
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
    start_index = int(i*len(all_sample_indices)/n)
    end_index = int((i+1)*len(all_sample_indices)/n)
    current_sample_indices = all_sample_indices[start_index:end_index]
    slide_transition_list = \
        compute_video_ocr_transitions(self.file_manager.generate_filepath(results['result']), current_sample_indices,
                                      results['threshold'], results['hash_threshold'],
                                      self.nlp_model.get_nlp_models(), language=language)
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
def detect_slides_callback_task(self, results, token):
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
        slide_tokens = {i+1:slide_tokens[i] for i in range(len(slide_tokens))}
        ocr_tokens = {i+1:ocr_tokens[i] for i in range(len(ocr_tokens))}
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Inserting fresh results into the database
        for slide_number in slide_tokens:
            db_manager = SlideDBCachingManager()
            db_manager.insert_or_update_details(
                slide_tokens[slide_number],
                {
                    'origin_token': token,
                    'timestamp': results['slides'][slide_number-1],
                    'slide_number': slide_number,
                    'ocr_tesseract_token': ocr_tokens[slide_number],
                    'date_added': current_datetime
                }
            )
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

    print('Loading video processing objects...')
    self.transcription_obj.load_model_whisper()
    print('Transcription model loaded')
    self.nlp_obj.get_nlp_models()
    print('NLP models loaded')
    self.translation_obj.load_models()
    print('Translation models loaded')

    return True