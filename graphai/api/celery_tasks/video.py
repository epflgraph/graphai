import os

from celery import shared_task, group
from graphai.api.common.log import log
from graphai.api.common.video import file_management_config, audio_db_manager, local_ocr_nlp_models
from graphai.core.common.video import generate_random_token, retrieve_file_from_url, detect_audio_format_and_duration, \
    extract_audio_from_video, extract_frames, generate_frame_sample_indices, compute_ocr_noise_level, \
    compute_ocr_threshold, compute_video_ocr_transitions, FRAME_FORMAT
from itertools import chain


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.retrieve_url', ignore_result=False,
             db_manager=audio_db_manager, file_manager=file_management_config)
def retrieve_file_from_url_task(self, url, filename):
    filename_with_path = self.file_manager.generate_filename(filename)
    results = retrieve_file_from_url(url, filename_with_path, filename)
    return {'token': results}


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.get_file', ignore_result=False,
             db_manager=audio_db_manager, file_manager=file_management_config)
def get_file_task(self, filename):
    return self.file_manager.generate_filename(filename)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.extract_audio', ignore_result=False,
             db_manager=audio_db_manager, file_manager=file_management_config)
def extract_audio_task(self, token, force=False):
    input_filename_with_path = self.file_manager.generate_filename(token)
    output_token, input_duration = detect_audio_format_and_duration(input_filename_with_path, token)
    if output_token is None:
        return {
            'token': None,
            'fresh': False,
            'duration': 0.0
        }

    output_filename_with_path = self.file_manager.generate_filename(output_token)
    # Here, the existing row can be None because the row is inserted into the table
    # only after extracting the audio from the video.
    if not force:
        existing = self.db_manager.get_details(output_token, cols=['duration'])
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
             name='video.extract_audio_callback', ignore_result=False,
             db_manager=audio_db_manager, file_manager=file_management_config)
def extract_audio_callback_task(self, results, origin_token):
    if results['fresh']:
        self.db_manager.insert_or_update_details(
            results['token'],
            {
                'duration': results['duration'],
                'origin_token': origin_token
            }
        )
    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.extract_and_sample_frames', ignore_result=False,
             db_manager=None, file_manager=file_management_config)
def extract_and_sample_frames_task(self, token, force=False):
    # TODO do the cache check here and populate the "slide_tokens" field of the result with them, if available
    # Extracting frames
    input_filename_with_path = self.file_manager.generate_filename(token)
    output_folder = token + '_all_frames'
    output_folder_with_path = self.file_manager.generate_filename(output_folder)
    output_folder = extract_frames(input_filename_with_path, output_folder_with_path, output_folder)
    if output_folder is None:
        return {
            'result': None,
            'sample_indices': None,
            'fresh': False,
            'slide_tokens': None
        }
    # Generating frame sample indices
    frame_indices = generate_frame_sample_indices(self.file_manager.generate_filename(output_folder))
    return {
        'result': output_folder,
        'sample_indices': frame_indices,
        'fresh': True,
        'slide_tokens': None
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.noise_level_parallel', ignore_result=False,
             db_manager=None, file_manager=file_management_config,
             nlp_model=local_ocr_nlp_models)
def compute_noise_level_parallel_task(self, results, i, n, language=None):
    if results['result'] is None:
        return {
            'result': None,
            'sample_indices': None,
            'noise_level': None,
            'fresh': False,
            'slide_tokens': None
        }
    all_sample_indices = results['sample_indices']
    start_index = int(i*len(all_sample_indices)/n)
    end_index = int((i+1)*len(all_sample_indices)/n)
    current_sample_indices = all_sample_indices[start_index:end_index]
    noise_level_list = \
        compute_ocr_noise_level(self.file_manager.generate_filename(results['result']),
                                current_sample_indices, self.nlp_model.get_nlp_models(), language=language)
    return {
        'result': results['result'],
        'sample_indices': results['sample_indices'],
        'noise_level': noise_level_list,
        'fresh': results['fresh'],
        'slide_tokens': None
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.noise_level_callback', ignore_result=False)
def compute_noise_threshold_callback_task(self, results):
    if results[0]['result'] is None:
        return {
            'result': None,
            'sample_indices': None,
            'threshold': None,
            'fresh': False,
            'slide_tokens': None
        }
    list_of_noise_value_lists = [x['noise_level'] for x in results]
    all_noise_values = list(chain.from_iterable(list_of_noise_value_lists))
    threshold = compute_ocr_threshold(all_noise_values)
    return {
        'result': results[0]['result'],
        'sample_indices': results[0]['sample_indices'],
        'threshold': threshold,
        'fresh': results[0]['fresh'],
        'slide_tokens': None
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.slide_transitions_parallel', ignore_result=False,
             file_manager=file_management_config, nlp_model=local_ocr_nlp_models)
def compute_slide_transitions_parallel_task(self, results, i, n, language=None):
    if results['result'] is None:
        return {
            'result': None,
            'transitions': None,
            'fresh': False,
            'slide_tokens': None
        }
    all_sample_indices = results['sample_indices']
    start_index = int(i*len(all_sample_indices)/n)
    end_index = int((i+1)*len(all_sample_indices)/n)
    current_sample_indices = all_sample_indices[start_index:end_index]
    slide_transition_list = \
        compute_video_ocr_transitions(self.file_manager.generate_filename(results['result']),
                                current_sample_indices, results['threshold'],
                                self.nlp_model.get_nlp_models(), language=language)
    return {
        'result': results['result'],
        'transitions': slide_transition_list,
        'fresh': results['fresh'],
        'slide_tokens': None
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.slide_transitions_callback', ignore_result=False)
def compute_slide_transitions_callback_task(self, results):
    if results[0]['result'] is None:
        return {
            'result': None,
            'slides': None,
            'fresh': False,
            'slide_tokens': None
        }
    list_of_slide_transition_lists = [x['transitions'] for x in results]
    all_transitions = list(chain.from_iterable(list_of_slide_transition_lists))
    # TODO also insert into database and stuff
    return {
        'result': results[0]['result'],
        'slides': all_transitions,
        'fresh': results[0]['fresh'],
        'slide_tokens': None
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.detect_slides_callback', ignore_result=False,
             file_manager=file_management_config)
def detect_slides_callback_task(self, results):
    if results['fresh']:
        # Delete frames aside from slides
        list_of_slides = [(FRAME_FORMAT) % (x) for x in results['slides']]
        base_folder = self.file_manager.generate_filename(results['result'])
        for frame_file in os.listdir(base_folder):
            if frame_file not in list_of_slides:
                os.remove(os.path.join(base_folder, frame_file))
        slides_folder = results['result'].replace('_all_frames', '_slides')
        # Rename `all_frames` folder into `slides`
        os.rename(base_folder,
                  self.file_manager.generate_filename(slides_folder))
        slide_tokens = [os.path.join(slides_folder, s) for s in list_of_slides]
        # TODO Insert results into database
    else:
        slide_tokens = results['slide_tokens']
    return {
        'slide_tokens': slide_tokens,
        'fresh': results['fresh']
    }


def retrieve_and_generate_token_master(url):
    token = generate_random_token()
    out_filename = token + '.' + url.split('.')[-1]
    task = retrieve_file_from_url_task.apply_async(args=[url, out_filename], priority=2)
    return {'id': task.id}


def get_file_master(token):
    return get_file_task.apply_async(args=[token], priority=2).get()


def extract_audio_master(token, force=False):
    task = (extract_audio_task.s(token, force) |
            extract_audio_callback_task.s(token)).apply_async(priority=2)
    return {'id': task.id}


def detect_slides_master(token, force=False, language=None, n_jobs=8):
    task = (extract_and_sample_frames_task.s(token, force) |
            group(compute_noise_level_parallel_task.s(i, n_jobs, language) for i in range(n_jobs)) |
            compute_noise_threshold_callback_task.s() |
            group(compute_slide_transitions_parallel_task.s(i, n_jobs, language) for i in range(n_jobs)) |
            compute_slide_transitions_callback_task.s() |
            detect_slides_callback_task.s()).\
                    apply_async(priority=2)
    return {'id': task.id}


