from celery import shared_task, group, chain
import time
from graphai.api.common.log import log
from graphai.api.common.video import generate_random_token, retrieve_file_from_url, extract_audio_from_video, \
    compute_signature, video_config, compute_video_slides, video_db_manager, detect_audio_format_and_duration
from graphai.core.utils.time.stopwatch import Stopwatch


# A task that will have several instances run in parallel
@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 1},
             name='video.example_parallel', ignore_result=False)
def example_parallel_task(self, x):
    time.sleep(x)
    return True


# A task that acts as a callback after a parallel operation
@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 1},
             name='video.example_callback', ignore_result=False)
def example_callback_task(self, l):
    return all(l)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.retrieve_url', ignore_result=False)
def retrieve_file_from_url_task(self, url, filename):
    filename_with_path = video_config.generate_filename(filename)
    results = retrieve_file_from_url(url, filename_with_path, filename)
    return {'token': results,
            'successful': results is not None}


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.compute_signature', ignore_result=False)
def compute_signature_task(self, filename, force=False):
    results, fresh = compute_signature(filename, force=force)
    return {'token': results,
            'successful': results is not None,
            'fresh': fresh}


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.get_file', ignore_result=False)
def get_file_task(self, filename):
    return video_config.generate_filename(filename)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.extract_audio', ignore_result=False)
def extract_audio_task(self, token, force=False):
    input_filename_with_path = video_config.generate_filename(token)
    output_token, input_duration = detect_audio_format_and_duration(input_filename_with_path, token)
    if output_token is None:
        return {
            'token': None,
            'successful': False,
            'fresh': False,
            'duration': 0.0
        }
    else:
        output_filename_with_path = video_config.generate_filename(output_token)
        if not force:
            existing = video_db_manager.get_audio_details(output_token, cols=['duration'])
        else:
            existing = None
        if existing is not None:
            print('Returning cached result')
            return {
                'token': existing['id_token'],
                'successful': True,
                'fresh': False,
                'duration': existing['duration']
            }
        else:
            results = extract_audio_from_video(input_filename_with_path,
                                                output_filename_with_path,
                                                output_token)
            if results is None:
                return {
                    'token': None,
                    'successful': False,
                    'fresh': False,
                    'duration': 0.0
                }
            return {
                'token': results,
                'successful': True,
                'fresh': True,
                'duration': input_duration
            }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.extract_audio_callback', ignore_result=False)
def extract_audio_callback_task(self, results, origin_token):
    if results['successful'] and results['fresh']:
        video_db_manager.insert_or_update_audio_details(
            results['token'],
            {
                'duration': results['duration'],
                'origin_token': origin_token
            }
        )
    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.detect_slides', ignore_result=False)
def detect_slides_task(self, filename, force=False):
    results, fresh, n_slides, result_filenames = compute_video_slides(filename, force=force)
    return {'token': results,
            'successful': results is not None,
            'fresh': fresh,
            'n_slides': n_slides,
            'files': result_filenames}


# The function that creates and calls the celery task
def celery_multiproc_example_master(data):
    sw = Stopwatch()
    # Note: If you have some large dataset that you don't want to individually copy to every single task,
    # take a look at custom Manager classes based on multiprocessing.managers.BaseManager, which enables you
    # to create one Manager object and share it among all the tasks.
    # Getting the parameters
    foo = data.foo
    bar = data.bar
    log(f'Got input parameters ({foo}, {bar})', sw.delta())
    # Here we run 'bar' instances of the parallel task in a group (which means in parallel), and
    # the results are then collected and fed into the callback task.
    # apply_async() schedules the task and get() blocks on it until completion and returns the results.
    t = (group(example_parallel_task.signature(args=[foo]) for i in range(bar)) |
         example_callback_task.signature(args=[])).apply_async(priority=2).get()
    log(f'Got all results', sw.delta())
    return {'baz': t}


def retrieve_and_generate_token_master(url):
    token = generate_random_token()
    out_filename = token + '.' + url.split('.')[-1]
    task = retrieve_file_from_url_task.apply_async(args=[url, out_filename], priority=2)
    return {'task_id': task.id}


def compute_signature_master(token, force=False):
    task = compute_signature_task.apply_async(args=[token, force],  priority=2)
    return {'task_id': task.id}


def get_file_master(token):
    return get_file_task.apply_async(args=[token], priority=2).get()


def extract_audio_master(token, force=False):
    task = (extract_audio_task.s(token, force) |
            extract_audio_callback_task.s(token)).apply_async(priority=2)
    return {'task_id': task.id}


def detect_slides_master(token, force=False):
    task = detect_slides_task.apply_async(args=[token, force], priority=2)
    return {'task_id': task.id}

