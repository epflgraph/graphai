from celery import shared_task
from graphai.api.common.log import log
from graphai.api.common.video import compute_mpeg7_signature, compute_video_slides, video_config, video_db_manager
from graphai.core.common.video import generate_random_token, retrieve_file_from_url, detect_audio_format_and_duration, \
    extract_audio_from_video


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.retrieve_url', ignore_result=False,
             db_manager=video_db_manager, file_manager=video_config)
def retrieve_file_from_url_task(self, url, filename):
    filename_with_path = self.file_manager.generate_filename(filename)
    results = retrieve_file_from_url(url, filename_with_path, filename)
    return {'token': results,
            'successful': results is not None}


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.compute_signature', ignore_result=False,
             db_manager=video_db_manager, file_manager=video_config)
def compute_signature_task(self, filename, force=False):
    results, fresh = compute_mpeg7_signature(filename, force=force)
    return {'token': results,
            'successful': results is not None,
            'fresh': fresh}


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.get_file', ignore_result=False,
             db_manager=video_db_manager, file_manager=video_config)
def get_file_task(self, filename):
    return self.file_manager.generate_filename(filename)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.extract_audio', ignore_result=False,
             db_manager=video_db_manager, file_manager=video_config)
def extract_audio_task(self, token, force=False):
    input_filename_with_path = self.file_manager.generate_filename(token)
    output_token, input_duration = detect_audio_format_and_duration(input_filename_with_path, token)
    if output_token is None:
        return {
            'token': None,
            'successful': False,
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
            'successful': True,
            'fresh': False,
            'duration': existing['duration']
        }
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
             name='video.extract_audio_callback', ignore_result=False,
             db_manager=video_db_manager, file_manager=video_config)
def extract_audio_callback_task(self, results, origin_token):
    if results['successful'] and results['fresh']:
        self.db_manager.insert_or_update_details(
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

