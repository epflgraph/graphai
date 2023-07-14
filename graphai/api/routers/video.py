from fastapi import APIRouter
from fastapi.responses import FileResponse
from celery import group, chain

from graphai.api.schemas.common import TaskIDResponse, FileRequest
from graphai.api.schemas.video import (
    RetrieveURLRequest,
    RetrieveURLResponse,
    ExtractAudioRequest,
    ExtractAudioResponse,
    DetectSlidesRequest,
    DetectSlidesResponse,
    VideoFingerprintRequest,
    VideoFingerprintResponse
)

from graphai.api.celery_tasks.common import format_api_results, ignore_fingerprint_results_callback_task
from graphai.api.celery_tasks.video import (
    retrieve_file_from_url_task,
    retrieve_file_from_url_callback_task,
    get_file_task,
    extract_audio_task,
    extract_audio_callback_task,
    extract_and_sample_frames_task,
    compute_noise_level_parallel_task,
    compute_noise_threshold_callback_task,
    compute_slide_transitions_parallel_task,
    compute_slide_transitions_callback_task,
    detect_slides_callback_task,
    dummy_task,
    compute_video_fingerprint_task,
    compute_video_fingerprint_callback_task,
    video_fingerprint_find_closest_retrieve_from_db_task,
    video_fingerprint_find_closest_parallel_task,
    video_fingerprint_find_closest_callback_task,
    retrieve_video_fingerprint_callback_task
)
from graphai.core.interfaces.celery_config import get_task_info
from graphai.core.common.video import FingerprintParameters


# Initialise video router
router = APIRouter(
    prefix='/video',
    tags=['video'],
    responses={404: {'description': 'Not found'}}
)


def get_video_fingerprint_chain_list(token, force, min_similarity=None, n_jobs=8,
                                     ignore_fp_results=False, results_to_return=None):
    # Retrieve minimum similarity parameter for video fingerprints
    if min_similarity is None:
        fp_parameters = FingerprintParameters()
        min_similarity = fp_parameters.get_min_sim_video()
    # The list of tasks involve video fingerprinting and its callback, followed by fingerprint lookup (preprocess,
    # parallel, callback).
    task_list = [
        compute_video_fingerprint_task.s(token, force),
        compute_video_fingerprint_callback_task.s(),
        video_fingerprint_find_closest_retrieve_from_db_task.s(),
        group(video_fingerprint_find_closest_parallel_task.s(i, n_jobs, min_similarity)
              for i in range(n_jobs)),
        video_fingerprint_find_closest_callback_task.s()
    ]
    # If the fingerprinting is part of another endpoint, its results are ignored, otherwise they are returned.
    if ignore_fp_results:
        task_list += [ignore_fingerprint_results_callback_task.s(results_to_return)]
    else:
        task_list += [retrieve_video_fingerprint_callback_task.s()]
    return task_list


@router.post('/retrieve_url', response_model=TaskIDResponse)
async def retrieve_file(data: RetrieveURLRequest):
    # The URL to be retrieved
    url = data.url
    # This flag determines if the URL is that of an m3u8 playlist or a video file (like a .mp4)
    is_playlist = data.playlist
    # Overriding the is_playlist flag if the url ends with m3u8 (playlist) or mp4/mkv/flv/avi/mov (video file)
    if url.endswith('.m3u8'):
        is_playlist = True
    elif any([url.endswith(e) for e in ['.mp4', '.mkv', '.flv', '.avi', '.mov']]):
        is_playlist = False
    # First retrieve the file, and then do the database callback
    task_list = [retrieve_file_from_url_task.s(url, is_playlist, False, None),
                 retrieve_file_from_url_callback_task.s(url)]
    task = chain(task_list)
    task = task.apply_async(priority=2)
    return {'task_id': task.id}


# For each async endpoint, we also have a status endpoint since they have different response models.
@router.get('/retrieve_url/status/{task_id}', response_model=RetrieveURLResponse)
async def get_retrieve_file_status(task_id):
    full_results = get_task_info(task_id)
    task_results = full_results['results']
    if task_results is not None:
        if 'token' in task_results:
            task_results = {
                'token': task_results['token'],
                'fresh': task_results['fresh'],
                'successful': task_results['token'] is not None
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)


@router.post('/calculate_fingerprint', response_model=TaskIDResponse)
async def calculate_fingerprint(data: VideoFingerprintRequest):
    token = data.token
    force = data.force
    # This is the fingerprinting endpoint, so ignore_fp_results is False
    task_list = get_video_fingerprint_chain_list(token, force,
                                                 ignore_fp_results=False)
    task = chain(task_list)
    task = task.apply_async(priority=6)
    return {'task_id': task.id}


@router.get('/calculate_fingerprint/status/{task_id}', response_model=VideoFingerprintResponse)
async def calculate_fingerprint_status(task_id):
    full_results = get_task_info(task_id)
    task_results = full_results['results']
    if task_results is not None:
        if 'result' in task_results:
            task_results = {
                'result': task_results['result'],
                'fresh': task_results['fresh'],
                'closest_token': task_results['closest'],
                'successful': task_results['result'] is not None
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)


@router.post('/get_file')
async def get_file(data: FileRequest):
    token = data.token
    return FileResponse(get_file_task.apply_async(args=[token], priority=2).get())


@router.post('/extract_audio', response_model=TaskIDResponse)
async def extract_audio(data: ExtractAudioRequest):
    token = data.token
    force = data.force
    force_non_self = data.force_non_self
    # Fingerprinting is always performed but with force=False, regardless of the provided force flag.
    task_list = get_video_fingerprint_chain_list(token, force=False,
                                                 ignore_fp_results=True, results_to_return=token)
    task_list += [extract_audio_task.s(force, force_non_self)]
    task_list += [extract_audio_callback_task.s(token, force)]
    task = chain(task_list)
    task = task.apply_async(priority=2)
    return {'task_id': task.id}


@router.get('/extract_audio/status/{task_id}', response_model=ExtractAudioResponse)
async def extract_audio_status(task_id):
    full_results = get_task_info(task_id)
    task_results = full_results['results']
    if task_results is not None:
        if 'token' in task_results:
            task_results = {
                'token': task_results['token'],
                'fresh': task_results['fresh'],
                'duration': task_results['duration'],
                'successful': task_results['token'] is not None
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)


@router.post('/detect_slides', response_model=TaskIDResponse)
async def detect_slides(data: DetectSlidesRequest):
    token = data.token
    force = data.force
    force_non_self = data.force_non_self
    language = data.language
    n_jobs = 8
    # This is the maximum similarity threshold used for image hashes when finding slide transitions.
    hash_thresh = 0.95
    # Fingerprinting is always performed but with force=False, regardless of the provided force flag.
    # Task list involves extracting and sampling frames, parallel noise level comp and its callback, then a dummy task
    # because of celery's need for an additional non-group task in the middle, then slide transition comp and its
    # callback, followed by a final callback that inserts the results into the cache db.
    task_list = get_video_fingerprint_chain_list(token, force=False,
                                                 ignore_fp_results=True, results_to_return=token)
    task_list += [extract_and_sample_frames_task.s(force, force_non_self)]
    task_list += [group(compute_noise_level_parallel_task.s(i, n_jobs, language) for i in range(n_jobs)),
                  compute_noise_threshold_callback_task.s(hash_thresh),
                  dummy_task.s(),
                  group(compute_slide_transitions_parallel_task.s(i, n_jobs, language) for i in range(n_jobs)),
                  compute_slide_transitions_callback_task.s(),
                  detect_slides_callback_task.s(token, force)]
    task = chain(task_list)
    task = task.apply_async(priority=2)
    return {'task_id': task.id}


@router.get('/detect_slides/status/{task_id}', response_model=DetectSlidesResponse)
async def detect_slides_status(task_id):
    full_results = get_task_info(task_id)
    task_results = full_results['results']
    if task_results is not None:
        if 'slide_tokens' in task_results:
            task_results = {
                'slide_tokens': task_results['slide_tokens'],
                'fresh': task_results['fresh'],
                'successful': task_results['slide_tokens'] is not None
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)
