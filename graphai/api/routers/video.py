from fastapi import APIRouter
from fastapi.responses import FileResponse
from celery import group

from graphai.api.schemas.common import TaskIDResponse, FileRequest
from graphai.api.schemas.video import (
    RetrieveURLRequest,
    RetrieveURLResponse,
    ExtractAudioRequest,
    ExtractAudioResponse,
    DetectSlidesRequest,
    DetectSlidesResponse,
)

from graphai.api.celery_tasks.common import format_api_results
from graphai.api.celery_tasks.video import (
    retrieve_file_from_url_task,
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
)
from graphai.core.interfaces.celery_config import get_task_info


# Initialise video router
router = APIRouter(
    prefix='/video',
    tags=['video'],
    responses={404: {'description': 'Not found'}}
)


@router.post('/retrieve_url', response_model=TaskIDResponse)
async def retrieve_file(data: RetrieveURLRequest):
    url = data.url
    is_kaltura = data.kaltura
    # Minimum timeout is 1 minute while maximum is 8 minutes.
    max_timeout = 480
    min_timeout = 60
    timeout = max([data.timeout, min_timeout])
    timeout = min([timeout, max_timeout])
    task = retrieve_file_from_url_task.s(url, is_kaltura, timeout).apply_async(priority=2)
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
                'successful': task_results['token'] is not None
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)


@router.post('/get_file/')
async def get_file(data: FileRequest):
    token = data.token
    return FileResponse(get_file_task.apply_async(args=[token], priority=2).get())


@router.post('/extract_audio', response_model=TaskIDResponse)
async def extract_audio(data: ExtractAudioRequest):
    token = data.token
    force = data.force
    task = (
        extract_audio_task.s(token, force)
        | extract_audio_callback_task.s(token)
    ).apply_async(priority=2)
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
    language = data.language
    n_jobs = 8
    hash_thresh = 0.85
    task = (
        extract_and_sample_frames_task.s(token, force)
        | group(compute_noise_level_parallel_task.s(i, n_jobs, language) for i in range(n_jobs))
        | compute_noise_threshold_callback_task.s(hash_thresh)
        | dummy_task.s()
        | group(compute_slide_transitions_parallel_task.s(i, n_jobs, language) for i in range(n_jobs))
        | compute_slide_transitions_callback_task.s()
        | detect_slides_callback_task.s(token)
    ).apply_async(priority=2)
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
