from fastapi import APIRouter, Security, Depends
from fastapi.responses import FileResponse
from fastapi_user_limiter.limiter import rate_limiter

from graphai.api.common.schemas import TaskIDResponse, FileRequest
from graphai.api.video.schemas import (
    RetrieveURLRequest,
    RetrieveURLResponse,
    ExtractAudioRequest,
    ExtractAudioResponse,
    DetectSlidesRequest,
    DetectSlidesResponse,
    VideoFingerprintRequest,
    VideoFingerprintResponse
)

from graphai.celery.common.tasks import format_api_results

from graphai.celery.video.jobs import (
    retrieve_url_job,
    fingerprint_job,
    extract_audio_job,
    detect_slides_job,
    get_file_job
)

from graphai.api.auth.router import get_current_active_user, get_user_for_rate_limiter
from graphai.api.auth.auth_utils import get_ratelimit_values

from graphai.core.interfaces.celery_config import get_task_info

# Initialise video router
router = APIRouter(
    prefix='/video',
    tags=['video'],
    responses={404: {'description': 'Not found'}},
    dependencies=[Security(get_current_active_user, scopes=['video'])]
)


@router.post('/retrieve_url', response_model=TaskIDResponse,
             dependencies=[Depends(rate_limiter(get_ratelimit_values()['video']['max_requests'],
                                                get_ratelimit_values()['video']['window'],
                                                user=get_user_for_rate_limiter))])
async def retrieve_file(data: RetrieveURLRequest):
    # The URL to be retrieved
    url = data.url
    force = data.force
    # This flag determines if the URL is that of an m3u8 playlist or a video file (like a .mp4)
    is_playlist = data.playlist
    task_id = retrieve_url_job(url, force, is_playlist)
    return {'task_id': task_id}


# For each async endpoint, we also have a status endpoint since they have different response models.
@router.get('/retrieve_url/status/{task_id}', response_model=RetrieveURLResponse)
async def get_retrieve_file_status(task_id):
    full_results = get_task_info(task_id)
    task_results = full_results['results']
    if task_results is not None:
        if 'token' in task_results:
            task_results = {
                'token': task_results['token'],
                'token_status': task_results['token_status'],
                'token_size': task_results['token_size'],
                'fresh': task_results['fresh'],
                'successful': task_results['token'] is not None
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)


@router.post('/calculate_fingerprint', response_model=TaskIDResponse)
async def calculate_video_fingerprint(data: VideoFingerprintRequest):
    token = data.token
    force = data.force
    task_id = fingerprint_job(token, force)
    return {'task_id': task_id}


@router.get('/calculate_fingerprint/status/{task_id}', response_model=VideoFingerprintResponse)
async def calculate_video_fingerprint_status(task_id):
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
    return FileResponse(get_file_job(token))


@router.post('/extract_audio', response_model=TaskIDResponse,
             dependencies=[Depends(rate_limiter(get_ratelimit_values()['video']['max_requests'],
                                                get_ratelimit_values()['video']['window'],
                                                user=get_user_for_rate_limiter))])
async def extract_audio(data: ExtractAudioRequest):
    token = data.token
    force = data.force
    recalculate = data.recalculate_cached
    task_id = extract_audio_job(token, force, recalculate)
    return {'task_id': task_id}


@router.get('/extract_audio/status/{task_id}', response_model=ExtractAudioResponse)
async def extract_audio_status(task_id):
    full_results = get_task_info(task_id)
    task_results = full_results['results']
    if task_results is not None:
        if 'token' in task_results:
            task_results = {
                'token': task_results['token'],
                'token_status': task_results['token_status'],
                'fresh': task_results['fresh'],
                'duration': task_results['duration'],
                'successful': task_results['token'] is not None
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)


@router.post('/detect_slides', response_model=TaskIDResponse,
             dependencies=[Depends(rate_limiter(get_ratelimit_values()['video']['max_requests'],
                                                get_ratelimit_values()['video']['window'],
                                                user=get_user_for_rate_limiter))])
async def detect_slides(data: DetectSlidesRequest):
    token = data.token
    force = data.force
    recalculate = data.recalculate_cached
    language = data.language
    task_id = detect_slides_job(token, language, force, recalculate)
    return {'task_id': task_id}


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
