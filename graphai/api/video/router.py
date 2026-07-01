import time
import uuid
from typing import Optional

from fastapi import APIRouter, Security, Depends, Request
from fastapi.responses import FileResponse
from graphai.api.common.rate_limiter import rate_limiter

from graphai.core.common.logging import get_logger
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

from graphai.api.common.utils import format_api_results

from graphai.celery.video.jobs import (
    retrieve_url_job,
    fingerprint_job,
    extract_audio_job,
    detect_slides_job,
    get_file_job
)

from graphai.api.auth.router import get_current_active_user, get_user_for_rate_limiter
from graphai.api.auth.auth_utils import get_ratelimit_values

from graphai.celery.common.celery_config import get_task_info

logger = get_logger('graphai.api.video')


def _request_id(request: Request) -> str:
    return request.headers.get('x-request-id') or uuid.uuid4().hex[:8]


def _duration_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


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
async def retrieve_file(request: Request, data: RetrieveURLRequest):
    start = time.perf_counter()
    url = data.url
    force = data.force
    is_playlist = data.playlist
    request_id = _request_id(request)
    log = logger.bind(
        endpoint='/video/retrieve_url',
        method=request.method,
        request_id=request_id,
        url=url,
        force=force,
        is_playlist=is_playlist,
    )
    log.info('🎬 Video retrieve_url endpoint invoked')
    task_id = retrieve_url_job(url, force, is_playlist, request_id=request_id)
    log.info('✅ Video retrieve_url job submitted', task_id=task_id, duration_ms=_duration_ms(start))
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
                'successful': task_results['token'] is not None,
                'failure_reason': task_results.get('failure_reason')
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)


@router.post('/calculate_fingerprint', response_model=TaskIDResponse)
async def calculate_video_fingerprint(request: Request, data: VideoFingerprintRequest):
    start = time.perf_counter()
    token = data.token
    force = data.force
    request_id = _request_id(request)
    log = logger.bind(
        endpoint='/video/calculate_fingerprint',
        method=request.method,
        request_id=request_id,
        token=token,
        force=force,
    )
    log.info('🎬 Video calculate_fingerprint endpoint invoked')
    task_id = fingerprint_job(token, force, request_id=request_id)
    log.info('✅ Video calculate_fingerprint job submitted', task_id=task_id, duration_ms=_duration_ms(start))
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
                'successful': task_results['result'] is not None,
                'file_found': task_results.get('file_found', None)
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)


@router.post('/get_file')
async def get_file(request: Request, data: FileRequest):
    start = time.perf_counter()
    token = data.token
    request_id = _request_id(request)
    log = logger.bind(
        endpoint='/video/get_file',
        method=request.method,
        request_id=request_id,
        token=token,
    )
    log.info('🎬 Video get_file endpoint invoked')
    file_path = get_file_job(token, request_id=request_id)
    log.info('✅ Video get_file job completed', file_path=file_path, duration_ms=_duration_ms(start))
    return FileResponse(file_path)


@router.post('/extract_audio', response_model=TaskIDResponse,
             dependencies=[Depends(rate_limiter(get_ratelimit_values()['video']['max_requests'],
                                                get_ratelimit_values()['video']['window'],
                                                user=get_user_for_rate_limiter))])
async def extract_audio(request: Request, data: ExtractAudioRequest):
    start = time.perf_counter()
    token = data.token
    force = data.force
    recalculate = data.recalculate_cached
    request_id = _request_id(request)
    log = logger.bind(
        endpoint='/video/extract_audio',
        method=request.method,
        request_id=request_id,
        token=token,
        force=force,
        recalculate_cached=recalculate,
    )
    log.info('🎵 Video extract_audio endpoint invoked')
    task_id = extract_audio_job(token, force, recalculate, request_id=request_id)
    log.info('✅ Video extract_audio job submitted', task_id=task_id, duration_ms=_duration_ms(start))
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
                'successful': task_results['token'] is not None,
                'file_found': task_results.get('file_found', None)
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)


@router.post('/detect_slides', response_model=TaskIDResponse,
             dependencies=[Depends(rate_limiter(get_ratelimit_values()['video']['max_requests'],
                                                get_ratelimit_values()['video']['window'],
                                                user=get_user_for_rate_limiter))])
async def detect_slides(request: Request, data: DetectSlidesRequest):
    start = time.perf_counter()
    token = data.token
    force = data.force
    recalculate = data.recalculate_cached
    language = data.language
    parameters = data.parameters
    request_id = _request_id(request)
    log = logger.bind(
        endpoint='/video/detect_slides',
        method=request.method,
        request_id=request_id,
        token=token,
        force=force,
        recalculate_cached=recalculate,
        language=language,
        hash_thresh=parameters.hash_thresh,
        multiplier=parameters.multiplier,
        default_threshold=parameters.default_threshold,
        include_first=parameters.include_first,
        include_last=parameters.include_last,
    )
    log.info('🖼️ Video detect_slides endpoint invoked')
    task_id = detect_slides_job(
        token, language, force, recalculate,
        hash_thresh=parameters.hash_thresh,
        multiplier=parameters.multiplier,
        default_threshold=parameters.default_threshold,
        include_first=parameters.include_first,
        include_last=parameters.include_last,
        request_id=request_id,
    )
    log.info('✅ Video detect_slides job submitted', task_id=task_id, duration_ms=_duration_ms(start))
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
                'successful': task_results['slide_tokens'] is not None,
                'file_found': task_results.get('file_found', None)
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)
