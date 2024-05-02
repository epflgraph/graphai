from fastapi import APIRouter, Security, Depends
from fastapi_user_limiter.limiter import rate_limiter

from graphai.api.celery_jobs.voice import (
    fingerprint_job,
    detect_language_job,
    transcribe_job
)
from graphai.api.schemas.common import TaskIDResponse
from graphai.api.schemas.voice import (
    AudioFingerprintRequest,
    AudioFingerprintResponse,
    AudioTranscriptionRequest,
    AudioTranscriptionResponse,
    AudioDetectLanguageRequest,
    AudioDetectLanguageResponse,
)
from graphai.api.routers.auth import get_current_active_user, get_user_for_rate_limiter
from graphai.api.celery_tasks.common import format_api_results

from graphai.core.interfaces.celery_config import get_task_info

# Initialise video router
router = APIRouter(
    prefix='/voice',
    tags=['voice'],
    responses={404: {'description': 'Not found'}},
    dependencies=[Security(get_current_active_user, scopes=['voice'])]
)


@router.post('/calculate_fingerprint', response_model=TaskIDResponse)
async def calculate_audio_fingerprint(data: AudioFingerprintRequest):
    token = data.token
    force = data.force
    task_id = fingerprint_job(token, force)
    return {'task_id': task_id}


@router.get('/calculate_fingerprint/status/{task_id}', response_model=AudioFingerprintResponse)
async def calculate_audio_fingerprint_status(task_id):
    full_results = get_task_info(task_id)
    task_results = full_results['results']
    if task_results is not None:
        if 'result' in task_results:
            task_results = {
                'result': task_results['result'],
                'fresh': task_results['fresh'],
                'closest_token': task_results['closest'],
                'closest_token_origin': task_results['closest_origin'],
                'duration': task_results['duration'],
                'successful': task_results['result'] is not None
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)


@router.post('/transcribe', response_model=TaskIDResponse,
             dependencies=[Depends(rate_limiter(10, 60, user=get_user_for_rate_limiter))])
async def transcribe(data: AudioTranscriptionRequest):
    token = data.token
    force = data.force
    lang = data.force_lang
    strict_silence = data.strict
    task_id = transcribe_job(token, force, lang, strict_silence)
    return {'task_id': task_id}


@router.get('/transcribe/status/{task_id}', response_model=AudioTranscriptionResponse)
async def transcribe_status(task_id):
    full_results = get_task_info(task_id)
    task_results = full_results['results']
    if task_results is not None:
        if 'transcript_results' in task_results:
            task_results = {
                'transcript_results': task_results['transcript_results'],
                'subtitle_results': task_results['subtitle_results'],
                'language': task_results['language'],
                'fresh': task_results['fresh']
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)


@router.post('/detect_language', response_model=TaskIDResponse,
             dependencies=[Depends(rate_limiter(10, 10, user=get_user_for_rate_limiter))])
async def detect_language(data: AudioDetectLanguageRequest):
    print('Detecting language')
    token = data.token
    force = data.force
    task_id = detect_language_job(token, force)
    return {'task_id': task_id}


@router.get('/detect_language/status/{task_id}', response_model=AudioDetectLanguageResponse)
async def detect_language_status(task_id):
    full_results = get_task_info(task_id)
    task_results = full_results['results']
    if task_results is not None:
        if 'language' in task_results:
            task_results = {
                'language': task_results['language'],
                'fresh': task_results['fresh']
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)
