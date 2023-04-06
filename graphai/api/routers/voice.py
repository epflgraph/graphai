from fastapi import APIRouter
from fastapi.responses import FileResponse

from graphai.api.schemas.voice import *
from graphai.api.schemas.common import *

from graphai.api.celery_tasks.voice import compute_audio_fingerprint_master, transcribe_master, detect_language_master
from graphai.api.celery_tasks.common import format_api_results
from graphai.core.interfaces.celery_config import get_task_info

# Initialise video router
router = APIRouter(
    prefix='/voice',
    tags=['voice'],
    responses={404: {'description': 'Not found'}}
)

@router.post('/calculate_fingerprint', response_model=TaskIDResponse)
async def calculate_audio_fingerprint(data: AudioFingerprintRequest):
    result = compute_audio_fingerprint_master(data.token, force=data.force,
                                              remove_silence=data.remove_silence, threshold=data.threshold)
    return {'task_id': result['id']}


@router.get('/calculate_fingerprint/status/{task_id}', response_model=AudioFingerprintResponse)
async def calculate_audio_fingerprint_status(task_id):
    full_results = get_task_info(task_id)
    task_results = full_results['results']
    if task_results is not None:
        if 'result' in task_results:
            task_results = {
                'result': task_results['result'],
                'fresh': task_results['fresh'],
                'duration': task_results['duration'],
                'fp_nosilence': True if task_results['fp_nosilence'] == 1 else False
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)


@router.post('/transcribe', response_model=TaskIDResponse)
async def transcribe(data: AudioTranscriptionRequest):
    result = transcribe_master(data.token, force=data.force,
                                              lang=data.force_lang)
    return {'task_id': result['id']}


@router.get('/transcribe/status/{task_id}', response_model=AudioTranscriptionResponse)
async def transcribe_status(task_id):
    full_results = get_task_info(task_id)
    task_results = full_results['results']
    if task_results is not None:
        if 'transcript_result' in task_results:
            task_results = {
                'transcript_result': task_results['transcript_result'],
                'subtitle_result': task_results['subtitle_result'],
                'language': task_results['language'],
                'fresh': task_results['fresh']
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)


@router.post('/detect_language', response_model=TaskIDResponse)
async def detect_language(data: AudioDetectLanguageRequest):
    result = detect_language_master(data.token, force=data.force)
    return {'task_id': result['id']}


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