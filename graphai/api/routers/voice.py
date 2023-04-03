from fastapi import APIRouter
from fastapi.responses import FileResponse

from graphai.api.schemas.voice import *
from graphai.api.schemas.common import *

from graphai.api.celery_tasks.voice import compute_audio_fingerprint_master, transcribe_master, detect_language_master
from graphai.core.celery_utils.celery_utils import get_task_info


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
    return result


@router.get('/calculate_fingerprint/status/{task_id}', response_model=AudioFingerprintResponse)
async def calculate_audio_fingerprint_status(task_id):
    return get_task_info(task_id)


@router.post('/transcribe', response_model=TaskIDResponse)
async def transcribe(data: AudioTranscriptionRequest):
    result = transcribe_master(data.token, force=data.force,
                                              lang=data.force_lang)
    return result


@router.get('/transcribe/status/{task_id}', response_model=AudioTranscriptionResponse)
async def transcribe_status(task_id):
    return get_task_info(task_id)


@router.post('/detect_language', response_model=TaskIDResponse)
async def transcribe(data: AudioDetectLanguageRequest):
    result = detect_language_master(data.token, force=data.force)
    return result


@router.get('/detect_language/status/{task_id}', response_model=AudioDetectLanguageResponse)
async def transcribe_status(task_id):
    return get_task_info(task_id)