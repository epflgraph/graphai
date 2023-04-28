from celery import group
from fastapi import APIRouter
from fastapi.responses import FileResponse

from graphai.api.schemas.voice import *
from graphai.api.schemas.common import *

from graphai.api.celery_tasks.voice import compute_audio_fingerprint_task, \
    compute_audio_fingerprint_callback_task, audio_fingerprint_find_closest_retrieve_from_db_task, \
    audio_fingerprint_find_closest_parallel_task, audio_fingerprint_find_closest_callback_task, \
    retrieve_audio_fingerprint_callback_task, remove_audio_silence_task, remove_audio_silence_callback_task, \
    detect_language_retrieve_from_db_and_split_task, detect_language_parallel_task, detect_language_callback_task, \
    transcribe_task, transcribe_callback_task, video_dummy
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
    token = data.token
    force = data.force
    remove_silence = data.remove_silence
    threshold = data.threshold
    n_jobs = 8
    min_similarity = 0.8

    if not remove_silence:
        task = (compute_audio_fingerprint_task.s({'fp_token': token}, token, force) |
                compute_audio_fingerprint_callback_task.s(token) |
                audio_fingerprint_find_closest_retrieve_from_db_task.s(token) |
                group(audio_fingerprint_find_closest_parallel_task.s(i, n_jobs, min_similarity) for i in range(n_jobs)) |
                audio_fingerprint_find_closest_callback_task.s(token) |
                retrieve_audio_fingerprint_callback_task.s()
                )
    else:
        task = (remove_audio_silence_task.s(token, force, threshold) |
                remove_audio_silence_callback_task.s(token) |
                compute_audio_fingerprint_task.s(token, force) |
                compute_audio_fingerprint_callback_task.s(token) |
                audio_fingerprint_find_closest_retrieve_from_db_task.s(token) |
                group(audio_fingerprint_find_closest_parallel_task.s(i, n_jobs, min_similarity) for i in range(n_jobs)) |
                audio_fingerprint_find_closest_callback_task.s(token) |
                retrieve_audio_fingerprint_callback_task.s()
                )
    task = task.apply_async(priority=2)
    return {'task_id': task.id}


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
                'fp_nosilence': True if task_results['fp_nosilence'] == 1 else False,
                'successful': task_results['result'] is not None
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)


@router.post('/transcribe', response_model=TaskIDResponse)
async def transcribe(data: AudioTranscriptionRequest):
    token = data.token
    force = data.force
    lang = data.force_lang
    if lang is not None:
        task = (transcribe_task.s({'token':token, 'lang': lang}, force) |
                transcribe_callback_task.s(token))
    else:
        n_divs = 5
        task = (detect_language_retrieve_from_db_and_split_task.s(token, force, n_divs, 30) |
                group(detect_language_parallel_task.s(i) for i in range(n_divs)) |
                detect_language_callback_task.s(token) |
                transcribe_task.s(force) |
                transcribe_callback_task.s(token))
    task = task.apply_async(priority=2)
    return {'task_id': task.id}


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
    print('Detecting language')
    token = data.token
    force = data.force
    n_divs = 5
    task = (detect_language_retrieve_from_db_and_split_task.s(token, force, n_divs, 30) |
            group(detect_language_parallel_task.s(i) for i in range(n_divs)) |
            detect_language_callback_task.s(token)).apply_async(priority=2)
    return {'task_id': task.id}


@router.post('/priority_test')
async def priority_test():
    print('launching a dummy')
    task = group(video_dummy.s() for i in range(8)).apply_async(priority=2)
    return {'id': task.id}


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
