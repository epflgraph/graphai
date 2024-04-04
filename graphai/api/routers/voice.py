from fastapi import APIRouter, Security

from celery import group, chain

from graphai.api.schemas.common import TaskIDResponse
from graphai.api.schemas.voice import (
    AudioFingerprintRequest,
    AudioFingerprintResponse,
    AudioTranscriptionRequest,
    AudioTranscriptionResponse,
    AudioDetectLanguageRequest,
    AudioDetectLanguageResponse,
)
from graphai.api.routers.auth import get_current_active_user

from graphai.api.celery_tasks.common import format_api_results, ignore_fingerprint_results_callback_task
from graphai.api.celery_tasks.voice import (
    compute_audio_fingerprint_task,
    compute_audio_fingerprint_callback_task,
    audio_fingerprint_find_closest_retrieve_from_db_task,
    audio_fingerprint_find_closest_parallel_task,
    audio_fingerprint_find_closest_callback_task,
    retrieve_audio_fingerprint_callback_task,
    remove_audio_silence_task,
    remove_audio_silence_callback_task,
    detect_language_retrieve_from_db_and_split_task,
    detect_language_parallel_task,
    detect_language_callback_task,
    transcribe_task,
    transcribe_callback_task,
    video_test_task,
)
from graphai.core.interfaces.celery_config import get_task_info
from graphai.core.common.caching import FingerprintParameters

# Initialise video router
router = APIRouter(
    prefix='/voice',
    tags=['voice'],
    responses={404: {'description': 'Not found'}},
    dependencies=[Security(get_current_active_user, scopes=['voice'])]
)


def get_audio_fingerprint_chain_list(token, force=False, min_similarity=None, n_jobs=8, remove_silence=False,
                                     threshold=None, ignore_fp_results=False, results_to_return=None):
    # Loading minimum similarity parameter for audio
    if min_similarity is None:
        fp_parameters = FingerprintParameters()
        min_similarity = fp_parameters.get_min_sim_audio()
    # If remove_silence=True, then a silence removal task and its callback are added at the beginning
    # Otherwise, the tasks are the same as any other fingerprinting chain: compute fp and callback, then lookup.
    if not remove_silence or threshold is None:
        task_list = [compute_audio_fingerprint_task.s({'fp_token': token}, token, force)]
    else:
        task_list = [remove_audio_silence_task.s(token, force, threshold),
                     remove_audio_silence_callback_task.s(token),
                     compute_audio_fingerprint_task.s(token, force)]
    task_list += [compute_audio_fingerprint_callback_task.s(force),
                  audio_fingerprint_find_closest_retrieve_from_db_task.s(),
                  group(audio_fingerprint_find_closest_parallel_task.s(i, n_jobs, min_similarity) for i in range(n_jobs)),
                  audio_fingerprint_find_closest_callback_task.s()]
    if ignore_fp_results:
        task_list += [ignore_fingerprint_results_callback_task.s(results_to_return)]
    else:
        task_list += [retrieve_audio_fingerprint_callback_task.s()]
    return task_list


def get_audio_language_detection_task_chain(token, force, n_divs=15, len_segment=30):
    return [
        detect_language_retrieve_from_db_and_split_task.s(force, n_divs, len_segment),
        group(detect_language_parallel_task.s(i) for i in range(n_divs)),
        detect_language_callback_task.s(token, force)
    ]


@router.post('/calculate_fingerprint', response_model=TaskIDResponse)
async def calculate_audio_fingerprint(data: AudioFingerprintRequest):
    token = data.token
    force = data.force
    remove_silence = data.remove_silence
    threshold = data.threshold
    task_list = get_audio_fingerprint_chain_list(token, force, remove_silence=remove_silence, threshold=threshold)
    task = chain(task_list)
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
                'closest_token': task_results['closest'],
                'closest_token_origin': task_results['closest_origin'],
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
    strict_silence = data.strict

    # If the language is already provided, we won't need to detect it. Otherwise, detection tasks are added.
    # Fingerprinting is always performed but with force=False, regardless of the provided force flag.
    # The tasks are transcription and its callback
    task_list = get_audio_fingerprint_chain_list(token, False, ignore_fp_results=True,
                                                 results_to_return={'token': token, 'language': lang})
    if lang is None:
        task_list += get_audio_language_detection_task_chain(token, force)
    task_list += [
        transcribe_task.s(strict_silence, force),
        transcribe_callback_task.s(token, force)
    ]
    task = chain(task_list)

    task = task.apply_async(priority=2)
    return {'task_id': task.id}


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


@router.post('/detect_language', response_model=TaskIDResponse)
async def detect_language(data: AudioDetectLanguageRequest):
    print('Detecting language')
    token = data.token
    force = data.force
    # Fingerprinting is always performed but with force=False, regardless of the provided force flag.
    # The tasks are splitting the audio into n_divs segments of 30 seconds each, parallel language detection,
    # and then aggregation and db insertion in the callback. Then the transcription tasks continue.
    task_list = get_audio_fingerprint_chain_list(token, False, ignore_fp_results=True,
                                                 results_to_return={'token': token})
    task_list += get_audio_language_detection_task_chain(token, force)
    task = chain(task_list)
    task = task.apply_async(priority=2)
    return {'task_id': task.id}


@router.post('/priority_test')
async def priority_test():
    print('launching a dummy')
    task = group(video_test_task.s() for i in range(8)).apply_async(priority=2)
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
