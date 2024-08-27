from celery import group, chain

from graphai.celery.video.jobs import get_audio_fingerprint_chain_list
from graphai.celery.celery_tasks.voice import (
    cache_lookup_audio_fingerprint_task,
    cache_lookup_audio_language_task,
    detect_language_retrieve_from_db_and_split_task,
    detect_language_parallel_task,
    detect_language_callback_task,
    cache_lookup_audio_transcript_task,
    transcribe_task,
    transcribe_callback_task
)
from graphai.celery.celery_jobs.common import direct_lookup_generic_job, DEFAULT_TIMEOUT

DEFAULT_TRANSCRIPT_TIMEOUT = 60


def get_audio_language_detection_task_chain(token, force, n_divs=15, len_segment=30):
    task_list = [
        detect_language_retrieve_from_db_and_split_task.s({'token': token}, n_divs, len_segment),
        group(detect_language_parallel_task.s(i) for i in range(n_divs)),
        detect_language_callback_task.s(token, force)
    ]
    return task_list


def fingerprint_lookup_job(token):
    return direct_lookup_generic_job(cache_lookup_audio_fingerprint_task, token, False, DEFAULT_TIMEOUT)


def language_lookup_job(token, return_results=False):
    return direct_lookup_generic_job(cache_lookup_audio_language_task, token, return_results, DEFAULT_TIMEOUT)


def fingerprint_job(token, force):
    ##############
    # Cache lookup
    ##############
    # First, we do the direct cache lookup using the caching queue, but only if force=False
    if not force:
        direct_lookup_task_id = fingerprint_lookup_job(token)
        if direct_lookup_task_id is not None:
            return direct_lookup_task_id

    #################
    # Computation job
    #################
    task_list = get_audio_fingerprint_chain_list(token, force, ignore_fp_results=False)
    task = chain(task_list)
    task = task.apply_async(priority=2)
    return task.id


def detect_language_job(token, force):
    ##############################
    # Detect language cache lookup
    ##############################
    if not force:
        direct_lookup_task_id = language_lookup_job(token)
        if direct_lookup_task_id is not None:
            return direct_lookup_task_id

    #############
    # Computation
    #############
    task_list = get_audio_language_detection_task_chain(token, force)

    task = chain(task_list)
    task = task.apply_async(priority=2)
    return task.id


def transcribe_job(token, force, lang=None, strict_silence=False):
    #########################
    # Transcribe cache lookup
    #########################
    if not force:
        direct_lookup_task_id = direct_lookup_generic_job(cache_lookup_audio_transcript_task, token,
                                                          False, DEFAULT_TRANSCRIPT_TIMEOUT)
        if direct_lookup_task_id is not None:
            return direct_lookup_task_id

    #######################################
    # Language cache lookup
    #######################################

    # If the language of the transcript has not been forced by the user and there is no forced recompute,
    # we do a language cache lookup too.
    if lang is None and not force:
        direct_language_lookup_task_results = language_lookup_job(token, return_results=True)
        if direct_language_lookup_task_results is not None:
            lang = direct_language_lookup_task_results['language']

    #############
    # Computation
    #############

    if lang is None:
        # If the language is not forced and has not been cached, then we do lang detection.
        task_list = get_audio_language_detection_task_chain(token, force)
        task_list += [
            transcribe_task.s(strict_silence)
        ]
    else:
        task_list = [
            transcribe_task.s({'token': token, 'language': lang}, strict_silence)
        ]

    # Finally we add the transcription callback task
    task_list += [
        transcribe_callback_task.s(token, force)
    ]

    task = chain(task_list)
    task = task.apply_async(priority=2)
    return task.id
