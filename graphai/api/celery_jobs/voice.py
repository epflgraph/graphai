from celery import group, chain

from graphai.api.celery_tasks.common import ignore_fingerprint_results_callback_task
from graphai.api.celery_tasks.voice import (
    cache_lookup_audio_fingerprint_task,
    compute_audio_fingerprint_task,
    compute_audio_fingerprint_callback_task,
    audio_fingerprint_find_closest_retrieve_from_db_task,
    audio_fingerprint_find_closest_parallel_task,
    audio_fingerprint_find_closest_callback_task,
    retrieve_audio_fingerprint_callback_task,
    cache_lookup_audio_language_task,
    detect_language_retrieve_from_db_and_split_task,
    detect_language_parallel_task,
    detect_language_callback_task,
    cache_lookup_audio_transcript_task,
    transcribe_task,
    transcribe_callback_task
)
from graphai.api.celery_jobs.common import direct_lookup_generic_job
from graphai.core.common.caching import FingerprintParameters


def get_audio_fingerprint_chain_list(token, force=False, min_similarity=None, n_jobs=8,
                                     ignore_fp_results=False, results_to_return=None):
    # Loading minimum similarity parameter for audio
    if min_similarity is None:
        fp_parameters = FingerprintParameters()
        min_similarity = fp_parameters.get_min_sim_audio()
    # If remove_silence=True, then a silence removal task and its callback are added at the beginning
    # Otherwise, the tasks are the same as any other fingerprinting chain: compute fp and callback, then lookup.
    task_list = [compute_audio_fingerprint_task.s(token),
                 compute_audio_fingerprint_callback_task.s(force),
                 audio_fingerprint_find_closest_retrieve_from_db_task.s(),
                 group(audio_fingerprint_find_closest_parallel_task.s(i, n_jobs, min_similarity) for i in range(n_jobs)),
                 audio_fingerprint_find_closest_callback_task.s()]
    if ignore_fp_results:
        task_list += [ignore_fingerprint_results_callback_task.s(results_to_return)]
    else:
        task_list += [retrieve_audio_fingerprint_callback_task.s()]
    return task_list


def get_audio_language_detection_task_chain(token, force, n_divs=15, len_segment=30, start_of_chain=False):
    if start_of_chain:
        task_list = [detect_language_retrieve_from_db_and_split_task.s({'token': token}, n_divs, len_segment)]
    else:
        task_list = [detect_language_retrieve_from_db_and_split_task.s(n_divs, len_segment)]
    task_list += [
        group(detect_language_parallel_task.s(i) for i in range(n_divs)),
        detect_language_callback_task.s(token, force)
    ]
    return task_list


def fingerprint_lookup_job(token):
    return direct_lookup_generic_job(cache_lookup_audio_fingerprint_task, token)


def language_lookup_job(token, return_results=False):
    return direct_lookup_generic_job(cache_lookup_audio_language_task, token, return_results)


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

    ##########################
    # Fingerprint cache lookup
    ##########################
    # Now we do a fingerprint lookup to see if we can skip the fingerprinting chain or if it needs to be included.
    # The force flag does not apply to fingerprinting, so we always do a cache lookup first.
    direct_fingerprint_lookup_task_id = fingerprint_lookup_job(token)
    if direct_fingerprint_lookup_task_id is None:
        # If the result is None, it means that there is no cached fingerprint, so we need to do fingerprinting
        task_list = get_audio_fingerprint_chain_list(token, False, ignore_fp_results=True,
                                                     results_to_return={'token': token})
        task_list += get_audio_language_detection_task_chain(token, force, start_of_chain=False)
    else:
        # We end up here if the fingerprint results are already cached.
        task_list = get_audio_language_detection_task_chain(token, force, start_of_chain=True)

    task = chain(task_list)
    task = task.apply_async(priority=2)
    return task.id


def transcribe_job(token, force, lang=None, strict_silence=False):
    #########################
    # Transcribe cache lookup
    #########################
    if not force:
        direct_lookup_task_id = direct_lookup_generic_job(cache_lookup_audio_transcript_task, token)
        if direct_lookup_task_id is not None:
            return direct_lookup_task_id

    task_list = list()
    #######################################
    # Fingerprint and language cache lookup
    #######################################

    # Fingerprint cache lookup
    # Fingerprinting is never forced in non-fingerprinting jobs
    direct_fingerprint_lookup_task_id = fingerprint_lookup_job(token)

    # If the language of the transcript has not been forced by the user and there is no forced recompute,
    # we do a language cache lookup too.
    if lang is None and not force:
        direct_language_lookup_task_results = language_lookup_job(token, return_results=True)
        if direct_language_lookup_task_results is not None:
            lang = direct_language_lookup_task_results['language']

    if direct_fingerprint_lookup_task_id is None:
        # If the fingerprint was not cached
        if lang is None:
            # If the language is not forced and has not been cached, then we do both fingerprinting and lang detection.
            task_list += get_audio_fingerprint_chain_list(token, False, ignore_fp_results=True,
                                                          results_to_return={'token': token})
            task_list += get_audio_language_detection_task_chain(token, force, start_of_chain=False)
        else:
            # Otherwise we only do fingerprinting.
            task_list += get_audio_fingerprint_chain_list(token, False, ignore_fp_results=True,
                                                          results_to_return={'token': token, 'language': lang})
    else:
        # If the fingerprint was already cached
        if lang is None:
            # If the language is not forced and has not been cached, then we do lang detection.
            task_list += get_audio_language_detection_task_chain(token, force, start_of_chain=True)

    # Now it's time to add the transcription tasks
    # If we have tasks in the job before the transcribe task, then we skip the first input of the task
    # Otherwise we have to include the first input as well.

    if len(task_list) > 0:
        task_list += [
            transcribe_task.s(strict_silence)
        ]
    else:
        task_list += [
            transcribe_task.s({'token': token, 'language': lang}, strict_silence)
        ]

    # Finally we add the transcription callback task
    task_list += [
        transcribe_callback_task.s(token, force)
    ]

    task = chain(task_list)
    task = task.apply_async(priority=2)
    return task.id
