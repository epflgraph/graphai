from celery import chain

from graphai.celery.translation.tasks import (
    translate_text_task,
    translate_text_callback_task,
    translate_text_return_list_callback_task,
    detect_text_language_task,
    cache_lookup_translation_text_fingerprint_task,
    cache_lookup_translate_text_task,
    compute_translation_text_fingerprint_task,
    compute_translation_text_fingerprint_callback_task,
    cache_lookup_translation_text_using_fingerprint_task
)

from graphai.celery.common.jobs import (
    direct_lookup_generic_job,
    DEFAULT_TIMEOUT
)

from graphai.core.translation.text_utils import generate_translation_text_token
from graphai.core.common.common_utils import convert_list_to_text


def get_translation_text_fingerprint_chain_list(token, text, src, tgt):
    # The tasks are fingerprinting and callback. The callback contains a lookup as well.
    task_list = [
        compute_translation_text_fingerprint_task.s(token, text),
        compute_translation_text_fingerprint_callback_task.s(text, src, tgt)
    ]
    return task_list


def fingerprint_lookup_job(token, return_results=False):
    return direct_lookup_generic_job(cache_lookup_translation_text_fingerprint_task, token,
                                     return_results, DEFAULT_TIMEOUT)


def fingerprint_compute_job(token, text, src, tgt, asynchronous=True):
    task_list = get_translation_text_fingerprint_chain_list(token, text, src, tgt)
    task = chain(task_list)
    task = task.apply_async(priority=6)
    task_id = task.id
    if asynchronous:
        return task_id
    else:
        results = task.get(timeout=20)
        if results is not None:
            return results
        else:
            return None


def fingerprint_job(text, src, tgt, force):
    token = generate_translation_text_token(text, src, tgt)
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
    text = convert_list_to_text(text)
    return fingerprint_compute_job(token, text, src, tgt, asynchronous=True)


def translation_job(text, src, tgt, force, no_cache=False, skip_sentence_segmentation=False):
    token = generate_translation_text_token(text, src, tgt)
    return_list = isinstance(text, list)
    text = convert_list_to_text(text)
    ##########################
    # Translation cache lookup
    ##########################
    if not force and not no_cache:
        direct_lookup_task_id = direct_lookup_generic_job(cache_lookup_translate_text_task, token,
                                                          False, DEFAULT_TIMEOUT,
                                                          return_list)
        if direct_lookup_task_id is not None:
            return direct_lookup_task_id

    ####################################
    # Fingerprinting and fp-based lookup
    ####################################
    if not no_cache:
        current_fingerprint = None
        # Fingerprint cache lookup
        direct_fingerprint_lookup_results = fingerprint_lookup_job(token, return_results=True)
        if direct_fingerprint_lookup_results is not None:
            current_fingerprint = direct_fingerprint_lookup_results['result']
        # If the fingerprint was not cached, fingerprinting
        if current_fingerprint is None:
            fp_computation_results = fingerprint_compute_job(token, text, src, tgt, asynchronous=False)
            current_fingerprint = fp_computation_results['result']
        # Fingerprint-based lookup
        if not force and current_fingerprint is not None:
            fp_based_lookup_task_id = direct_lookup_generic_job(cache_lookup_translation_text_using_fingerprint_task,
                                                                token, False, DEFAULT_TIMEOUT,
                                                                current_fingerprint, src, tgt, return_list)
            if fp_based_lookup_task_id is not None:
                return fp_based_lookup_task_id
    # If we're here, both lookups have failed, so it's time for the actual computation
    #################
    # Computation job
    #################
    if no_cache:
        task_list = [
            translate_text_task.s(text, src, tgt, skip_sentence_segmentation),
            translate_text_return_list_callback_task.s(return_list)
        ]
    else:
        task_list = [
            translate_text_task.s(text, src, tgt, skip_sentence_segmentation),
            translate_text_callback_task.s(token, text, src, tgt, force),
            translate_text_return_list_callback_task.s(return_list)
        ]
    task = chain(task_list)
    task = task.apply_async(priority=6)
    return task.id


def detect_text_language_job(text):
    text = convert_list_to_text(text)
    # The only task is language detection because this task does not go through the caching logic
    task = (detect_text_language_task.s(text)).apply_async(priority=6)
    return task.id
