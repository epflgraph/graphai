from celery import group, chain

from graphai.api.celery_tasks.translation import (
    translate_text_task,
    translate_text_callback_task,
    detect_text_language_task,
    cache_lookup_translation_text_fingerprint_task,
    cache_lookup_translate_text_task,
    compute_translation_text_fingerprint_task,
    compute_translation_text_fingerprint_callback_task,
    translation_text_fingerprint_find_closest_retrieve_from_db_task,
    translation_text_fingerprint_find_closest_parallel_task,
    translation_text_fingerprint_find_closest_direct_task,
    translation_text_fingerprint_find_closest_callback_task,
    translation_retrieve_text_fingerprint_callback_task,
)
from graphai.api.celery_tasks.common import (
    ignore_fingerprint_results_callback_task,
)
from graphai.api.celery_jobs.common import direct_lookup_generic_job

from graphai.core.common.text_utils import (
    generate_src_tgt_dict,
    generate_translation_text_token,
    translation_list_to_text
)

from graphai.core.common.caching import FingerprintParameters


def get_translation_text_fingerprint_chain_list(token, text, src, tgt, min_similarity=None, n_jobs=8,
                                                ignore_fp_results=False):
    # Generating the equality condition dictionary
    equality_conditions = generate_src_tgt_dict(src, tgt)
    # The tasks are fingerprinting and callback, then lookup. The lookup is only among cache rows that satisfy the
    # equality conditions (source and target languages).
    task_list = [
        compute_translation_text_fingerprint_task.s(text),
        compute_translation_text_fingerprint_callback_task.s(text, src, tgt)
    ]
    if ignore_fp_results:
        task_list += [ignore_fingerprint_results_callback_task.s()]
    else:
        task_list += [translation_retrieve_text_fingerprint_callback_task.s()]
    return task_list


def fingerprint_lookup_job(token):
    return direct_lookup_generic_job(cache_lookup_translation_text_fingerprint_task, token)


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
    text = translation_list_to_text(text)
    task_list = get_translation_text_fingerprint_chain_list(token, text, src, tgt,
                                                            ignore_fp_results=False)
    task = chain(task_list)
    task = task.apply_async(priority=6)
    return task.id


def translation_job(text, src, tgt, force):
    token = generate_translation_text_token(text, src, tgt)
    return_list = isinstance(text, list)
    text = translation_list_to_text(text)
    ##########################
    # Translation cache lookup
    ##########################
    if not force:
        direct_lookup_task_id = direct_lookup_generic_job(cache_lookup_translate_text_task, token,
                                                          False, return_list)
        if direct_lookup_task_id is not None:
            return direct_lookup_task_id

    ####################################
    # Fingerprinting and fp-based lookup
    ####################################
    direct_fingerprint_lookup_task_id = fingerprint_lookup_job(token)
    if direct_fingerprint_lookup_task_id is not None and not force:
        return direct_fingerprint_lookup_task_id
    #################
    # Computation job
    #################
    task_list = [
        translate_text_task.s(text, src, tgt),
        translate_text_callback_task.s(token, text, src, tgt, force, return_list)
    ]
    task = chain(task_list)
    task = task.apply_async(priority=6)
    return task.id


def detect_text_language_job(text):
    text = translation_list_to_text(text)
    # The only task is language detection because this task does not go through the caching logic
    task = (detect_text_language_task.s(text)).apply_async(priority=6)
    return task.id
