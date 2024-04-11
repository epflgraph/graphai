from celery import group, chain

from graphai.api.celery_tasks.common import ignore_fingerprint_results_callback_task

from graphai.api.celery_tasks.image import (
    cache_lookup_slide_fingerprint_task,
    compute_slide_fingerprint_task,
    compute_slide_fingerprint_callback_task,
    slide_fingerprint_find_closest_direct_task,
    slide_fingerprint_find_closest_retrieve_from_db_task,
    slide_fingerprint_find_closest_parallel_task,
    slide_fingerprint_find_closest_callback_task,
    retrieve_slide_fingerprint_callback_task,
    cache_lookup_extract_slide_text_task,
    extract_slide_text_task,
    extract_slide_text_callback_task
)

from graphai.api.celery_jobs.common import direct_lookup_generic_job

from graphai.core.common.caching import FingerprintParameters


def get_slide_fingerprint_chain_list(token, force=False, min_similarity=None, n_jobs=8,
                                     ignore_fp_results=False, results_to_return=None):
    # Loading minimum similarity parameter for image
    if min_similarity is None:
        fp_parameters = FingerprintParameters()
        min_similarity = fp_parameters.get_min_sim_image()
    # The usual fingerprinting task list consists of fingerprinting and its callback, then lookup
    task_list = [
        compute_slide_fingerprint_task.s(token),
        compute_slide_fingerprint_callback_task.s(force)
    ]
    if min_similarity == 1:
        task_list += [slide_fingerprint_find_closest_direct_task.s()]
    else:
        task_list += [
            slide_fingerprint_find_closest_retrieve_from_db_task.s(),
            group(slide_fingerprint_find_closest_parallel_task.s(i, n_jobs, min_similarity) for i in range(n_jobs))
        ]
    task_list += [slide_fingerprint_find_closest_callback_task.s()]
    if ignore_fp_results:
        task_list += [ignore_fingerprint_results_callback_task.s(results_to_return)]
    else:
        task_list += [retrieve_slide_fingerprint_callback_task.s()]
    return task_list


def fingerprint_lookup_job(token):
    return direct_lookup_generic_job(cache_lookup_slide_fingerprint_task, token, False)


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
    task_list = get_slide_fingerprint_chain_list(token, force, ignore_fp_results=False)
    task = chain(task_list)
    task = task.apply_async(priority=2)
    return task.id


def ocr_job(token, force=False, method='google'):
    ##################
    # OCR cache lookup
    ##################
    if not force:
        direct_lookup_task_id = direct_lookup_generic_job(cache_lookup_extract_slide_text_task, token,
                                                          False, method)
        if direct_lookup_task_id is not None:
            return direct_lookup_task_id

    ##########################
    # Fingerprint cache lookup
    ##########################
    # Fingerprint cache lookup
    # Fingerprinting is never forced in non-fingerprinting jobs
    direct_fingerprint_lookup_task_id = fingerprint_lookup_job(token)
    if direct_fingerprint_lookup_task_id is None:
        # If the result is None, it means that there is no cached fingerprint, so we need to do fingerprinting
        task_list = get_slide_fingerprint_chain_list(token, False, ignore_fp_results=True,
                                                     results_to_return=token)
        # Reduced signature for the extract text task since first arg comes from prior tasks in the chain
        task_list += [extract_slide_text_task.s(method)]
    else:
        # We end up here if the fingerprint results are already cached.
        # Full signature for the extract text task
        task_list = [extract_slide_text_task.s(token, method)]
    #############################
    # Rest of OCR computation job
    #############################
    task_list += [
        extract_slide_text_callback_task.s(token, force)
    ]
    task = chain(task_list)
    task = task.apply_async(priority=2)
    return task.id
