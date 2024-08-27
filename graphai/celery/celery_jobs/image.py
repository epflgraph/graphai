from celery import chain

from graphai.celery.video.jobs import get_slide_fingerprint_chain_list
from graphai.celery.celery_tasks.image import (
    cache_lookup_slide_fingerprint_task,
    cache_lookup_extract_slide_text_task,
    extract_slide_text_task,
    extract_slide_text_callback_task
)

from graphai.celery.celery_jobs.common import direct_lookup_generic_job, DEFAULT_TIMEOUT


def fingerprint_lookup_job(token):
    return direct_lookup_generic_job(cache_lookup_slide_fingerprint_task, token, False, DEFAULT_TIMEOUT)


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
    task_list = get_slide_fingerprint_chain_list(token=token, force=force, ignore_fp_results=False)
    task = chain(task_list)
    task = task.apply_async(priority=2)
    return task.id


def ocr_job(token, force=False, method='google', api_token=None):
    ##################
    # OCR cache lookup
    ##################
    if not force:
        direct_lookup_task_id = direct_lookup_generic_job(cache_lookup_extract_slide_text_task, token,
                                                          False, DEFAULT_TIMEOUT, method)
        if direct_lookup_task_id is not None:
            return direct_lookup_task_id

    #####################
    # OCR computation job
    #####################
    task_list = [
        extract_slide_text_task.s(token, method, api_token),
        extract_slide_text_callback_task.s(token, force)
    ]
    task = chain(task_list)
    task = task.apply_async(priority=2)
    return task.id
