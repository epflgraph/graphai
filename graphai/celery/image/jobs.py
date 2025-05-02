from celery import chain, group

from graphai.celery.video.jobs import get_slide_fingerprint_chain_list
from graphai.celery.image.tasks import (
    cache_lookup_retrieve_image_from_url_task,
    retrieve_image_from_url_task,
    upload_image_from_file_task,
    retrieve_image_from_url_callback_task,
    cache_lookup_slide_fingerprint_task,
    cache_lookup_extract_slide_text_task,
    extract_slide_text_task,
    extract_slide_text_callback_task,
    convert_pdf_to_pages_task,
    extract_multi_image_text_task,
    collect_multi_image_ocr_task
)
from graphai.celery.video.tasks import add_token_status_to_single_image_results_callback_task
from graphai.celery.common.jobs import (
    direct_lookup_generic_job,
    DEFAULT_TIMEOUT
)
from graphai.core.image.image import create_origin_token_using_info
from graphai.core.common.common_utils import is_pdf


def retrieve_image_from_url_job(url, force=False, no_cache=False):
    if not force:
        direct_lookup_task_id = direct_lookup_generic_job(cache_lookup_retrieve_image_from_url_task, url,
                                                          False, DEFAULT_TIMEOUT)
        if direct_lookup_task_id is not None:
            return direct_lookup_task_id

    # First retrieve the file, and then do the database callback
    task_list = [retrieve_image_from_url_task.s(url, None),
                 retrieve_image_from_url_callback_task.s(url)]
    if not no_cache:
        task_list += get_slide_fingerprint_chain_list(None, None, ignore_fp_results=True)
    else:
        task_list += [add_token_status_to_single_image_results_callback_task.s()]
    task = chain(task_list)
    task = task.apply_async(priority=2)
    return task.id


def upload_image_from_file_job(contents, file_extension, origin, origin_info, force=False, no_cache=False):
    effective_url = create_origin_token_using_info(origin, origin_info)
    if not force:
        direct_lookup_task_id = direct_lookup_generic_job(cache_lookup_retrieve_image_from_url_task, effective_url,
                                                          False, DEFAULT_TIMEOUT)
        if direct_lookup_task_id is not None:
            return direct_lookup_task_id
    task_list = [
        upload_image_from_file_task.s(contents, file_extension),
        retrieve_image_from_url_callback_task.s(effective_url)
    ]
    if not no_cache:
        task_list += get_slide_fingerprint_chain_list(None, None, ignore_fp_results=True)
    else:
        task_list += [add_token_status_to_single_image_results_callback_task.s()]
    task = chain(task_list)
    task = task.apply_async(priority=2)
    return task.id


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


def ocr_job(token, force=False, no_cache=False, method='google', api_token=None, openai_token=None, gemini_token=None,
            pdf_in_pages=True, model_type=None):
    ##################
    # OCR cache lookup
    ##################
    if not force and not no_cache:
        direct_lookup_task_id = direct_lookup_generic_job(cache_lookup_extract_slide_text_task, token,
                                                          False, DEFAULT_TIMEOUT, method)
        if direct_lookup_task_id is not None:
            return direct_lookup_task_id

    #####################
    # OCR computation job
    #####################
    if not is_pdf(token):
        task_list = [
            extract_slide_text_task.s(token, method, api_token, openai_token, gemini_token, model_type)
        ]
    else:
        n_parallel = 8
        task_list = [
            convert_pdf_to_pages_task.s(token),
            group(
                extract_multi_image_text_task.s(i, n_parallel, method, api_token, openai_token, gemini_token, model_type)
                for i in range(n_parallel)
            ),
            collect_multi_image_ocr_task.s()
        ]
    if not no_cache:
        task_list.append(extract_slide_text_callback_task.s(token, force))
    task = chain(task_list)
    task = task.apply_async(priority=2)
    return task.id
