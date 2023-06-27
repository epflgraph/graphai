from fastapi import APIRouter

from celery import group, chain

from graphai.api.schemas.common import TaskIDResponse
from graphai.api.schemas.image import (
    ImageFingerprintRequest,
    ImageFingerprintResponse,
    ExtractTextRequest,
    ExtractTextResponse,
    DetectOCRLanguageResponse,
)
from graphai.api.celery_tasks.common import (
    ignore_fingerprint_results_callback_task,
    format_api_results,
)
from graphai.api.celery_tasks.image import (
    compute_slide_fingerprint_task,
    compute_slide_fingerprint_callback_task,
    slide_fingerprint_find_closest_retrieve_from_db_task,
    slide_fingerprint_find_closest_parallel_task,
    slide_fingerprint_find_closest_direct_task,
    slide_fingerprint_find_closest_callback_task,
    retrieve_slide_fingerprint_callback_task,
    extract_slide_text_task,
    extract_slide_text_callback_task,
)
from graphai.core.interfaces.celery_config import get_task_info
from graphai.core.common.video import FingerprintParameters


# Initialise video router
router = APIRouter(
    prefix='/image',
    tags=['image'],
    responses={404: {'description': 'Not found'}}
)


def get_slide_fingerprint_chain_list(token, force, min_similarity=None, n_jobs=8,
                                     ignore_fp_results=False, results_to_return=None):
    # Loading minimum similarity parameter for image
    if min_similarity is None:
        fp_parameters = FingerprintParameters()
        min_similarity = fp_parameters.get_min_sim_image()
    # The usual fingerprinting task list consists of fingerprinting and its callback, then lookup
    task_list = [
        compute_slide_fingerprint_task.s(token, force),
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


@router.post('/calculate_fingerprint', response_model=TaskIDResponse)
async def calculate_fingerprint(data: ImageFingerprintRequest):
    token = data.token
    force = data.force
    task_list = get_slide_fingerprint_chain_list(token, force)
    task = chain(task_list)
    task = task.apply_async(priority=2)
    return {'task_id': task.id}


@router.get('/calculate_fingerprint/status/{task_id}', response_model=ImageFingerprintResponse)
async def calculate_fingerprint_status(task_id):
    full_results = get_task_info(task_id)
    task_results = full_results['results']
    if task_results is not None:
        if 'result' in task_results:
            task_results = {
                'result': task_results['result'],
                'fresh': task_results['fresh'],
                'closest_token': task_results['closest'],
                'closest_token_origin': task_results['closest_origin'],
                'successful': task_results['result'] is not None
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)


@router.post('/extract_text', response_model=TaskIDResponse)
@router.post('/detect_language', response_model=TaskIDResponse)
async def extract_text(data: ExtractTextRequest):
    token = data.token
    method = data.method
    force = data.force
    assert method in ['google', 'tesseract']
    # If force=True, fingerprinting is skipped
    # The tasks are slide text extraction and callback. These are the same for extract_text and detect_language.
    if not force:
        task_list = get_slide_fingerprint_chain_list(token, force, ignore_fp_results=True, results_to_return=token)
        task_list += [extract_slide_text_task.s(method, force)]
    else:
        task_list = [extract_slide_text_task.s(token, method, force)]

    task_list += [extract_slide_text_callback_task.s(token, force)]
    task = chain(task_list)
    task = task.apply_async(priority=2)
    return {'task_id': task.id}


@router.get('/extract_text/status/{task_id}', response_model=ExtractTextResponse)
async def extract_text_status(task_id):
    full_results = get_task_info(task_id)
    task_results = full_results['results']
    if task_results is not None:
        if 'results' in task_results:
            task_results = {
                'result': task_results['results'],
                'language': task_results['language'],
                'fresh': task_results['fresh'],
                'successful': task_results['results'] is not None
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)


@router.get('/detect_language/status/{task_id}', response_model=DetectOCRLanguageResponse)
async def detect_ocr_language_status(task_id):
    full_results = get_task_info(task_id)
    task_results = full_results['results']
    if task_results is not None:
        if 'language' in task_results:
            task_results = {
                'language': task_results['language'],
                'fresh': task_results['fresh'],
                'successful': task_results['results'] is not None
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)
