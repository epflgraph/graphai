from celery import group
from fastapi import APIRouter
from graphai.api.schemas.image import *
from graphai.api.schemas.common import *
from graphai.api.celery_tasks.common import format_api_results
from graphai.core.interfaces.celery_config import get_task_info

from ..celery_tasks.image import compute_slide_fingerprint_task, \
    compute_slide_fingerprint_callback_task, slide_fingerprint_find_closest_retrieve_from_db_task, \
    slide_fingerprint_find_closest_parallel_task, slide_fingerprint_find_closest_callback_task, \
    retrieve_slide_fingerprint_callback_task, extract_slide_text_task, extract_slide_text_callback_task

# Initialise video router
router = APIRouter(
    prefix='/image',
    tags=['image'],
    responses={404: {'description': 'Not found'}}
)


@router.post('/calculate_fingerprint', response_model=TaskIDResponse)
async def calculate_fingerprint(data: ImageFingerprintRequest):
    token = data.token
    force = data.force
    min_similarity = 1
    n_jobs = 8
    task = (
            compute_slide_fingerprint_task.s(token, force) |
            compute_slide_fingerprint_callback_task.s(token) |
            slide_fingerprint_find_closest_retrieve_from_db_task.s(token) |
            group(slide_fingerprint_find_closest_parallel_task.s(token, i, n_jobs, min_similarity) for i in range(n_jobs)) |
            slide_fingerprint_find_closest_callback_task.s(token) |
            retrieve_slide_fingerprint_callback_task.s()
        ).apply_async(priority=2)
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
    task = (extract_slide_text_task.s(token, method, force) |
            extract_slide_text_callback_task.s(token)).apply_async(priority=2)
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

