from fastapi import APIRouter, Security

from graphai.api.schemas.common import TaskIDResponse
from graphai.api.schemas.image import (
    ImageFingerprintRequest,
    ImageFingerprintResponse,
    ExtractTextRequest,
    ExtractTextResponse,
    DetectOCRLanguageResponse,
)
from graphai.api.celery_tasks.common import (
    format_api_results,
)

from graphai.api.celery_jobs.image import (
    fingerprint_job,
    ocr_job
)
from graphai.api.routers.auth import get_current_active_user

from graphai.core.interfaces.celery_config import get_task_info

# Initialise video router
router = APIRouter(
    prefix='/image',
    tags=['image'],
    responses={404: {'description': 'Not found'}},
    dependencies=[Security(get_current_active_user, scopes=['image'])]
)


@router.post('/calculate_fingerprint', response_model=TaskIDResponse)
async def calculate_slide_fingerprint(data: ImageFingerprintRequest):
    token = data.token
    force = data.force
    task_id = fingerprint_job(token, force)
    return {'task_id': task_id}


@router.get('/calculate_fingerprint/status/{task_id}', response_model=ImageFingerprintResponse)
async def calculate_slide_fingerprint_status(task_id):
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
    # Language detection requires OCR, so they have the same handler
    token = data.token
    method = data.method
    force = data.force
    assert method in ['google', 'tesseract']
    task_id = ocr_job(token, force, method)
    return {'task_id': task_id}


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
