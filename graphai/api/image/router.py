from fastapi import APIRouter, Security, Depends
from fastapi_user_limiter.limiter import rate_limiter

from graphai.api.common.schemas import TaskIDResponse
from graphai.api.image.schemas import (
    ImageFingerprintRequest,
    ImageFingerprintResponse,
    ExtractTextRequest,
    ExtractTextResponse,
    DetectOCRLanguageResponse,
)
from graphai.celery.celery_tasks.common import (
    format_api_results,
)

from graphai.celery.celery_jobs.image import (
    fingerprint_job,
    ocr_job
)
from graphai.api.auth.router import get_current_active_user, get_user_for_rate_limiter
from graphai.api.auth.auth_utils import get_ratelimit_values

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


@router.post('/extract_text', response_model=TaskIDResponse,
             dependencies=[Depends(rate_limiter(get_ratelimit_values()['image']['max_requests'],
                                                get_ratelimit_values()['image']['window'],
                                                user=get_user_for_rate_limiter))])
@router.post('/detect_language', response_model=TaskIDResponse,
             dependencies=[Depends(rate_limiter(get_ratelimit_values()['image']['max_requests'],
                                                get_ratelimit_values()['image']['window'],
                                                user=get_user_for_rate_limiter))])
async def extract_text(data: ExtractTextRequest):
    # Language detection requires OCR, so they have the same handler
    token = data.token
    method = data.method
    force = data.force
    api_token = data.google_api_token
    assert method in ['google', 'tesseract']
    assert api_token is not None or method != 'google'
    task_id = ocr_job(token, force, method, api_token)
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
