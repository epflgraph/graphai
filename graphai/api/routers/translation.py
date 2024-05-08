from fastapi import APIRouter, Security, Depends
from fastapi_user_limiter.limiter import rate_limiter

from graphai.api.schemas.common import TaskIDResponse
from graphai.api.schemas.translation import (
    TranslationRequest,
    TranslationResponse,
    TextDetectLanguageRequest,
    TextDetectLanguageResponse,
    TextFingerprintResponse,
)

from graphai.api.celery_jobs.translation import (
    fingerprint_job,
    translation_job,
    detect_text_language_job
)
from graphai.api.celery_tasks.common import (
    format_api_results,
)
from graphai.api.routers.auth import get_current_active_user, get_user_for_rate_limiter
from graphai.api.common.auth_utils import get_ratelimit_values

from graphai.core.interfaces.celery_config import get_task_info


router = APIRouter(
    prefix='/translation',
    tags=['translation'],
    responses={404: {'description': 'Not found'}},
    dependencies=[Security(get_current_active_user, scopes=['translation'])]
)


@router.post('/calculate_fingerprint', response_model=TaskIDResponse)
async def calculate_translation_text_fingerprint(data: TranslationRequest):
    text = data.text
    src = data.source
    tgt = data.target
    force = data.force
    task_id = fingerprint_job(text, src, tgt, force)
    return {'task_id': task_id}


@router.get('/calculate_fingerprint/status/{task_id}', response_model=TextFingerprintResponse)
async def calculate_translation_text_fingerprint_status(task_id):
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


@router.post('/translate', response_model=TaskIDResponse,
             dependencies=[Depends(rate_limiter(get_ratelimit_values()['translation']['max_requests'],
                                                get_ratelimit_values()['translation']['window'],
                                                user=get_user_for_rate_limiter))])
async def translate(data: TranslationRequest):
    text = data.text
    src = data.source
    tgt = data.target
    force = data.force
    task_id = translation_job(text, src, tgt, force)
    return {'task_id': task_id}


@router.get('/translate/status/{task_id}', response_model=TranslationResponse)
async def translate_status(task_id):
    full_results = get_task_info(task_id)
    task_results = full_results['results']
    if task_results is not None:
        if 'result' in task_results:
            task_results = {
                'result': task_results['result'],
                'text_too_large': task_results['text_too_large'],
                'successful': task_results['successful'],
                'fresh': task_results['fresh'],
                'device': task_results['device']
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)


@router.post('/detect_language', response_model=TaskIDResponse)
async def text_detect_language(data: TextDetectLanguageRequest):
    text = data.text
    task_id = detect_text_language_job(text)
    return {'task_id': task_id}


@router.get('/detect_language/status/{task_id}', response_model=TextDetectLanguageResponse)
async def text_detect_language_status(task_id):
    full_results = get_task_info(task_id)
    task_results = full_results['results']
    if task_results is not None:
        if 'language' in task_results:
            task_results = {
                'language': task_results['language'],
                'successful': task_results['successful']
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)
