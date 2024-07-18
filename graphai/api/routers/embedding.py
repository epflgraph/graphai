from fastapi import APIRouter, Security, Depends
from fastapi_user_limiter.limiter import rate_limiter
import json

from graphai.api.schemas.common import TaskIDResponse
from graphai.api.schemas.embedding import (
    EmbeddingFingerprintRequest,
    EmbeddingRequest,
    EmbeddingResponse,
)
from graphai.api.schemas.translation import (
    TextFingerprintResponse
)

from graphai.api.celery_jobs.embedding import (
    fingerprint_job,
    embedding_job
)
from graphai.api.celery_tasks.common import (
    format_api_results,
)
from graphai.api.routers.auth import get_current_active_user, get_user_for_rate_limiter
from graphai.api.common.auth_utils import get_ratelimit_values

from graphai.core.interfaces.celery_config import get_task_info


router = APIRouter(
    prefix='/embedding',
    tags=['embedding'],
    responses={404: {'description': 'Not found'}},
    dependencies=[Security(get_current_active_user, scopes=['translation'])]
)


@router.post('/calculate_fingerprint', response_model=TaskIDResponse)
async def calculate_embedding_text_fingerprint(data: EmbeddingFingerprintRequest):
    text = data.text
    model_type = data.model_type
    force = data.force
    task_id = fingerprint_job(text, model_type, force)
    return {'task_id': task_id}


@router.get('/calculate_fingerprint/status/{task_id}', response_model=TextFingerprintResponse)
async def calculate_embedding_text_fingerprint_status(task_id):
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


@router.post('/embed', response_model=TaskIDResponse,
             dependencies=[Depends(rate_limiter(get_ratelimit_values()['translation']['max_requests'],
                                                get_ratelimit_values()['translation']['window'],
                                                user=get_user_for_rate_limiter))])
async def embed_text(data: EmbeddingRequest):
    text = data.text
    model_type = data.model_type
    force = data.force
    task_id = embedding_job(text, model_type, force)
    return {'task_id': task_id}


@router.get('/embed/status/{task_id}', response_model=EmbeddingResponse)
async def embed_text_status(task_id):
    full_results = get_task_info(task_id)
    task_results = full_results['results']
    if task_results is not None:
        if isinstance(task_results, dict):
            if 'result' in task_results:
                task_results = {
                    'result': task_results['result'],
                    'successful': task_results['successful'],
                    'text_too_large': task_results['text_too_large'],
                    'model_type': task_results['model_type'],
                    'fresh': task_results['fresh'],
                    'device': task_results['device']
                }
            else:
                task_results = {
                    'result': f"Server overloaded, try again later. Details: {json.dumps(task_results)}",
                    'successful': False,
                    'text_too_large': False,
                    'model_type': None,
                    'fresh': False,
                    'device': None
                }
        elif isinstance(task_results, list):
            task_results = [{
                'result': tr['result'],
                'successful': tr['successful'],
                'text_too_large': tr['text_too_large'],
                'model_type': tr['model_type'],
                'fresh': tr['fresh'],
                'device': tr['device']
            } for tr in task_results]
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)
