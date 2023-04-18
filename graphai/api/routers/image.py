from fastapi import APIRouter
from graphai.api.schemas.image import *
from graphai.api.schemas.common import *
from graphai.api.celery_tasks.common import format_api_results
from graphai.core.interfaces.celery_config import get_task_info

from ..celery_tasks.video import compute_slide_fingerprint_master

# Initialise video router
router = APIRouter(
    prefix='/image',
    tags=['image'],
    responses={404: {'description': 'Not found'}}
)

@router.post('/calculate_fingerprint', response_model=TaskIDResponse)
async def calculate_fingerprint(data: ImageFingerprintRequest):
    result = compute_slide_fingerprint_master(data.token, force=data.force)
    return {'task_id': result['id']}


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