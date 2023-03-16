from fastapi import APIRouter
from fastapi.responses import FileResponse

from graphai.api.schemas.video import *
from graphai.api.schemas.common import *

from graphai.api.celery_tasks.video import celery_multiproc_example_master, \
    retrieve_and_generate_token_master, compute_signature_master, get_file_master
from graphai.core.celery_utils.celery_utils import get_task_info


# Initialise video router
router = APIRouter(
    prefix='/video',
    tags=['video'],
    responses={404: {'description': 'Not found'}}
)


@router.post('/multiprocessing_example', response_model=MultiprocessingExampleResponse)
async def multiprocessing_example(data: MultiprocessingExampleRequest):
    result = celery_multiproc_example_master(data)
    return result


@router.post('/retrieve_url', response_model=TaskIDResponse)
async def retrieve_file(data: RetrieveURLRequest):
    result = retrieve_and_generate_token_master(data.url)
    return result


# For each async endpoint, we also have a status endpoint since they have different response models.
@router.get('/retrieve_url/status/{task_id}', response_model=RetrieveURLResponse)
async def get_retrieve_file_status(task_id):
    return get_task_info(task_id)


@router.post('/calculate_fingerprint', response_model=TaskIDResponse)
async def calculate_fingerprint(data: ComputeSignatureRequest):
    result = compute_signature_master(data.token)
    return result


@router.get('/calculate_fingerprint/status/{task_id}', response_model=ComputeSignatureResponse)
async def calculate_fingerprint_status(task_id):
    return get_task_info(task_id)


@router.post('/get_file/')
async def get_file(data: FileRequest):
    return FileResponse(get_file_master(data.filename))

