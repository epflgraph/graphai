from fastapi import APIRouter

from graphai.api.schemas.video import *
from graphai.api.schemas.common import *

from graphai.api.celery_tasks.video import celery_multiproc_example_task, retrieve_and_generate_token
from graphai.core.celery_utils.celery_utils import get_task_info


# Initialise video router
router = APIRouter(
    prefix='/video',
    tags=['video'],
    responses={404: {'description': 'Not found'}}
)


@router.post('/multiprocessing_example', response_model=MultiprocessingExampleResponse)
async def multiprocessing_example(data: MultiprocessingExampleRequest):
    result = celery_multiproc_example_task(data)
    return result


@router.post('/retrieve_file', response_model=TaskIDResponse)
async def retrieve_file(data: RetrieveFileRequest):
    result = retrieve_and_generate_token(data.url)
    return result


@router.get('/retrieve_file/status/{task_id}', response_model=RetrieveFileResponse)
async def get_retrieve_file_status(task_id):
    return get_task_info(task_id)

