from fastapi import APIRouter

from graphai.api.schemas.video import *

from graphai.api.celery_tasks.video import celery_multiproc_example_task


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

