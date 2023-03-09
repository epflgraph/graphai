from fastapi import APIRouter
from graphai.api.celery_tasks.video import *
from graphai.api.schemas.common import *
from graphai.api.schemas.video import *
from starlette.responses import JSONResponse


# Initialise video router
router = APIRouter(
    prefix='/video',
    tags=['video'],
    responses={404: {'description': 'Not found'}}
)


@router.post('/multiprocessing_example', response_model=MultiprocessingExampleResponse)
async def multiprocessing_example(data: MultiprocessingExampleRequest):
    result = multiproc_example_task.apply_async(args=[data]).get()
    return result

@router.post('/multiprocessing_example_purecelery', response_model=MultiprocessingExampleResponse)
async def multiprocessing_example(data: MultiprocessingExampleRequest):
    result = celery_multiproc_example_task(data)
    return result
