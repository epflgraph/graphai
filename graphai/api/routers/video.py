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


@router.post('/multiprocessing_example', response_model=TaskIDResponse)
async def multiprocessing_example(data: MultiprocessingExampleRequest):
    task = multiproc_example_task.apply_async(args=[data])
    return JSONResponse({"task_id": task.id})

