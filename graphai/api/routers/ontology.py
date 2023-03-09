from fastapi import APIRouter

from graphai.api.schemas.ontology import *
from graphai.api.schemas.common import *
from graphai.api.common.ontology import ontology

from graphai.api.common.log import log
from graphai.core.celery_utils.celery_utils import get_task_info, format_results
from graphai.api.celery_tasks.ontology import get_ontology_tree_task, get_whatever
from starlette.responses import JSONResponse

# Initialise ontology router
router = APIRouter(
    prefix='/ontology',
    tags=['ontology'],
    responses={404: {'description': 'Not found'}}
)


@router.get('/tree', response_model=TreeResponse)
async def tree():
    log('Returning the ontology tree')
    task = get_ontology_tree_task.apply_async(args=[ontology], priority=6)
    results = task.get()
    print(task.name)
    return format_results(task.id, task.name, task.status, results)


@router.get('/whatever')
async def whatever():
    task = get_whatever.apply_async(priority=3)
    return task.get()
