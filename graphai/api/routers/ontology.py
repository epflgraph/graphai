from fastapi import APIRouter

from graphai.api.schemas.ontology import *
from graphai.api.schemas.common import *
from graphai.api.common.ontology import ontology

from graphai.api.common.log import log
from graphai.core.celery_utils.celery_utils import get_task_info, format_results
from graphai.api.celery_tasks.ontology import get_ontology_tree_task, get_category_parent_task, \
    get_category_children_task
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


@router.get('/tree/parent/{category_id}', response_model=TreeResponse)
async def parent(category_id):
    log('Returning the parent of category %s' % category_id)
    task = get_category_parent_task.apply_async(args=[ontology, int(category_id)], priority=6)
    results = task.get()
    print(task.name)
    return format_results(task.id, task.name, task.status, results)


@router.get('/tree/children/{category_id}', response_model=TreeResponse)
async def children(category_id):
    log('Returning the children of category %s' % category_id)
    task = get_category_children_task.apply_async(args=[ontology, int(category_id)], priority=6)
    results = task.get()
    print(task.name)
    return format_results(task.id, task.name, task.status, results)
