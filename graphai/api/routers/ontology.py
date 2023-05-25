from fastapi import APIRouter

from graphai.api.schemas.ontology import TreeResponse

from graphai.api.common.log import log
from graphai.api.celery_tasks.ontology import get_ontology_tree_task, get_category_parent_task, \
    get_category_children_task
from graphai.api.celery_tasks.common import format_api_results
from graphai.core.interfaces.celery_config import get_task_info

# Initialise ontology router
router = APIRouter(
    prefix='/ontology',
    tags=['ontology'],
    responses={404: {'description': 'Not found'}}
)


def ontology_tree_response_handler(id_and_results):
    full_results = get_task_info(id_and_results['id'], task_results=id_and_results['results'])
    task_results = id_and_results['results']
    if task_results is not None:
        if 'child_to_parent' in task_results:
            task_results = task_results['child_to_parent']
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)


@router.get('/tree', response_model=TreeResponse)
async def tree():
    log('Returning the ontology tree')
    task = (get_ontology_tree_task.s()).apply_async(priority=6)
    task_id = task.id
    try:
        results = task.get(timeout=10)
    except TimeoutError as e:
        print(e)
        results = None
    id_and_results = {'id': task_id, 'results': results}
    return ontology_tree_response_handler(id_and_results)


@router.get('/tree/parent/{category_id}', response_model=TreeResponse)
async def parent(category_id):
    log('Returning the parent of category %s' % category_id)
    task = (get_category_parent_task.s(int(category_id))).apply_async(priority=6)
    try:
        results = task.get(timeout=10)
    except TimeoutError as e:
        print(e)
        results = None
    id_and_results = {'id': task.id, 'results': results}
    return ontology_tree_response_handler(id_and_results)


@router.get('/tree/children/{category_id}', response_model=TreeResponse)
async def children(category_id):
    log('Returning the children of category %s' % category_id)
    task = (get_category_children_task.s(int(category_id))).apply_async(priority=6)
    task_id = task.id
    try:
        results = task.get(timeout=10)
    except TimeoutError as e:
        print(e)
        results = None
    id_and_results = {'id': task_id, 'results': results}
    return ontology_tree_response_handler(id_and_results)
