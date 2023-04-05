from fastapi import APIRouter

from graphai.api.schemas.ontology import *
from graphai.api.schemas.common import *

from graphai.api.common.log import log
from graphai.api.celery_tasks.ontology import get_ontology_tree_master, get_category_parent_master, \
    get_category_children_master
from graphai.api.celery_tasks.common import format_api_results
from graphai.core.celery_utils.celery_utils import compile_task_results

# Initialise ontology router
router = APIRouter(
    prefix='/ontology',
    tags=['ontology'],
    responses={404: {'description': 'Not found'}}
)


def ontology_tree_response_handler(id_and_results):
    full_results = compile_task_results(id_and_results['id'], task_results=id_and_results['results'])
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
    id_and_results = get_ontology_tree_master()
    return ontology_tree_response_handler(id_and_results)


@router.get('/tree/parent/{category_id}', response_model=TreeResponse)
async def parent(category_id):
    log('Returning the parent of category %s' % category_id)
    id_and_results = get_category_parent_master(int(category_id))
    return ontology_tree_response_handler(id_and_results)


@router.get('/tree/children/{category_id}', response_model=TreeResponse)
async def children(category_id):
    log('Returning the children of category %s' % category_id)
    id_and_results = get_category_children_master(int(category_id))
    return ontology_tree_response_handler(id_and_results)
