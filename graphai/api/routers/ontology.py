from fastapi import APIRouter

from graphai.api.schemas.ontology import *
from graphai.api.schemas.common import *

from graphai.api.common.log import log
from graphai.api.celery_tasks.ontology import get_ontology_tree_master, get_category_parent_master, \
    get_category_children_master

# Initialise ontology router
router = APIRouter(
    prefix='/ontology',
    tags=['ontology'],
    responses={404: {'description': 'Not found'}}
)


@router.get('/tree', response_model=TreeResponse)
async def tree():
    log('Returning the ontology tree')
    return get_ontology_tree_master()


@router.get('/tree/parent/{category_id}', response_model=TreeResponse)
async def parent(category_id):
    log('Returning the parent of category %s' % category_id)
    return get_category_parent_master(int(category_id))


@router.get('/tree/children/{category_id}', response_model=TreeResponse)
async def children(category_id):
    log('Returning the children of category %s' % category_id)
    return get_category_children_master(int(category_id))
