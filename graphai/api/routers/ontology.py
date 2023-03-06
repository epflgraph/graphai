from fastapi import APIRouter

from graphai.api.schemas.ontology import *

from graphai.api.common.log import log
from graphai.api.common.ontology import ontology

# Initialise ontology router
router = APIRouter(
    prefix='/ontology',
    tags=['ontology'],
    responses={404: {'description': 'Not found'}}
)


@router.get('/tree', response_model=TreeResponse)
async def tree():
    log('Returning the ontology tree')
    return ontology.categories_categories.to_dict(orient='records')
