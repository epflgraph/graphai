from fastapi import APIRouter

from api.schemas.ontology import *
from api.common.log import log
from api.common.ontology import ontology


router = APIRouter(
    prefix='/ontology',
    tags=['ontology'],
    responses={404: {'description': 'Not found'}}
)


@router.get('/tree', response_model=TreeResponse)
async def tree():
    log('Returning the ontology tree')
    return ontology.categories_categories.to_dict(orient='records')
