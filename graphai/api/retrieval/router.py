from fastapi import APIRouter, Security, Depends
from fastapi_user_limiter.limiter import rate_limiter

from graphai.api.auth.router import get_current_active_user, get_user_for_rate_limiter
from graphai.api.auth.auth_utils import get_ratelimit_values
from graphai.api.retrieval.schemas import (
    RetrievalRequest,
    RetrievalResponse,
    RetrievalInfoResponse,
    ChunkRequest,
    ChunkResponse
)
from graphai.celery.retrieval.jobs import (
    retrieve_lex_job,
    chunk_text_job
)

router = APIRouter(
    prefix='/rag',
    tags=['rag'],
    responses={404: {'description': 'Not found'}},
    dependencies=[Security(get_current_active_user, scopes=['rag'])]
)


@router.post('/retrieve', response_model=RetrievalResponse,
             dependencies=[Depends(rate_limiter(get_ratelimit_values()['rag']['max_requests'],
                                                get_ratelimit_values()['rag']['window'],
                                                user=get_user_for_rate_limiter))])
async def retrieve_from_es(data: RetrievalRequest):
    text = data.text
    filters = data.filters
    limit = data.limit
    index_to_search_in = data.index
    results = retrieve_lex_job(text, index_to_search_in, filters, limit)
    return results


@router.get('/retrieve/info', response_model=RetrievalInfoResponse,
            dependencies=[Depends(rate_limiter(get_ratelimit_values()['rag']['max_requests'],
                                               get_ratelimit_values()['rag']['window'],
                                               user=get_user_for_rate_limiter))])
async def retrieve_endpoint_info():
    return {
        "indexes": [
            {
                "index": "lex",
                "filters": {
                    "lang": ["en", "fr", None]
                }
            },
            {
                "index": "servicedesk",
                "filters": {
                    "lang": ["en", "fr", None],
                    "category": ["EPFL", "Public", "Finances", "Research", "Service Desk", "Human Resources", None]
                }
            }
        ]
    }


@router.post('/chunk', response_model=ChunkResponse,
             dependencies=[Depends(rate_limiter(get_ratelimit_values()['rag']['max_requests'],
                                                get_ratelimit_values()['rag']['window'],
                                                user=get_user_for_rate_limiter))])
async def chunk_text(data: ChunkRequest):
    text = data.text
    chunk_size = data.chunk_size
    chunk_overlap = data.chunk_overlap
    return chunk_text_job(text, chunk_size, chunk_overlap)
