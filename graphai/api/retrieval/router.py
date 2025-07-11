from fastapi import APIRouter, Security, Depends
from typing import Annotated
from fastapi_user_limiter.limiter import rate_limiter

from graphai.api.auth.router import get_current_active_user, get_user_for_rate_limiter
from graphai.api.auth.auth_utils import (
    User,
    get_ratelimit_values,
    has_rag_access_rights
)
from graphai.api.retrieval.schemas import (
    RetrievalRequest,
    RetrievalResponse,
    RetrievalInfoResponse,
    ChunkRequest,
    ChunkResponse,
    AnonymizeRequest,
    AnonymizeResponse
)
from graphai.celery.retrieval.jobs import (
    retrieve_from_es_job,
    chunk_text_job,
    anonymize_text_job
)
from graphai.core.retrieval.retrieval_utils import INSUFFICIENT_ACCESS_ERROR

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
async def retrieve_from_es_index(data: RetrievalRequest,
                                 current_user: Annotated[User, Security(get_current_active_user, scopes=['user'])]):
    text = data.text
    filters = data.filters
    limit = data.limit
    index_to_search_in = data.index
    return_scores = data.return_scores
    filter_by_date = data.filter_by_date
    if not has_rag_access_rights(current_user.username, index_to_search_in):
        return INSUFFICIENT_ACCESS_ERROR
    results = retrieve_from_es_job(text, index_to_search_in, filters, limit, return_scores, filter_by_date)
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
                    "lang": ["en", "fr"]
                }
            },
            {
                "index": "servicedesk",
                "filters": {
                    "lang": ["en", "fr"],
                    "category": ["EPFL", "Public", "Finances", "Research", "Service Desk", "Human Resources"]
                }
            },
            {
                "index": "sac",
                "filters": {
                    "lang": ["en", "fr"]
                }
            },
            {
                "index": "*ANY OTHER*",
                "filters": {
                    "lang": [],
                    "category": []
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
    one_chunk_per_page = data.one_chunk_per_page
    one_chunk_per_doc = data.one_chunk_per_doc
    return chunk_text_job(text, chunk_size, chunk_overlap, one_chunk_per_page, one_chunk_per_doc)


@router.post('/anonymize', response_model=AnonymizeResponse,
             dependencies=[Depends(rate_limiter(get_ratelimit_values()['rag']['max_requests'],
                                                get_ratelimit_values()['rag']['window'],
                                                user=get_user_for_rate_limiter))])
async def anonymize_text(data: AnonymizeRequest):
    text = data.text
    lang = data.lang
    return anonymize_text_job(text, lang)
