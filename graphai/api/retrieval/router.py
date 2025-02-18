from fastapi import APIRouter, Security, Depends
from fastapi_user_limiter.limiter import rate_limiter

from graphai.api.common.utils import format_api_results
from graphai.api.auth.router import get_current_active_user, get_user_for_rate_limiter
from graphai.api.auth.auth_utils import get_ratelimit_values

from graphai.celery.retrieval.jobs import (
    retrieve_lex_job,
    chunk_text_job
)
from graphai.celery.common.celery_config import get_task_info


from graphai.api.common.schemas import TaskIDResponse
from graphai.api.retrieval.schemas import (
    RetrievalRequest,
    RetrievalResponse,
    ChunkRequest,
    ChunkResponse
)

router = APIRouter(
    prefix='/rag',
    tags=['rag'],
    responses={404: {'description': 'Not found'}},
    dependencies=[Security(get_current_active_user, scopes=['rag'])]
)


@router.post('/retrieve', response_model=TaskIDResponse,
             dependencies=[Depends(rate_limiter(get_ratelimit_values()['rag']['max_requests'],
                                                get_ratelimit_values()['rag']['window'],
                                                user=get_user_for_rate_limiter))])
async def retrieve_from_es(data: RetrievalRequest):
    text = data.text
    lang = data.lang
    limit = data.limit
    index_to_search_in = data.index
    task_id = retrieve_lex_job(text, index_to_search_in, lang, limit)
    return {'task_id': task_id}


@router.get('/retrieve/status/{task_id}', response_model=RetrievalResponse)
async def retrieve_from_es_status(task_id):
    full_results = get_task_info(task_id)
    task_results = full_results['results']
    if task_results is not None:
        if 'result' in task_results:
            task_results = {
                'n_results': task_results['n_results'],
                'result': task_results['result'],
                'successful': task_results['successful'],
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)


@router.post('/chunk', response_model=ChunkResponse)
async def chunk_text(data: ChunkRequest):
    text = data.text
    chunk_size = data.chunk_size
    chunk_overlap = data.chunk_overlap
    return chunk_text_job(text, chunk_size, chunk_overlap)
