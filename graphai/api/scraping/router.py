from fastapi import APIRouter, Security, Depends
from fastapi_user_limiter.limiter import rate_limiter

from graphai.api.common.schemas import TaskIDResponse

from graphai.api.scraping.schemas import (
    GetSublinksRequest,
    GetSublinksResponse,
    ExtractContentRequest,
    ExtractContentResponse
)

from graphai.celery.common.tasks import (
    format_api_results,
)

from graphai.celery.scraping.jobs import (
    extract_sublinks_job,
    extract_content_job
)
from graphai.api.auth.router import get_current_active_user, get_user_for_rate_limiter
from graphai.api.auth.auth_utils import get_ratelimit_values

from graphai.celery.common.celery_config import get_task_info

router = APIRouter(
    prefix='/scraping',
    tags=['scraping'],
    responses={404: {'description': 'Not found'}},
    dependencies=[Security(get_current_active_user, scopes=['scraping'])]
)


@router.post('/sublinks', response_model=TaskIDResponse,
             dependencies=[Depends(rate_limiter(get_ratelimit_values()['scraping']['max_requests'],
                                                get_ratelimit_values()['scraping']['window'],
                                                user=get_user_for_rate_limiter))]
             )
async def extract_sublinks(data: GetSublinksRequest):
    url = data.url
    force = data.force
    task_id = extract_sublinks_job(url, force)
    return {'task_id': task_id}


@router.get('/sublinks/status/{task_id}', response_model=GetSublinksResponse)
async def extract_sublinks_status(task_id):
    full_results = get_task_info(task_id)
    task_results = full_results['results']
    if task_results is not None:
        if 'sublinks' in task_results:
            task_results = {
                'token': task_results['token'],
                'validated_url': task_results['validated_url'],
                'sublinks': task_results['sublinks'],
                'status_msg': task_results['status_msg'],
                'successful': task_results['sublinks'] is not None,
                'fresh': task_results['fresh']
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)


@router.post('/content', response_model=TaskIDResponse,
             dependencies=[Depends(rate_limiter(get_ratelimit_values()['scraping']['max_requests'],
                                                get_ratelimit_values()['scraping']['window'],
                                                user=get_user_for_rate_limiter))]
             )
async def extract_page_content(data: ExtractContentRequest):
    url = data.url
    force = data.force
    headers = data.remove_headers
    long_patterns = data.remove_long_patterns
    task_id = extract_content_job(url, force, headers, long_patterns)
    return {'task_id': task_id}


@router.get('/content/status/{task_id}', response_model=ExtractContentResponse)
async def extract_page_content_status(task_id):
    full_results = get_task_info(task_id)
    task_results = full_results['results']
    if task_results is not None:
        if 'data' in task_results:
            task_results = {
                'token': task_results['token'],
                'validated_url': task_results['validated_url'],
                'sublinks': task_results['sublinks'],
                'status_msg': task_results['status_msg'],
                'data': task_results['data'],
                'successful': task_results['sublinks'] is not None,
                'fresh': task_results['fresh']
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)
