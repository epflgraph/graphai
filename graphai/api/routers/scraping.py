from fastapi import APIRouter
from celery import group, chain

from graphai.api.schemas.common import TaskIDResponse

from graphai.api.schemas.scraping import (
    GetSublinksRequest,
    GetSublinksResponse
)

from graphai.api.celery_tasks.common import (
    format_api_results,
    ignore_fingerprint_results_callback_task,
)

from graphai.api.celery_tasks.scraping import (
    initialize_scraping_url_task,
    get_scraping_sublinks_task,
    scraping_sublinks_callback_task
)

from graphai.core.common.scraping import create_base_url_token

from graphai.core.interfaces.celery_config import get_task_info

router = APIRouter(
    prefix='/scraping',
    tags=['scraping'],
    responses={404: {'description': 'Not found'}}
)

@router.post('/sublinks', response_model=TaskIDResponse)
async def extract_sublinks(data: GetSublinksRequest):
    url = data.url
    force = data.force

    token = create_base_url_token(url)
    task_list = [
        initialize_scraping_url_task.s(token, url, force),
        get_scraping_sublinks_task.s(),
        scraping_sublinks_callback_task.s(token)
    ]
    task = chain(task_list)
    task = task.apply_async(priority=6)
    return {'task_id': task.id}


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
                'successful': task_results['sublinks'] is not None,
                'fresh': task_results['fresh']
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)
