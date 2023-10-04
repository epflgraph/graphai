from fastapi import APIRouter
from celery import group, chain

from graphai.api.schemas.common import TaskIDResponse

from graphai.api.schemas.scraping import (
    GetSublinksRequest,
    GetSublinksResponse,
    ExtractContentRequest,
    ExtractContentResponse
)

from graphai.api.celery_tasks.common import (
    format_api_results,
    text_dummy_task
)

from graphai.api.celery_tasks.scraping import (
    initialize_url_and_get_sublinks_task,
    scraping_sublinks_callback_task,
    process_all_scraping_sublinks_preprocess_task,
    process_all_scraping_sublinks_parallel_task,
    process_all_scraping_sublinks_callback_task,
    remove_junk_scraping_parallel_task,
    extract_scraping_content_callback_task
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
        initialize_url_and_get_sublinks_task.s(token, url, force),
        scraping_sublinks_callback_task.s()
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
                'status_msg': task_results['status_msg'],
                'successful': task_results['sublinks'] is not None,
                'fresh': task_results['fresh']
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)


@router.post('/content', response_model=TaskIDResponse)
async def extract_page_content(data: ExtractContentRequest):
    url = data.url
    force = data.force
    headers = data.remove_headers
    long_patterns = data.remove_long_patterns
    n_jobs = 8

    token = create_base_url_token(url)
    task_list = [
        initialize_url_and_get_sublinks_task.s(token, url, force),
        scraping_sublinks_callback_task.s(),
        process_all_scraping_sublinks_preprocess_task.s(headers, long_patterns, force),
        group(process_all_scraping_sublinks_parallel_task.s(i, n_jobs) for i in range(n_jobs)),
        process_all_scraping_sublinks_callback_task.s(),
        text_dummy_task.s(),
        group(remove_junk_scraping_parallel_task.s(i, n_jobs, headers, long_patterns) for i in range(n_jobs)),
        extract_scraping_content_callback_task.s(headers, long_patterns)
    ]
    task = chain(task_list)
    task = task.apply_async(priority=6)
    return {'task_id': task.id}


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
