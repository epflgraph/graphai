from fastapi import APIRouter
from celery import group, chain

from graphai.api.schemas.common import TaskIDResponse

from graphai.api.schemas.summarization import (
    SummarizationRequest,
    SummarizationResponse,
    SummaryFingerprintRequest,
    SummaryFingerprintResponse
)

from graphai.api.celery_tasks.common import (
    format_api_results,
    ignore_fingerprint_results_callback_task,
)

from graphai.api.celery_tasks.summarization import (
    lookup_text_summary_task,
    get_keywords_for_summarization_task,
    summarize_text_task,
    summarize_text_callback_task
)

from graphai.core.common.video import FingerprintParameters, generate_summary_type_dict, generate_summary_text_token
from graphai.core.interfaces.celery_config import get_task_info


router = APIRouter(
    prefix='/summarization',
    tags=['summarization'],
    responses={404: {'description': 'Not found'}}
)


def get_summary_task_chain(text, text_type, title=False, keywords=True, force=False):
    token = generate_summary_text_token(text, title=title)
    task_list = [
        lookup_text_summary_task.s(token, text, force),
        get_keywords_for_summarization_task.s(keywords),
        summarize_text_task.s(text_type, title),
        summarize_text_callback_task.s(force)
    ]
    return task_list


@router.post('/summary', response_model=TaskIDResponse)
async def summarize(data: SummarizationRequest):
    text = data.text
    text_type = data.text_type
    keywords = data.use_keywords
    force = data.force
    task_list = get_summary_task_chain(text, text_type, title=False, keywords=keywords, force=force)
    tasks = chain(task_list)
    tasks = tasks.apply_async(priority=6)
    return {'task_id': tasks.id}


@router.post('/title', response_model=TaskIDResponse)
async def create_title(data: SummarizationRequest):
    text = data.text
    text_type = data.text_type
    keywords = data.use_keywords
    force = data.force
    task_list = get_summary_task_chain(text, text_type, title=True, keywords=keywords, force=force)
    tasks = chain(task_list)
    tasks = tasks.apply_async(priority=6)
    return {'task_id': tasks.id}


@router.get('/title/status/{task_id}', response_model=SummarizationResponse)
@router.get('/summary/status/{task_id}', response_model=SummarizationResponse)
async def translate_status(task_id):
    full_results = get_task_info(task_id)
    task_results = full_results['results']
    if task_results is not None:
        if 'summary' in task_results:
            task_results = {
                'summary': task_results['summary'],
                'summary_type': task_results['summary_type'],
                'text_too_large': task_results['too_many_tokens'],
                'successful': task_results['successful'],
                'fresh': task_results['fresh']
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)
