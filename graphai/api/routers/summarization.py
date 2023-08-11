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
    compute_summarization_text_fingerprint_task,
    compute_summarization_text_fingerprint_callback_task,
    summarization_text_fingerprint_find_closest_retrieve_from_db_task,
    summarization_text_fingerprint_find_closest_direct_task,
    summarization_text_fingerprint_find_closest_parallel_task,
    summarization_text_fingerprint_find_closest_callback_task,
    summarization_retrieve_text_fingerprint_callback_task,
    lookup_text_summary_task,
    get_keywords_for_summarization_task,
    summarize_text_task,
    summarize_text_callback_task
)

from graphai.core.common.video import FingerprintParameters, generate_summary_type_dict, \
    generate_summary_text_token
from graphai.core.interfaces.celery_config import get_task_info


router = APIRouter(
    prefix='/completion',
    tags=['completion'],
    responses={404: {'description': 'Not found'}}
)


def get_summary_text_fingerprint_chain_list(token, text, text_type, summary_type, len_class, tone,
                                            force, min_similarity=None, n_jobs=8,
                                            ignore_fp_results=False, results_to_return=None):
    # Loading min similarity parameter for text
    if min_similarity is None:
        fp_parameters = FingerprintParameters()
        min_similarity = fp_parameters.get_min_sim_text()

    # Generating the equality condition dictionary
    equality_conditions = generate_summary_type_dict(text_type, summary_type, len_class, tone)
    # The tasks are fingerprinting and callback, then lookup. The lookup is only among cache rows that satisfy the
    # equality conditions (source and target languages).
    task_list = [
        compute_summarization_text_fingerprint_task.s(token, text, force),
        compute_summarization_text_fingerprint_callback_task.s(text, text_type, summary_type, len_class, tone)
    ]
    if min_similarity == 1:
        task_list += [summarization_text_fingerprint_find_closest_direct_task.s(equality_conditions)]
    else:
        task_list += [
            summarization_text_fingerprint_find_closest_retrieve_from_db_task.s(equality_conditions),
            group(summarization_text_fingerprint_find_closest_parallel_task.s(i, n_jobs,
                                                                              equality_conditions, min_similarity)
                  for i in range(n_jobs))
        ]
    task_list += [summarization_text_fingerprint_find_closest_callback_task.s()]
    if ignore_fp_results:
        task_list += [ignore_fingerprint_results_callback_task.s(results_to_return)]
    else:
        task_list += [summarization_retrieve_text_fingerprint_callback_task.s()]
    return task_list


def get_summary_task_chain(token, text, text_type, summary_type, len_class, tone,
                           keywords=True, force=False, skip_token=False):
    if skip_token:
        task_list = [lookup_text_summary_task.s(text, force)]
    else:
        task_list = [lookup_text_summary_task.s(token, text, force)]
    task_list += [
        get_keywords_for_summarization_task.s(keywords),
        summarize_text_task.s(text_type, summary_type, len_class, tone),
        summarize_text_callback_task.s(force)
    ]
    return task_list


@router.post('/calculate_fingerprint', response_model=TaskIDResponse)
async def calculate_fingerprint(data: SummaryFingerprintRequest):
    text = data.text
    summary_type = data.summary_type
    text_type = data.text_type
    len_class = data.len_class
    tone = data.tone
    force = data.force
    token = generate_summary_text_token(text, text_type, summary_type, len_class, tone)
    task_list = get_summary_text_fingerprint_chain_list(token, text, text_type, summary_type, len_class, tone, force,
                                                        ignore_fp_results=False)
    task = chain(task_list)
    task = task.apply_async(priority=6)
    return {'task_id': task.id}


@router.get('/calculate_fingerprint/status/{task_id}', response_model=SummaryFingerprintResponse)
async def calculate_fingerprint_status(task_id):
    full_results = get_task_info(task_id)
    task_results = full_results['results']
    if task_results is not None:
        if 'result' in task_results:
            task_results = {
                'result': task_results['result'],
                'fresh': task_results['fresh'],
                'closest_token': task_results['closest'],
                'successful': task_results['result'] is not None
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)


@router.post('/summary', response_model=TaskIDResponse)
async def summarize(data: SummarizationRequest):
    text = data.text
    text_type = data.text_type
    len_class = data.len_class
    keywords = data.use_keywords
    tone = data.tone
    force = data.force

    token = generate_summary_text_token(text, text_type, 'summary', len_class, tone)
    if not force:
        task_list = get_summary_text_fingerprint_chain_list(token, text, text_type, 'summary', len_class, tone, force,
                                                            ignore_fp_results=True, results_to_return=token)
        skip_token = True
    else:
        task_list = []
        skip_token = False
    task_list += get_summary_task_chain(token, text, text_type, 'summary', len_class, tone,
                                        keywords=keywords, force=force, skip_token=skip_token)
    tasks = chain(task_list)
    tasks = tasks.apply_async(priority=6)
    return {'task_id': tasks.id}


@router.post('/title', response_model=TaskIDResponse)
async def create_title(data: SummarizationRequest):
    text = data.text
    text_type = data.text_type
    len_class = data.len_class
    keywords = data.use_keywords
    tone = data.tone
    force = data.force

    token = generate_summary_text_token(text, text_type, 'title', len_class, tone)
    if not force:
        task_list = get_summary_text_fingerprint_chain_list(token, text, text_type, 'title', len_class, tone, force,
                                                            ignore_fp_results=True, results_to_return=token)
        skip_token = True
    else:
        task_list = []
        skip_token = False
    task_list += get_summary_task_chain(token, text, text_type, 'title', len_class, tone,
                                        keywords=keywords, force=force, skip_token=skip_token)
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
