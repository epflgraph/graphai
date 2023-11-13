from fastapi import APIRouter
from celery import group, chain

from graphai.api.schemas.common import TaskIDResponse

from graphai.api.schemas.completion import (
    LectureSummarizationRequest,
    GenericSummarizationRequest,
    CleanupRequest,
    SummaryResponse,
    CleanupResponse,
    SummaryFingerprintRequest,
    SummaryFingerprintResponse,
    SlideSubsetRequest,
    SlideSubsetResponse,
    AcademicEntityDescriptor,
    AcademicEntitySummarizationRequest,
    AcademicEntitySummaryResponse
)

from graphai.api.celery_tasks.common import (
    format_api_results,
    ignore_fingerprint_results_callback_task,
)

from graphai.api.celery_tasks.completion import (
    compute_summarization_text_fingerprint_task,
    compute_summarization_text_fingerprint_callback_task,
    summarization_text_fingerprint_find_closest_retrieve_from_db_task,
    summarization_text_fingerprint_find_closest_direct_task,
    summarization_text_fingerprint_find_closest_parallel_task,
    summarization_text_fingerprint_find_closest_callback_task,
    summarization_retrieve_text_fingerprint_callback_task,
    lookup_text_completion_task,
    get_keywords_for_summarization_task,
    completion_text_callback_task,
    request_text_completion_task,
    simulate_completion_task,
    choose_best_subset_task
)

from graphai.core.common.text_utils import generate_summary_text_token, generate_completion_type_dict
from graphai.core.common.caching import FingerprintParameters
from graphai.core.interfaces.celery_config import get_task_info


router = APIRouter(
    prefix='/completion',
    tags=['completion'],
    responses={404: {'description': 'Not found'}}
)


def populate_academic_entity_dict(data):
    text = {
        'entity': data.entity,
        'name': data.name,
        'subtype': data.subtype,
        'possible_subtypes': data.possible_subtypes,
        'text': data.text,
        'categories': data.categories
    }
    return text


def populate_slide_concepts_dict(slides):
    return {slide.number: slide.concepts for slide in slides}


def get_completion_text_fingerprint_chain_list(token, text, text_type, completion_type,
                                               force, min_similarity=None, n_jobs=8,
                                               ignore_fp_results=False, results_to_return=None):
    # Loading min similarity parameter for text
    if min_similarity is None:
        fp_parameters = FingerprintParameters()
        min_similarity = fp_parameters.get_min_sim_text()

    # Generating the equality condition dictionary
    equality_conditions = generate_completion_type_dict(text_type, completion_type)
    # The tasks are fingerprinting and callback, then lookup. The lookup is only among cache rows that satisfy the
    # equality conditions (source and target languages).
    task_list = [
        compute_summarization_text_fingerprint_task.s(token, text, force),
        compute_summarization_text_fingerprint_callback_task.s(text, text_type, completion_type)
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


def get_completion_task_chain(token, text, text_type, result_type,
                              keywords=True, force=False, skip_token=False, debug=False):
    if skip_token:
        task_list = [lookup_text_completion_task.s(text, force)]
    else:
        task_list = [lookup_text_completion_task.s(token, text, force)]
    if keywords:
        task_list.append(get_keywords_for_summarization_task.s())
    task_list.append(request_text_completion_task.s(text_type, result_type, debug))
    task_list.append(completion_text_callback_task.s(force))
    return task_list


def get_cleanup_task_chain(token, text, text_type, result_type='cleanup',
                           force=False, skip_token=False, debug=False):
    if skip_token:
        task_list = [lookup_text_completion_task.s(text, force)]
    else:
        task_list = [lookup_text_completion_task.s(token, text, force)]
    task_list += [
        request_text_completion_task.s(text_type, result_type, debug),
        completion_text_callback_task.s(force)
    ]
    return task_list


@router.post('/calculate_fingerprint', response_model=TaskIDResponse)
async def calculate_summary_text_fingerprint(data: SummaryFingerprintRequest):
    text = data.text
    completion_type = data.completion_type
    text_type = data.text_type
    assert (
        (completion_type == 'cleanup' and (isinstance(text, str) or isinstance(text, dict)))
        or (completion_type == 'summary' and text_type != 'lecture'
            and (isinstance(text, str) or isinstance(text, dict)))
        or (completion_type == 'summary' and text_type == 'lecture' and isinstance(text, list))
        or (completion_type == 'summary' and (text_type == 'unit' or text_type == 'person')
            and isinstance(text, AcademicEntityDescriptor))
    )
    if completion_type == 'summary' and text_type == 'lecture':
        text = populate_slide_concepts_dict(text)
    if completion_type == 'summary' and (text_type == 'unit' or text_type == 'person'):
        text = populate_academic_entity_dict(text)

    force = data.force
    token = generate_summary_text_token(text, text_type, completion_type)
    task_list = get_completion_text_fingerprint_chain_list(token, text, text_type, completion_type, force,
                                                           ignore_fp_results=False)
    task = chain(task_list)
    task = task.apply_async(priority=6)
    return {'task_id': task.id}


@router.get('/calculate_fingerprint/status/{task_id}', response_model=SummaryFingerprintResponse)
async def calculate_summary_text_fingerprint_status(task_id):
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


@router.post('/summary/lecture', response_model=TaskIDResponse)
async def summarize_lecture(data: LectureSummarizationRequest):
    slides = data.slides
    text = populate_slide_concepts_dict(slides)
    text_type = 'lecture'
    force = data.force
    debug = data.debug
    simulate = data.simulate

    if not simulate:
        token = generate_summary_text_token(text, text_type, 'summary')
        if not force:
            task_list = get_completion_text_fingerprint_chain_list(token, text, text_type, 'summary', force,
                                                                   ignore_fp_results=True, results_to_return=token)
            skip_token = True
        else:
            task_list = []
            skip_token = False
        task_list += get_completion_task_chain(token, text, text_type, 'summary',
                                               keywords=False, force=force, skip_token=skip_token, debug=debug)
    else:
        task_list = [simulate_completion_task.s(text, text_type, 'summary')]
    tasks = chain(task_list)
    tasks = tasks.apply_async(priority=6)
    return {'task_id': tasks.id}


@router.post('/summary/academic_entity', response_model=TaskIDResponse)
async def summarize_academic_entity(data: AcademicEntitySummarizationRequest):
    text = populate_academic_entity_dict(data)
    text_type = 'academic_entity'
    force = data.force
    debug = data.debug
    simulate = data.simulate

    if not simulate:
        token = generate_summary_text_token(text, text_type, 'summary')
        if not force:
            task_list = get_completion_text_fingerprint_chain_list(token, text, text_type, 'summary', force,
                                                                   ignore_fp_results=True, results_to_return=token)
            skip_token = True
        else:
            task_list = []
            skip_token = False
        task_list += get_completion_task_chain(token, text, text_type, 'summary',
                                               keywords=False, force=force, skip_token=skip_token, debug=debug)
    else:
        task_list = [simulate_completion_task.s(text, text_type, 'summary')]

    tasks = chain(task_list)
    tasks = tasks.apply_async(priority=6)
    return {'task_id': tasks.id}


@router.post('/summary/generic', response_model=TaskIDResponse)
async def summarize_text(data: GenericSummarizationRequest):
    text = data.text
    text_type = 'text'
    force = data.force
    debug = data.debug
    keywords = data.keywords
    simulate = data.simulate

    if not simulate:
        token = generate_summary_text_token(text, text_type, 'summary')
        if not force:
            task_list = get_completion_text_fingerprint_chain_list(token, text, text_type, 'summary', force,
                                                                   ignore_fp_results=True, results_to_return=token)
            skip_token = True
        else:
            task_list = []
            skip_token = False
        task_list += get_completion_task_chain(token, text, text_type, 'summary',
                                               keywords=keywords, force=force, skip_token=skip_token, debug=debug)
    else:
        task_list = [simulate_completion_task.s(text, text_type, 'summary')]
    tasks = chain(task_list)
    tasks = tasks.apply_async(priority=6)
    return {'task_id': tasks.id}


@router.post('/cleanup', response_model=TaskIDResponse)
async def clean_up(data: CleanupRequest):
    text = data.text
    text_type = data.text_type
    force = data.force
    debug = data.debug
    simulate = data.simulate

    if not simulate:
        token = generate_summary_text_token(text, text_type, 'cleanup')
        if not force:
            task_list = get_completion_text_fingerprint_chain_list(token, text, text_type, 'cleanup', force,
                                                                   ignore_fp_results=True, results_to_return=token)
            skip_token = True
        else:
            task_list = []
            skip_token = False
        task_list += get_completion_task_chain(token, text, text_type, 'cleanup',
                                               keywords=False, force=force, skip_token=skip_token, debug=debug)
    else:
        task_list = [simulate_completion_task.s(text, text_type, 'cleanup')]
    tasks = chain(task_list)
    tasks = tasks.apply_async(priority=6)
    return {'task_id': tasks.id}


@router.get('/summary/lecture/status/{task_id}', response_model=SummaryResponse)
@router.get('/summary/generic/status/{task_id}', response_model=SummaryResponse)
@router.get('/summary/academic_entity/status/{task_id}', response_model=AcademicEntitySummaryResponse)
@router.get('/cleanup/status/{task_id}', response_model=CleanupResponse)
async def summarize_status(task_id):
    full_results = get_task_info(task_id)
    task_results = full_results['results']
    if task_results is not None:
        if 'result' in task_results:
            task_results = {
                'result': task_results['result'],
                'result_type': task_results['result_type'],
                'text_too_large': task_results['too_many_tokens'],
                'successful': task_results['successful'],
                'fresh': task_results['fresh'],
                'debug_message': task_results['full_message'],
                'tokens':
                    {k: task_results['n_tokens_total'][k] for k in task_results['n_tokens_total']
                     if 'tokens' in k} if task_results['n_tokens_total'] is not None else None,
                'approx_cost':
                    task_results['n_tokens_total']['cost'] if task_results['n_tokens_total'] is not None else None
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)


@router.post('/slide_subset', response_model=SlideSubsetResponse)
async def choose_best_subset(data: SlideSubsetRequest):
    slides_and_concepts = data.slides
    slides_and_concepts = {slide.number: slide.concepts for slide in slides_and_concepts}
    coverage = data.coverage
    min_freq = data.min_freq
    task_list = [choose_best_subset_task.s(slides_and_concepts, coverage, min_freq)]
    tasks = chain(task_list)
    results = tasks.apply_async(priority=6).get(timeout=300)
    return results
