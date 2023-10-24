from fastapi import APIRouter
from celery import group, chain

from graphai.api.schemas.common import TaskIDResponse
from graphai.api.schemas.translation import (
    TranslationRequest,
    TranslationResponse,
    TextDetectLanguageRequest,
    TextDetectLanguageResponse,
    TextFingerprintRequest,
    TextFingerprintResponse,
)

from graphai.api.celery_tasks.translation import (
    translate_text_task,
    translate_text_callback_task,
    detect_text_language_task,
    compute_translation_text_fingerprint_task,
    compute_translation_text_fingerprint_callback_task,
    translation_text_fingerprint_find_closest_retrieve_from_db_task,
    translation_text_fingerprint_find_closest_parallel_task,
    translation_text_fingerprint_find_closest_direct_task,
    translation_text_fingerprint_find_closest_callback_task,
    translation_retrieve_text_fingerprint_callback_task,
)
from graphai.api.celery_tasks.common import (
    format_api_results,
    ignore_fingerprint_results_callback_task,
)
from graphai.core.common.text_utils import generate_src_tgt_dict, generate_translation_text_token, \
    translation_list_to_text, translation_text_back_to_list
from graphai.core.common.caching import FingerprintParameters
from graphai.core.interfaces.celery_config import get_task_info


router = APIRouter(
    prefix='/translation',
    tags=['translation'],
    responses={404: {'description': 'Not found'}}
)


def get_translation_text_fingerprint_chain_list(token, text, src, tgt, force, min_similarity=None, n_jobs=8,
                                                ignore_fp_results=False, results_to_return=None):
    # Loading min similarity parameter for text
    if min_similarity is None:
        fp_parameters = FingerprintParameters()
        min_similarity = fp_parameters.get_min_sim_text()

    # Generating the equality condition dictionary
    equality_conditions = generate_src_tgt_dict(src, tgt)
    # The tasks are fingerprinting and callback, then lookup. The lookup is only among cache rows that satisfy the
    # equality conditions (source and target languages).
    task_list = [
        compute_translation_text_fingerprint_task.s(token, text, force),
        compute_translation_text_fingerprint_callback_task.s(text, src, tgt)
    ]
    if min_similarity == 1:
        task_list += [translation_text_fingerprint_find_closest_direct_task.s(equality_conditions)]
    else:
        task_list += [
            translation_text_fingerprint_find_closest_retrieve_from_db_task.s(equality_conditions),
            group(translation_text_fingerprint_find_closest_parallel_task.s(i, n_jobs, equality_conditions, min_similarity)
                  for i in range(n_jobs))
        ]
    task_list += [translation_text_fingerprint_find_closest_callback_task.s()]
    if ignore_fp_results:
        task_list += [ignore_fingerprint_results_callback_task.s(results_to_return)]
    else:
        task_list += [translation_retrieve_text_fingerprint_callback_task.s()]
    return task_list


@router.post('/calculate_fingerprint', response_model=TaskIDResponse)
async def calculate_translation_text_fingerprint(data: TextFingerprintRequest):
    text = data.text
    src = data.source
    tgt = data.target
    force = data.force
    token = generate_translation_text_token(text, src, tgt)
    text = translation_list_to_text(text)
    task_list = get_translation_text_fingerprint_chain_list(token, text, src, tgt, force,
                                                            ignore_fp_results=False)
    task = chain(task_list)
    task = task.apply_async(priority=6)
    return {'task_id': task.id}


@router.get('/calculate_fingerprint/status/{task_id}', response_model=TextFingerprintResponse)
async def calculate_translation_text_fingerprint_status(task_id):
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


@router.post('/translate', response_model=TaskIDResponse)
async def translate(data: TranslationRequest):
    text = data.text
    src = data.source
    tgt = data.target
    force = data.force
    token = generate_translation_text_token(text, src, tgt)
    return_list = isinstance(text, list)
    text = translation_list_to_text(text)
    # If force=True, fingerprinting is skipped
    # The tasks are translation and its callback
    if not force:
        task_list = get_translation_text_fingerprint_chain_list(token, text, src, tgt, force,
                                                                ignore_fp_results=True, results_to_return=token)
        task_list += [translate_text_task.s(text, src, tgt, force)]
    else:
        task_list = [translate_text_task.s(token, text, src, tgt, force)]
    task_list += [translate_text_callback_task.s(token, text, src, tgt, force, return_list)]
    task = chain(task_list)
    task = task.apply_async(priority=6)
    return {'task_id': task.id}


@router.get('/translate/status/{task_id}', response_model=TranslationResponse)
async def translate_status(task_id):
    full_results = get_task_info(task_id)
    task_results = full_results['results']
    if task_results is not None:
        if 'result' in task_results:
            task_results = {
                'result': translation_text_back_to_list(task_results['result'],
                                                        return_list=task_results['return_list']),
                'text_too_large': task_results['text_too_large'],
                'successful': task_results['successful'],
                'fresh': task_results['fresh'],
                'device': task_results['device']
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)


@router.post('/detect_language', response_model=TaskIDResponse)
async def text_detect_language(data: TextDetectLanguageRequest):
    text = data.text
    text = translation_list_to_text(text)
    # The only task is language detection because this task does not go through the caching logic
    task = (detect_text_language_task.s(text)).apply_async(priority=6)
    return {'task_id': task.id}


@router.get('/detect_language/status/{task_id}', response_model=TextDetectLanguageResponse)
async def text_detect_language_status(task_id):
    full_results = get_task_info(task_id)
    task_results = full_results['results']
    if task_results is not None:
        if 'language' in task_results:
            task_results = {
                'language': task_results['language'],
                'successful': task_results['successful']
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)
