from celery import group, chain
from fastapi import APIRouter
from graphai.api.schemas.common import TaskIDResponse
from graphai.api.schemas.translation import TranslationRequest, TranslationResponse, \
    TextDetectLanguageRequest, TextDetectLanguageResponse
from graphai.api.celery_tasks.translation import translate_text_task, translate_text_callback_task, \
    detect_text_language_task, compute_text_fingerprint_task, compute_text_fingerprint_callback_task, \
    text_fingerprint_find_closest_retrieve_from_db_task, text_fingerprint_find_closest_parallel_task, \
    text_fingerprint_find_closest_callback_task, retrieve_text_fingerprint_callback_task
from graphai.core.interfaces.celery_config import get_task_info
from graphai.api.celery_tasks.common import format_api_results, ignore_fingerprint_results_callback_task
from graphai.core.common.video import md5_text


router = APIRouter(
    prefix='/translation',
    tags=['translation'],
    responses={404: {'description': 'Not found'}}
)


def generate_src_tgt_dict(src, tgt):
    return {'source_lang': src, 'target_lang': tgt}


def generate_text_token(s, src, tgt):
    return md5_text(s) + '_' + src + '_' + tgt


def get_text_fingerprint_chain_list(token, text, src, tgt, force, min_similarity=0.99, n_jobs=8,
                                     ignore_fp_results=False, results_to_return=None):
    equality_conditions = generate_src_tgt_dict(src, tgt)
    task_list = [
        compute_text_fingerprint_task.s(token, text, force),
        compute_text_fingerprint_callback_task.s(token),
        text_fingerprint_find_closest_retrieve_from_db_task.s(token, equality_conditions),
        group(text_fingerprint_find_closest_parallel_task.s(token, i, n_jobs, equality_conditions, min_similarity)
              for i in range(n_jobs)),
        text_fingerprint_find_closest_callback_task.s(token)
    ]
    if ignore_fp_results:
        task_list += [ignore_fingerprint_results_callback_task.s(results_to_return)]
    else:
        task_list += [retrieve_text_fingerprint_callback_task.s()]
    return task_list


@router.post('/translate/', response_model=TaskIDResponse)
async def translate(data: TranslationRequest):
    text = data.text
    src = data.source
    tgt = data.target
    force = data.force
    token = generate_text_token(text, src, tgt)
    if not force:
        task_list = get_text_fingerprint_chain_list(token, text, src, tgt, force,
                                                    ignore_fp_results=True, results_to_return=token)
        task_list += [translate_text_task.s(text, src, tgt, force)]
    else:
        task_list = [translate_text_task.s(token, text, src, tgt, force)]
    task_list += [translate_text_callback_task.s(token, text, src, tgt)]
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
                'result': task_results['result'],
                'text_too_large': task_results['text_too_large'],
                'successful': task_results['successful'],
                'fresh': task_results['fresh']
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)


@router.post('/detect_language/', response_model=TaskIDResponse)
async def text_detect_language(data: TextDetectLanguageRequest):
    text = data.text
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