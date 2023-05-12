from fastapi import APIRouter
from ..schemas.common import TaskIDResponse
from ..schemas.translation import TranslationRequest, TranslationResponse, \
    TextDetectLanguageRequest, TextDetectLanguageResponse
from ..celery_tasks.translation import translate_text_task, detect_text_language_task
from graphai.core.interfaces.celery_config import get_task_info
from ..celery_tasks.common import format_api_results


router = APIRouter(
    prefix='/translation',
    tags=['translation'],
    responses={404: {'description': 'Not found'}}
)


@router.post('/translate/', response_model=TaskIDResponse)
async def translate(data: TranslationRequest):
    text = data.text
    src = data.source
    tgt = data.target
    task = (translate_text_task.s(text, src, tgt)).apply_async(priority=6)
    return {'task_id': task.id}


@router.get('/translate/status/{task_id}', response_model=TranslationResponse)
async def translate_status(task_id):
    full_results = get_task_info(task_id)
    task_results = full_results['results']
    if task_results is not None:
        if 'result' in task_results:
            task_results = {
                'result': task_results['result'],
                'successful': task_results['successful']
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