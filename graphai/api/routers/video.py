from fastapi import APIRouter
from fastapi.responses import FileResponse

from graphai.api.schemas.video import *
from graphai.api.schemas.common import *

from graphai.api.celery_tasks.video import retrieve_and_generate_token_master, \
    get_file_master, extract_audio_master, detect_slides_master
from graphai.api.celery_tasks.common import format_api_results
from graphai.core.interfaces.celery_config import get_task_info

# Initialise video router
router = APIRouter(
    prefix='/video',
    tags=['video'],
    responses={404: {'description': 'Not found'}}
)


@router.post('/retrieve_url', response_model=TaskIDResponse)
async def retrieve_file(data: RetrieveURLRequest):
    result = retrieve_and_generate_token_master(data.url)
    return {'task_id': result['id']}


# For each async endpoint, we also have a status endpoint since they have different response models.
@router.get('/retrieve_url/status/{task_id}', response_model=RetrieveURLResponse)
async def get_retrieve_file_status(task_id):
    full_results = get_task_info(task_id)
    task_results = full_results['results']
    if task_results is not None:
        if 'token' in task_results:
            task_results = {
                'token': task_results['token'],
                'successful': task_results['token'] is not None
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)


@router.post('/get_file/')
async def get_file(data: FileRequest):
    return FileResponse(get_file_master(data.token))


@router.post('/extract_audio', response_model=TaskIDResponse)
async def extract_audio(data: ExtractAudioRequest):
    result = extract_audio_master(data.token, force=data.force)
    return {'task_id': result['id']}


@router.get('/extract_audio/status/{task_id}', response_model=ExtractAudioResponse)
async def extract_audio_status(task_id):
    full_results = get_task_info(task_id)
    task_results = full_results['results']
    if task_results is not None:
        if 'token' in task_results:
            task_results = {
                'token': task_results['token'],
                'fresh': task_results['fresh'],
                'duration': task_results['duration'],
                'successful': task_results['token'] is not None
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)


@router.post('/detect_slides', response_model=TaskIDResponse)
async def detect_slides(data: DetectSlidesRequest):
    result = detect_slides_master(data.token, force=data.force, language=data.language)
    return {'task_id': result['id']}