from fastapi import APIRouter
from fastapi.responses import FileResponse

from graphai.api.schemas.video import *
from graphai.api.schemas.common import *

from graphai.api.celery_tasks.video import retrieve_and_generate_token_master, \
    get_file_master, extract_audio_master
from graphai.core.celery_utils.celery_utils import get_task_info


# Initialise video router
router = APIRouter(
    prefix='/video',
    tags=['video'],
    responses={404: {'description': 'Not found'}}
)


@router.post('/retrieve_url', response_model=TaskIDResponse)
async def retrieve_file(data: RetrieveURLRequest):
    result = retrieve_and_generate_token_master(data.url)
    return result


# For each async endpoint, we also have a status endpoint since they have different response models.
@router.get('/retrieve_url/status/{task_id}', response_model=RetrieveURLResponse)
async def get_retrieve_file_status(task_id):
    return get_task_info(task_id)


# @router.post('/calculate_fingerprint', response_model=TaskIDResponse)
# async def calculate_fingerprint(data: ComputeSignatureRequest):
#     result = compute_signature_master(data.token, force=data.force)
#     return result


@router.get('/calculate_fingerprint/status/{task_id}', response_model=ComputeSignatureResponse)
async def calculate_fingerprint_status(task_id):
    return get_task_info(task_id)


@router.post('/get_file/')
async def get_file(data: FileRequest):
    return FileResponse(get_file_master(data.token))


@router.post('/extract_audio', response_model=TaskIDResponse)
async def extract_audio(data: ExtractAudioRequest):
    result = extract_audio_master(data.token, force=data.force)
    return result


@router.get('/extract_audio/status/{task_id}', response_model=ExtractAudioResponse)
async def extract_audio_status(task_id):
    return get_task_info(task_id)


# @router.post('/detect_slides', response_model=TaskIDResponse)
# async def detect_slides(data: DetectSlidesRequest):
#     result = detect_slides_master(data.token, force=data.force)
#     return result


@router.get('/detect_slides/status/{task_id}', response_model=DetectSlidesResponse)
async def detect_slides_status(task_id):
    return get_task_info(task_id)