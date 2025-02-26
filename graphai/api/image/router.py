from fastapi import APIRouter, Security, Depends
from fastapi_user_limiter.limiter import rate_limiter

from graphai.api.common.schemas import TaskIDResponse
from graphai.api.image.schemas import (
    RetrieveImageURLRequest,
    RetrieveImageURLResponse,
    UploadImageRequest,
    ImageFingerprintRequest,
    ImageFingerprintResponse,
    ExtractTextRequest,
    ExtractTextResponse,
    DetectOCRLanguageResponse,
)
from graphai.api.common.utils import format_api_results

from graphai.celery.image.jobs import (
    retrieve_image_from_url_job,
    upload_image_from_file_job,
    fingerprint_job,
    ocr_job
)
from graphai.api.auth.router import get_current_active_user, get_user_for_rate_limiter
from graphai.api.auth.auth_utils import get_ratelimit_values

from graphai.celery.common.celery_config import get_task_info

# Initialise video router
router = APIRouter(
    prefix='/image',
    tags=['image'],
    responses={404: {'description': 'Not found'}},
    dependencies=[Security(get_current_active_user, scopes=['image'])]
)


@router.post('/retrieve_url', response_model=TaskIDResponse,
             dependencies=[Depends(rate_limiter(get_ratelimit_values()['image']['max_requests'],
                                                get_ratelimit_values()['image']['window'],
                                                user=get_user_for_rate_limiter))])
async def retrieve_image_file(data: RetrieveImageURLRequest):
    # The URL to be retrieved
    url = data.url
    force = data.force
    task_id = retrieve_image_from_url_job(url, force)
    return {'task_id': task_id}


@router.post('/upload_file', response_model=TaskIDResponse,
             dependencies=[Depends(rate_limiter(get_ratelimit_values()['image']['max_requests'],
                                                get_ratelimit_values()['image']['window'],
                                                user=get_user_for_rate_limiter)),
                           Security(get_current_active_user, scopes=['upload'])])
async def upload_image_file(data: UploadImageRequest):
    # The URL to be retrieved
    contents = data.contents
    extension = data.file_extension
    origin = data.origin
    origin_info = {
        "id": data.origin_info.id,
        "name": data.origin_info.name
    }
    force = data.force
    task_id = upload_image_from_file_job(contents, extension, origin, origin_info, force)
    return {'task_id': task_id}


@router.get('/retrieve_url/status/{task_id}', response_model=RetrieveImageURLResponse)
@router.get('/upload_file/status/{task_id}', response_model=RetrieveImageURLResponse)
async def get_retrieve_image_file_status(task_id):
    full_results = get_task_info(task_id)
    task_results = full_results['results']
    if task_results is not None:
        if 'token' in task_results:
            task_results = {
                'token': task_results['token'],
                'token_status': task_results['token_status'],
                'token_size': task_results['token_size'],
                'fresh': task_results['fresh'],
                'error': task_results.get('error', None),
                'successful': task_results['token'] is not None
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)


@router.post('/calculate_fingerprint', response_model=TaskIDResponse)
async def calculate_slide_fingerprint(data: ImageFingerprintRequest):
    token = data.token
    force = data.force
    task_id = fingerprint_job(token, force)
    return {'task_id': task_id}


@router.get('/calculate_fingerprint/status/{task_id}', response_model=ImageFingerprintResponse)
async def calculate_slide_fingerprint_status(task_id):
    full_results = get_task_info(task_id)
    task_results = full_results['results']
    if task_results is not None:
        if 'result' in task_results:
            task_results = {
                'result': task_results['result'],
                'fresh': task_results['fresh'],
                'closest_token': task_results['closest'],
                'closest_token_origin': task_results['closest_origin'],
                'successful': task_results['result'] is not None
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)


@router.post('/extract_text', response_model=TaskIDResponse,
             dependencies=[Depends(rate_limiter(get_ratelimit_values()['image']['max_requests'],
                                                get_ratelimit_values()['image']['window'],
                                                user=get_user_for_rate_limiter))])
@router.post('/detect_language', response_model=TaskIDResponse,
             dependencies=[Depends(rate_limiter(get_ratelimit_values()['image']['max_requests'],
                                                get_ratelimit_values()['image']['window'],
                                                user=get_user_for_rate_limiter))])
async def extract_text(data: ExtractTextRequest):
    # Language detection requires OCR, so they have the same handler
    token = data.token
    method = data.method
    force = data.force
    api_token = data.google_api_token
    openai_token = data.openai_api_token
    pdf_in_pages = data.pdf_in_pages
    task_id = ocr_job(token, force, method, api_token, openai_token, pdf_in_pages)
    return {'task_id': task_id}


@router.get('/extract_text/status/{task_id}', response_model=ExtractTextResponse)
async def extract_text_status(task_id):
    full_results = get_task_info(task_id)
    task_results = full_results['results']
    if task_results is not None:
        if 'results' in task_results:
            task_results = {
                'result': task_results['results'],
                'language': task_results['language'],
                'fresh': task_results['fresh'],
                'successful': task_results['results'] is not None and not task_results['results'][0].get('fail', False)
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)


@router.get('/detect_language/status/{task_id}', response_model=DetectOCRLanguageResponse)
async def detect_ocr_language_status(task_id):
    full_results = get_task_info(task_id)
    task_results = full_results['results']
    if task_results is not None:
        if 'language' in task_results:
            task_results = {
                'language': task_results['language'],
                'fresh': task_results['fresh'],
                'successful': task_results['results'] is not None
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)
