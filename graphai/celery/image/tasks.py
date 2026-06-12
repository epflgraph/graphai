from celery import shared_task, group, chord, signature

from graphai.core.image.image import (
    cache_lookup_retrieve_image_from_url,
    retrieve_image_file_from_url,
    upload_image_from_file,
    retrieve_image_file_from_url_callback,
    cache_lookup_extract_slide_text,
    extract_slide_text,
    extract_slide_text_callback,
    break_pdf_into_images,
    extract_multi_image_text,
    collect_multi_image_ocr
)
from graphai.core.common.caching import (
    SlideDBCachingManager,
    VideoConfig
)
from graphai.core.common.lookup import fingerprint_cache_lookup_with_most_similar

file_management_config = VideoConfig()


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 2},
    name="caching.cache_lookup_retrieve_image",
    ignore_result=False,
    file_manager=file_management_config,
)
def cache_lookup_retrieve_image_from_url_task(self, url):
    return cache_lookup_retrieve_image_from_url(url, self.file_manager)


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 2},
    name="image.retrieve_image",
    ignore_result=False,
    file_manager=file_management_config,
)
def retrieve_image_from_url_task(self, url, force_token=None):
    return retrieve_image_file_from_url(url, self.file_manager, force_token)


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 2},
    name="image.upload_image",
    ignore_result=False,
    file_manager=file_management_config,
)
def upload_image_from_file_task(self, contents, file_extension):
    return upload_image_from_file(contents, file_extension, self.file_manager)


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 2},
    name="image.retrieve_image_callback",
    ignore_result=False,
    file_manager=file_management_config,
)
def retrieve_image_from_url_callback_task(self, results, url):
    return retrieve_image_file_from_url_callback(results, url)


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 2},
    name="caching.cache_lookup_fingerprint_slide",
    ignore_result=False,
)
def cache_lookup_slide_fingerprint_task(self, token):
    return fingerprint_cache_lookup_with_most_similar(
        token, SlideDBCachingManager(), None
    )


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 2},
    name="caching.cache_lookup_extract_slide_text",
    ignore_result=False,
)
def cache_lookup_extract_slide_text_task(self, token, method="tesseract"):
    return cache_lookup_extract_slide_text(token, method)


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 2},
    name="image.extract_slide_text",
    ignore_result=False,
    file_manager=file_management_config,
)
def extract_slide_text_task(
    self,
    token,
    method="google",
    google_api_token=None,
    openai_api_token=None,
    gemini_api_token=None,
    rcp_api_token=None,
    model_type=None,
    enable_tikz=False,
):
    return extract_slide_text(
        token,
        self.file_manager,
        method,
        google_api_token,
        openai_api_token,
        gemini_api_token,
        rcp_api_token,
        model_type,
        enable_tikz,
    )


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 2},
    name="image.pdf_to_pages",
    ignore_result=False,
    file_manager=file_management_config,
)
def convert_pdf_to_pages_task(self, token):
    print(f'Starting {convert_pdf_to_pages_task} task for token {token}')
    return break_pdf_into_images(token, self.file_manager)


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 2},
    name="image.fanout_pdf_ocr_task",
    ignore_result=False,
)
def fanout_pdf_ocr_task(
    self,
    pdf_pages_payload,
    method,
    google_api_token=None,
    openai_api_token=None,
    gemini_api_token=None,
    rcp_api_token=None,
    model_type=None,
    enable_tikz=False,
):
    if pdf_pages_payload.get('file_found') is False or not pdf_pages_payload.get('pages'):
        return {
            'results': None,
            'language': None,
            'fresh': False,
            'file_found': pdf_pages_payload.get('file_found')
        }

    # Build one OCR task per page.
    page_ocr_tasks = [
        signature(
            'image.extract_multi_image_text',
            args=(
                page,
                method,
                google_api_token,
                openai_api_token,
                gemini_api_token,
                rcp_api_token,
                model_type,
                enable_tikz,
            ),
        )
        for page in pdf_pages_payload['pages']
    ]
    header = group(page_ocr_tasks)

    # When all pages are OCR'd, collect results.
    callback = signature(
        'image.extract_multi_image_text_callback',
        args=(pdf_pages_payload.get('file_found'),),
    )

    # Replace this task with the chord so the outer chain waits properly.
    raise self.replace(chord(header, callback))


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 2},
    name="image.extract_multi_image_text",
    ignore_result=False,
)
def extract_multi_image_text_task(
    self,
    page_and_filename,
    method="google",
    google_api_token=None,
    openai_api_token=None,
    gemini_api_token=None,
    rcp_api_token=None,
    model_type=None,
    enable_tikz=False,
):
    print(f'Starting {extract_multi_image_text_task} task for page_and_filename {page_and_filename}')
    return extract_multi_image_text(
        page_and_filename,
        method,
        google_api_token,
        openai_api_token,
        gemini_api_token,
        rcp_api_token,
        model_type,
        enable_tikz,
    )


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 2},
    name="image.extract_multi_image_text_callback",
    ignore_result=False,
)
def collect_multi_image_ocr_task(self, results, file_found=True):
    print(f'Starting {collect_multi_image_ocr_task} task for results {results}')
    return collect_multi_image_ocr(results, file_found)


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 2},
    name="image.extract_slide_text_callback",
    ignore_result=False,
)
def extract_slide_text_callback_task(self, results, token, force=False):
    return extract_slide_text_callback(results, token, force)
