from celery import shared_task

from graphai.api.common.video import file_management_config
from graphai.core.common.video import (
    perform_tesseract_ocr,
    GoogleOCRModel,
    get_ocr_colnames
)
from graphai.core.common.text_utils import detect_text_language
from graphai.core.common.caching import SlideDBCachingManager


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='caching_6.cache_lookup_fingerprint_slide', ignore_result=False)
def cache_lookup_slide_fingerprint_task(self, token):
    db_manager = SlideDBCachingManager()
    existing_list = db_manager.get_details(token, cols=['fingerprint'],
                                           using_most_similar=True)
    for existing in existing_list:
        if existing is None:
            continue
        if existing['fingerprint'] is not None:
            # We have a cache hit, now we gather all the results that should be returned
            # The closest match
            existing_closest = db_manager.get_closest_match(token)
            if existing_closest is not None:
                # If the closest match exists, we also want to return its origin token
                existing_closest_origin = db_manager.get_origin(existing_closest)
            else:
                existing_closest_origin = None
            print('Returning cached result')
            return {
                'result': existing['fingerprint'],
                'fresh': False,
                'closest': existing_closest,
                'closest_origin': existing_closest_origin
            }

    return None


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='caching_6.cache_lookup_extract_slide_text', ignore_result=False)
def cache_lookup_extract_slide_text_task(self, token, method='tesseract'):
    ocr_colnames = get_ocr_colnames(method)

    db_manager = SlideDBCachingManager()
    existing_list = db_manager.get_details(token, ocr_colnames + ['language'],
                                           using_most_similar=True)
    # Checking whether the token even exists
    if existing_list[0] is None:
        return {
            'results': None,
            'language': None,
            'fresh': False
        }

    for existing in existing_list:
        if existing is None:
            continue

        if all([existing[ocr_colname] is not None for ocr_colname in ocr_colnames]):
            print('Returning cached result')
            results = [
                {
                    'method': ocr_colname,
                    'text': existing[ocr_colname],
                }
                for ocr_colname in ocr_colnames
            ]
            language = existing['language']
            fresh = False

            if language is None:
                language = detect_text_language(results[0]['text'])
                fresh = True

            return {
                'results': results,
                'language': language,
                'fresh': fresh
            }

    return None


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.extract_slide_text', ignore_result=False,
             file_manager=file_management_config)
def extract_slide_text_task(self, token, method='tesseract'):
    ocr_colnames = get_ocr_colnames(method)

    if method == 'tesseract':
        res = perform_tesseract_ocr(self.file_manager.generate_filepath(token))
        if res is None:
            results = None
            language = None
        else:
            language = detect_text_language(res)
            results = [
                {
                    'method': ocr_colnames[0],
                    'text': res
                }
            ]
    else:
        ocr_model = GoogleOCRModel()
        ocr_model.establish_connection()
        res1, res2 = ocr_model.perform_ocr(self.file_manager.generate_filepath(token))

        if res1 is None or res2 is None:
            results = None
            language = None
        else:
            # Since DTD usually performs better, method #1 is our point of reference for langdetect
            language = detect_text_language(res1)
            res_list = [res1, res2]
            results = [
                {
                    'method': ocr_colnames[i],
                    'text': res_list[i]
                }
                for i in range(len(res_list))
            ]

    return {
        'results': results,
        'language': language,
        'fresh': results is not None
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.extract_slide_text_callback', ignore_result=False)
def extract_slide_text_callback_task(self, results, token, force=False):
    if results['fresh']:
        values_dict = {
            result['method']: result['text']
            for result in results['results']
        }

        values_dict.update({'language': results['language']})
        db_manager = SlideDBCachingManager()

        # Inserting values for original token
        db_manager.insert_or_update_details(
            token, values_dict
        )
        if not force:
            # Inserting the same values for closest token if different than original token
            # Happens if the other token has been fingerprinted before being OCRed, or if the two tokens
            # have identical but separate parent videos.
            closest = db_manager.get_closest_match(token)
            if closest is not None and closest != token:
                db_manager.insert_or_update_details(
                    closest, values_dict
                )
    return results
