from celery import shared_task

from graphai.api.celery_tasks.common import fingerprint_lookup_retrieve_from_db, fingerprint_lookup_parallel, \
    fingerprint_lookup_callback, fingerprint_lookup_direct
from graphai.api.common.video import file_management_config
from graphai.core.common.video import perceptual_hash_image, read_txt_gz_file, write_txt_gz_file, perform_tesseract_ocr, \
    GoogleOCRModel
from graphai.core.common.text_utils import detect_text_language
from graphai.core.common.caching import SlideDBCachingManager


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.slide_fingerprint', ignore_result=False,
             file_manager=file_management_config)
def compute_slide_fingerprint_task(self, token, force=False):
    # Checking for existing cached results
    db_manager = SlideDBCachingManager()
    existing_slide_list = db_manager.get_details(token, cols=['fingerprint'], using_most_similar=True)
    if existing_slide_list[0] is None:
        return {
            'result': None,
            'fp_token': None,
            'perform_lookup': False,
            'fresh': False
        }
    if not force:
        for existing_slide in existing_slide_list:
            if existing_slide is None:
                continue
            if existing_slide['fingerprint'] is not None:
                return {
                    'result': existing_slide['fingerprint'],
                    'fp_token': existing_slide['id_token'],
                    'perform_lookup': False,
                    'fresh': False
                }
    slide_with_path = self.file_manager.generate_filepath(token)
    fingerprint = perceptual_hash_image(slide_with_path)
    if fingerprint is None:
        return {
            'result': None,
            'fp_token': None,
            'perform_lookup': False,
            'fresh': False
        }

    return {
        'result': fingerprint,
        'fp_token': token,
        'perform_lookup': True,
        'fresh': True
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.slide_fingerprint_callback', ignore_result=False)
def compute_slide_fingerprint_callback_task(self, results, force=False):
    if results['fresh']:
        token = results['fp_token']
        db_manager = SlideDBCachingManager()
        db_manager.insert_or_update_details(
            token,
            {
                'fingerprint': results['result'],
            }
        )
        if not force:
            closest_token = db_manager.get_closest_match(token)
            # If this token has a closest token, it means that their relationship comes from their parent videos,
            # and that the closest token's fingerprint has not been calculated either (otherwise `fresh` wouldn't be True).
            # In that case, we insert the computed fingerprint for the closest token as well, and then we will perform the
            # fingerprint lookup for that token instead of the one we computed the fingerprint for.
            if closest_token is not None and closest_token != token:
                db_manager.insert_or_update_details(
                    closest_token,
                    {
                        'fingerprint': results['result'],
                    }
                )
                results['fp_token'] = closest_token
    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.slide_fingerprint_find_closest_retrieve_from_db', ignore_result=False)
def slide_fingerprint_find_closest_retrieve_from_db_task(self, results):
    db_manager = SlideDBCachingManager()
    return fingerprint_lookup_retrieve_from_db(results, db_manager)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.slide_fingerprint_find_closest_parallel', ignore_result=False)
def slide_fingerprint_find_closest_parallel_task(self, input_dict, i, n_total, min_similarity=1):
    db_manager = SlideDBCachingManager()
    return fingerprint_lookup_parallel(input_dict, i, n_total, min_similarity, db_manager, data_type='image')


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.slide_fingerprint_find_closest_direct', ignore_result=False)
def slide_fingerprint_find_closest_direct_task(self, results):
    db_manager = SlideDBCachingManager()
    return fingerprint_lookup_direct(results, db_manager)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.slide_fingerprint_find_closest_callback', ignore_result=False)
def slide_fingerprint_find_closest_callback_task(self, results_list):
    db_manager = SlideDBCachingManager()
    return fingerprint_lookup_callback(results_list, db_manager)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.retrieve_slide_fingerprint_final_callback', ignore_result=False)
def retrieve_slide_fingerprint_callback_task(self, results):
    # Returning the fingerprinting results, which is the part of this task whose results are sent back to the user.
    results_to_return = results['fp_results']
    results_to_return['closest'] = results['closest']
    db_manager = SlideDBCachingManager()

    if results_to_return['closest'] is not None:
        results_to_return['closest_origin'] = db_manager.get_origin(results_to_return['closest'])
    else:
        results_to_return['closest_origin'] = None

    return results_to_return


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.extract_slide_text', ignore_result=False,
             file_manager=file_management_config)
def extract_slide_text_task(self, token, method='tesseract', force=False):
    if method == 'tesseract':
        ocr_colnames = ['ocr_tesseract_results']
    else:
        ocr_colnames = ['ocr_google_1_results', 'ocr_google_2_results']

    if not force:
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
