from celery import shared_task

from graphai.api.celery_tasks.common import fingerprint_lookup_retrieve_from_db, fingerprint_lookup_parallel, \
    fingerprint_lookup_callback
from graphai.api.common.video import slide_db_manager, file_management_config
from graphai.core.common.video import perceptual_hash_image, read_txt_gz_file, write_txt_gz_file, perform_tesseract_ocr, \
    GoogleOCRModel


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.slide_fingerprint', ignore_result=False,
             db_manager=slide_db_manager, file_manager=file_management_config)
def compute_slide_fingerprint_task(self, token, force=False):
    # Checking for existing cached results
    existing_slide = self.db_manager.get_details(token, cols=['fingerprint'])
    if existing_slide is None:
        return {
            'result': None,
            'fresh': False
        }
    if not force and existing_slide['fingerprint'] is not None:
        return {
            'result': existing_slide['fingerprint'],
            'fresh': False
        }
    slide_with_path = self.file_manager.generate_filepath(token)
    fingerprint = perceptual_hash_image(slide_with_path)
    if fingerprint is None:
        return {
            'result': None,
            'fresh': False
        }
    return {
        'result': fingerprint,
        'fresh': True
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.slide_fingerprint_callback', ignore_result=False,
             db_manager=slide_db_manager)
def compute_slide_fingerprint_callback_task(self, results, token):
    if results['fresh']:
        self.db_manager.insert_or_update_details(
            token,
            {
                'fingerprint': results['result'],
            }
        )
    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.slide_fingerprint_find_closest_retrieve_from_db', ignore_result=False,
             db_manager=slide_db_manager)
def slide_fingerprint_find_closest_retrieve_from_db_task(self, results, token):
    return fingerprint_lookup_retrieve_from_db(results, token, self.db_manager)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.slide_fingerprint_find_closest_parallel', ignore_result=False,
             db_manager=slide_db_manager)
def slide_fingerprint_find_closest_parallel_task(self, input_dict, i, n_total, min_similarity=1):
    return fingerprint_lookup_parallel(input_dict, i, n_total, min_similarity, data_type='image')


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.slide_fingerprint_find_closest_callback', ignore_result=False,
             db_manager=slide_db_manager)
def slide_fingerprint_find_closest_callback_task(self, results_list, original_token):
    return fingerprint_lookup_callback(results_list, original_token, self.db_manager)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.retrieve_slide_fingerprint_final_callback', ignore_result=False)
def retrieve_slide_fingerprint_callback_task(self, results):
    # Returning the fingerprinting results, which is the part of this task whose results are sent back to the user.
    return results['fp_results']


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.extract_slide_text', ignore_result=False,
             db_manager=slide_db_manager, file_manager=file_management_config)
def extract_slide_text_task(self, token, method='tesseract', force=False):
    if method == 'tesseract':
        ocr_colnames = ['ocr_tesseract_token']
    else:
        ocr_colnames = ['ocr_google_1_token', 'ocr_google_2_token']

    if not force:
        existing = self.db_manager.get_details(token, ocr_colnames,
                                               using_most_similar=True)
        if existing is None:
            return {
                'results': None,
                'fresh': False
            }
        if all([existing[ocr_colname] is not None for ocr_colname in ocr_colnames]):
            print('Returning cached result')
            results = [
                {
                    'method': ocr_colname,
                    'token': existing[ocr_colname],
                    'text': read_txt_gz_file(self.file_manager.generate_filepath(existing[ocr_colname]))
                }
                for ocr_colname in ocr_colnames
            ]

            return {
                'results': results,
                'fresh': False
            }
    if method == 'tesseract':
        res = perform_tesseract_ocr(self.file_manager.generate_filepath(token))
        if res is None:
            results = None
        else:
            res_token = token+'_'+ocr_colnames[0]+'.txt.gz'
            write_txt_gz_file(res, self.file_manager.generate_filepath(res_token))
            results = [
                {
                    'method': ocr_colnames[0],
                    'token': res_token,
                    'text': res
                }
            ]
    else:
        ocr_model = GoogleOCRModel()
        ocr_model.establish_connection()
        res1, res2 = ocr_model.perform_ocr(self.file_manager.generate_filepath(token))
        if res1 is None or res2 is None:
            results = None
        else:
            res_list = [res1, res2]
            res_token_list = list()
            for i in range(len(res_list)):
                current_token = token+'_'+ocr_colnames[i]+'.txt.gz'
                write_txt_gz_file(res_list[i], self.file_manager.generate_filepath(current_token))
                res_token_list.append(current_token)
            results = [
                {
                    'method': ocr_colnames[i],
                    'token': res_token_list[i],
                    'text': res_list[i]
                }
                for i in range(len(res_list))
            ]
    return {
        'results': results,
        'fresh': results is not None
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video.extract_slide_text_callback', ignore_result=False,
             db_manager=slide_db_manager)
def extract_slide_text_callback_task(self, results, token):
    if results['fresh']:
        values_dict = {
            result['method']: result['token']
            for result in results['results']
        }
        # Inserting values for original token
        self.db_manager.insert_or_update_details(
            token, values_dict
        )
        # Inserting the same values for closest token if different than original token
        closest = self.db_manager.get_closest_match(token)
        if closest is not None and closest != token:
            self.db_manager.insert_or_update_details(
                closest, values_dict
            )
    return results


