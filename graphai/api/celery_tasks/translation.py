from celery import shared_task
from datetime import datetime
from graphai.api.common.video import translation_models
from graphai.core.common.video import detect_text_language, perceptual_hash_text
from graphai.core.common.caching import TextDBCachingManager
from graphai.api.celery_tasks.common import fingerprint_lookup_retrieve_from_db, \
    fingerprint_lookup_parallel, fingerprint_lookup_callback

LONG_TEXT_ERROR = "Unpunctuated text too long (over 512 tokens), " \
                  "try adding punctuation or providing a smaller chunk of text."


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.fingerprint_text', ignore_result=False)
def compute_text_fingerprint_task(self, token, text, force=False):
    db_manager = TextDBCachingManager()
    existing = db_manager.get_details(token, cols=['fingerprint'])[0]

    if existing is not None and not force:
        if existing['fingerprint'] is not None:
            return {
                'result': existing['fingerprint'],
                'fp_token': existing['id_token'],
                'perform_lookup': False,
                'fresh': False
            }

    fp = perceptual_hash_text(text)

    return {
        'result': fp,
        'fp_token': token,
        'perform_lookup': True,
        'fresh': True
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.fingerprint_text_callback', ignore_result=False)
def compute_text_fingerprint_callback_task(self, results, text, src, tgt):
    if results['fresh']:
        token = results['fp_token']
        db_manager = TextDBCachingManager()
        values_dict = {
            'fingerprint': results['result'],
            'source': text,
            'source_lang': src,
            'target_lang': tgt
        }
        existing = db_manager.get_details(token, ['date_added'], using_most_similar=False)[0]
        if existing is None or existing['date_added'] is None:
            current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            values_dict['date_added'] = current_datetime
        db_manager.insert_or_update_details(token, values_dict)
    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.text_fingerprint_find_closest_retrieve_from_db', ignore_result=False)
def text_fingerprint_find_closest_retrieve_from_db_task(self, results, equality_conditions):
    db_manager = TextDBCachingManager()
    return fingerprint_lookup_retrieve_from_db(results, db_manager, equality_conditions=equality_conditions)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.text_fingerprint_find_closest_parallel', ignore_result=False)
def text_fingerprint_find_closest_parallel_task(self, input_dict, i, n_total, equality_conditions,
                                                min_similarity=1):
    db_manager = TextDBCachingManager()
    return fingerprint_lookup_parallel(input_dict, i, n_total, min_similarity, db_manager, data_type='text',
                                       equality_conditions=equality_conditions)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.text_fingerprint_find_closest_callback', ignore_result=False)
def text_fingerprint_find_closest_callback_task(self, results_list):
    db_manager = TextDBCachingManager()
    return fingerprint_lookup_callback(results_list, db_manager)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.retrieve_text_fingerprint_final_callback', ignore_result=False)
def retrieve_text_fingerprint_callback_task(self, results):
    # Returning the fingerprinting results, which is the part of this task whose results are sent back to the user.
    results_to_return = results['fp_results']
    results_to_return['closest'] = results['closest']
    return results_to_return


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.translate_text', translation_obj=translation_models, ignore_result=False)
def translate_text_task(self, token, text, src, tgt, force):
    if src == tgt:
        return {
            'result': "'source' and 'target' languages must be different!",
            'text_too_large': False,
            'successful': False,
            'fresh': False
        }

    db_manager = TextDBCachingManager()
    if not force:
        existing_list = db_manager.get_details(token, ['target'], using_most_similar=True)
        # Unlike with audio and image, the token may not already exist in the table when this task is invoked.
        #  Therefore, the task doesn't fail if the token doesn't exist.
        for existing in existing_list:
            if existing is None:
                continue
            if existing['target'] is not None:
                print('Returning cached result')
                return {
                    'result': existing['target'],
                    'text_too_large': False,
                    'successful': True,
                    'fresh': False
                }

    how = f"{src}-{tgt}"
    try:
        translated_text, large_warning = self.translation_obj.translate(text, how=how)
        if translated_text is not None:
            success = True
        else:
            success = False
        if large_warning:
            translated_text = LONG_TEXT_ERROR
    except NotImplementedError as e:
        print(e)
        translated_text = str(e)
        success = False
        large_warning = False

    return {
        'result': translated_text,
        'text_too_large': large_warning,
        'successful': success,
        'fresh': success
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.translate_text_callback', translation_obj=translation_models, ignore_result=False)
def translate_text_callback_task(self, results, token, text, src, tgt, force=False):
    db_manager = TextDBCachingManager()
    if results['fresh']:
        values_dict = {
            'source': text,
            'target': results['result'],
            'source_lang': src,
            'target_lang': tgt
        }
        existing = db_manager.get_details(token, ['date_added'], using_most_similar=False)[0]
        if existing is None or existing['date_added'] is None:
            current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            values_dict['date_added'] = current_datetime
        # Inserting values for original token
        db_manager.insert_or_update_details(
            token, values_dict
        )
        if not force:
            # Inserting the same values for closest token if different than original token
            closest = db_manager.get_closest_match(token)
            if closest is not None and closest != token:
                db_manager.insert_or_update_details(
                    closest, values_dict
                )
    elif not results['successful']:
        # in case we fingerprinted something and then failed to translate it, we delete its cache row
        db_manager.delete_cache_rows([token])

    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.detect_text_language', ignore_result=False)
def detect_text_language_task(self, text):
    result = detect_text_language(text)
    return {
        'language': result,
        'successful': result is not None
    }
