from celery import shared_task
from graphai.api.common.translation import translation_models
from graphai.core.translation.text_utils import (
    detect_text_language,
    translation_text_back_to_list
)
from graphai.core.common.fingerprinting import perceptual_hash_text
from graphai.core.common.common_utils import get_current_datetime
from graphai.core.interfaces.caching import TextDBCachingManager

LONG_TEXT_ERROR = "Unpunctuated text too long (over 512 tokens), " \
                  "try adding punctuation or providing a smaller chunk of text."


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='caching_6.cache_lookup_fingerprint_translation_text', ignore_result=False)
def cache_lookup_translation_text_fingerprint_task(self, token):
    db_manager = TextDBCachingManager()
    existing = db_manager.get_details(token, cols=['fingerprint'])[0]
    if existing is not None and existing['fingerprint'] is not None:
        existing_closest = db_manager.get_closest_match(token)
        return {
            'result': existing['fingerprint'],
            'closest': existing_closest,
            'fresh': False
        }
    return None


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.fingerprint_translation_text', ignore_result=False)
def compute_translation_text_fingerprint_task(self, token, text):
    fp = perceptual_hash_text(text)

    return {
        'result': fp,
        'token': token,
        'fresh': True
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.fingerprint_translation_text_callback', ignore_result=False)
def compute_translation_text_fingerprint_callback_task(self, results, text, src, tgt):
    # This task does not have the condition of the 'fresh' flag being True because text fingerprinting never fails
    fp = results['result']
    token = results['token']
    db_manager = TextDBCachingManager()
    values_dict = {
        'fingerprint': fp,
        'source': text,
        'source_lang': src,
        'target_lang': tgt,
        'date_added': get_current_datetime()
    }
    db_manager.insert_or_update_details(token, values_dict)

    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='caching_6.translation_text_lookup_using_fingerprint', ignore_result=False)
def cache_lookup_translation_text_using_fingerprint_task(self, token, fp, src, tgt, return_list=False):
    db_manager = TextDBCachingManager()
    # Super quick fingerprint lookup
    closest_text = db_manager.get_all_details(['target'], allow_nulls=False,
                                              equality_conditions={'fingerprint': fp,
                                                                   'source_lang': src,
                                                                   'target_lang': tgt})
    if closest_text is not None:
        all_keys = list(closest_text.keys())
        translation = closest_text[all_keys[0]]['target']
        db_manager.insert_or_update_details(token, {
            'target': translation,
        })
        return {
            'result': translation_text_back_to_list(translation, return_list=return_list),
            'text_too_large': False,
            'successful': True,
            'fresh': False,
            'device': None
        }
    return None


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='caching_6.cache_lookup_translate_text', translation_obj=translation_models, ignore_result=False)
def cache_lookup_translate_text_task(self, token, return_list=False):
    db_manager = TextDBCachingManager()
    existing = db_manager.get_details(token, ['target'], using_most_similar=False)[0]
    if existing is not None and existing['target'] is not None:
        print('Returning cached result')
        return {
            'result': translation_text_back_to_list(existing['target'], return_list=return_list),
            'text_too_large': False,
            'successful': True,
            'fresh': False,
            'device': None
        }
    return None


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.translate_text', translation_obj=translation_models, ignore_result=False)
def translate_text_task(self, text, src, tgt):
    if src == tgt:
        return {
            'result': "'source' and 'target' languages must be different!",
            'text_too_large': False,
            'successful': False,
            'fresh': False,
            'device': None
        }

    how = f"{src}-{tgt}"
    try:
        translated_text, large_warning, all_large_warnings = self.translation_obj.translate(text, how=how)
        if translated_text is not None and not large_warning:
            success = True
        else:
            success = False
        if large_warning:
            large_warning_indices = [str(i) for i in range(len(all_large_warnings)) if all_large_warnings[i]]
            translated_text = (
                    LONG_TEXT_ERROR
                    + f"This happened for inputs at indices {','.join(large_warning_indices)}."
            )
    except NotImplementedError as e:
        print(e)
        translated_text = str(e)
        success = False
        large_warning = False

    return {
        'result': translated_text,
        'text_too_large': large_warning,
        'successful': success,
        'fresh': success,
        'device': self.translation_obj.get_device()
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.translate_text_callback', translation_obj=translation_models, ignore_result=False)
def translate_text_callback_task(self, results, token, text, src, tgt, force=False, return_list=False):
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
            current_datetime = get_current_datetime()
            values_dict['date_added'] = current_datetime
        # Inserting values for original token
        db_manager.insert_or_update_details(
            token, values_dict
        )
        if not force:
            # Inserting the same values for closest token if different than original token
            # Only happens if the other token has been fingerprinted first without being translated.
            closest = db_manager.get_closest_match(token)
            if closest is not None and closest != token:
                db_manager.insert_or_update_details(
                    closest, values_dict
                )
    elif not results['successful']:
        # in case we fingerprinted something and then failed to translate it, we delete its cache row
        db_manager.delete_cache_rows([token])

    # If the computation was successful and return_list is True, we want to convert the text results
    # back to a list (because this flag means that the original input was a list of strings)
    if results['successful']:
        results['result'] = translation_text_back_to_list(results['result'], return_list=return_list)
    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.detect_text_language', ignore_result=False)
def detect_text_language_task(self, text):
    result = detect_text_language(text)
    return {
        'language': result,
        'successful': result is not None
    }
