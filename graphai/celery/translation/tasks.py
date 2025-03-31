from celery import shared_task
from graphai.core.translation.text_utils import (
    HUGGINGFACE_UNLOAD_WAITING_PERIOD,
    TranslationModels
)
from graphai.core.common.fingerprinting import compute_text_fingerprint
from graphai.core.common.common_utils import (
    strtobool
)
from graphai.core.common.caching import (
    TextDBCachingManager
)
from graphai.core.common.lookup import fingerprint_cache_lookup
from graphai.core.common.config import config
from graphai.core.translation.translation import (
    translate_text,
    translate_text_callback,
    translate_text_return_list_callback,
    compute_translation_text_fingerprint_callback,
    cache_lookup_translation_text_using_fingerprint,
    cache_lookup_translate_text,
    detect_language_translation
)

translation_models = TranslationModels()


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.init_translation', ignore_result=False,
             translation_obj=translation_models)
def translation_init_task(self):
    # This task initialises the video celery worker by loading into memory the transcription and NLP models
    print('Start init_translation task')

    if strtobool(config['preload'].get('video', 'no')):
        print('Loading translation models...')
        self.translation_obj.load_models()
    else:
        print('Skipping preloading for video endpoints.')

    print('Initializing db caching managers...')
    TextDBCachingManager(initialize_database=True)

    return True


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='caching_6.cache_lookup_fingerprint_translation_text', ignore_result=False)
def cache_lookup_translation_text_fingerprint_task(self, token):
    return fingerprint_cache_lookup(token, TextDBCachingManager())


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.fingerprint_translation_text', ignore_result=False)
def compute_translation_text_fingerprint_task(self, token, text):
    return compute_text_fingerprint(token, text)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.fingerprint_translation_text_callback', ignore_result=False)
def compute_translation_text_fingerprint_callback_task(self, results, text, src, tgt):
    return compute_translation_text_fingerprint_callback(results, text, src, tgt)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='caching_6.translation_text_lookup_using_fingerprint', ignore_result=False)
def cache_lookup_translation_text_using_fingerprint_task(self, token, fp, src, tgt, return_list=False):
    return cache_lookup_translation_text_using_fingerprint(token, fp, src, tgt, return_list)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='caching_6.cache_lookup_translate_text', ignore_result=False)
def cache_lookup_translate_text_task(self, token, return_list=False):
    return cache_lookup_translate_text(token, return_list)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.translate_text', translation_obj=translation_models, ignore_result=False)
def translate_text_task(self, text, src, tgt, skip_sentence_segmentation=False):
    return translate_text(text, src, tgt, self.translation_obj, skip_sentence_segmentation)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.translate_text_callback', ignore_result=False)
def translate_text_callback_task(self, results, token, text, src, tgt, force=False):
    return translate_text_callback(results, token, text, src, tgt, force)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.translate_text_return_list_callback', ignore_result=False)
def translate_text_return_list_callback_task(self, results, return_list=False):
    return translate_text_return_list_callback(results, return_list)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.detect_text_language', ignore_result=False)
def detect_text_language_task(self, text):
    return detect_language_translation(text)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.clean_up_translation_object', model=translation_models, ignore_result=False)
def cleanup_translation_object_task(self):
    return self.model.unload_model(HUGGINGFACE_UNLOAD_WAITING_PERIOD)
