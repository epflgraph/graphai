from celery import shared_task
from graphai.api.common.video import translation_models
from graphai.core.common.video import detect_text_language

@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.translate_text', translation_obj=translation_models, ignore_result=False)
def translate_text_task(self, text, how='fr-en'):
    return self.translation_obj.translate(text, how=how)

@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.detect_text_language', ignore_result=False)
def detect_text_language_task(self, text):
    return detect_text_language(text)