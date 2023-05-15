from celery import shared_task
from graphai.api.common.video import translation_models
from graphai.core.common.video import detect_text_language

LONG_TEXT_ERROR = "Unpunctuated text too long (over 512 tokens), " \
                  "try adding punctuation or providing a smaller chunk of text."

@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.translate_text', translation_obj=translation_models, ignore_result=False)
def translate_text_task(self, text, src, tgt):
    if src == tgt:
        return {
            'result': "'source' and 'target' languages must be different!",
            'text_too_large': False,
            'successful': False
        }
    how = f"{src}-{tgt}"
    translated_text, large_warning = self.translation_obj.translate(text, how=how)
    if translated_text is not None:
        success = True
    else:
        success = False
        if large_warning:
            translated_text = LONG_TEXT_ERROR
    return {
        'result': translated_text,
        'text_too_large': large_warning,
        'successful': success
    }

@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.detect_text_language', ignore_result=False)
def detect_text_language_task(self, text):
    result = detect_text_language(text)
    return {
        'language': result,
        'successful': result is not None
    }