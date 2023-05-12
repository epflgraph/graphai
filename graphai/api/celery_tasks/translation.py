from celery import shared_task
from graphai.api.common.video import translation_models
from graphai.core.common.video import detect_text_language

@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.translate_text', translation_obj=translation_models, ignore_result=False)
def translate_text_task(self, text, src, tgt):
    if src == tgt:
        return {
            'result': "'source' and 'target' languages must be different!",
            'successful': False
        }
    how = f"{src}-{tgt}"
    try:
        translated_text = self.translation_obj.translate(text, how=how)
        if translated_text is None:
            success = False
        else:
            success = True
    except IndexError as e:
        translated_text = "Unpunctuated text too long (over 512 tokens), " \
                          "try adding punctuation or providing a smaller chunk of text."
        success = False
    return {
        'result': translated_text,
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