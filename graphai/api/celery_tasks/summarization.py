from celery import shared_task
from graphai.core.common.video import ChatGPTSummarizer, perceptual_hash_text, get_current_datetime
from graphai.core.common.caching import SummaryDBCachingManager

@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.summarize_text_chatgpt', ignore_result=False)
def summarize_text_task(self, token_and_text, text_type='lecture', title=False, force=False):
    token = token_and_text['token']
    text = token_and_text['text']
    is_keywords = token_and_text['keyword']
    # TODO add cache check
    summarizer = ChatGPTSummarizer()
    results, too_many_tokens = summarizer.generate_summary(
        text, text_type=text_type, keywords=is_keywords, ordered=not is_keywords,
        title=title, max_len=20 if title else 200)
    return {
        'result': results,
        'fresh': results is not None,
        'successful': results is not None,
        'too_many_tokens': too_many_tokens
    }



