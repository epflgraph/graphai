from celery import shared_task
from graphai.core.retrieval.retrieval_utils import search_lex


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='retrieval_10.retrieve_lex', ignore_result=False)
def retrieve_lex_task(self, embedding_results, text, lang_filter, limit=10):
    return search_lex(text,
                      embedding_results['result'] if embedding_results['successful'] else None,
                      lang_filter,
                      limit)
