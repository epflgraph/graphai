from celery import shared_task
from graphai.core.retrieval.retrieval_utils import (
    search_lex,
    search_servicedesk,
    chunk_text
)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='retrieval_10.retrieve', ignore_result=False)
def retrieve_from_es_task(self, embedding_results, text, index_to_search_in, filters=None, limit=10):
    if filters is None:
        filters = dict()
    if index_to_search_in == 'lex':
        return search_lex(text,
                          embedding_results['result'] if embedding_results['successful'] else None,
                          filters.get('lang', None),
                          limit)
    elif index_to_search_in == 'servicedesk':
        return search_servicedesk(text,
                                  embedding_results['result'] if embedding_results['successful'] else None,
                                  filters.get('lang', None),
                                  filters.get('category', None),
                                  limit)
    else:
        return {
            'n_results': 0,
            'result': f'Index "{index_to_search_in}" does not exist.',
            'successful': False
        }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='retrieval_10.chunk', ignore_result=False)
def chunk_text_task(self, text, chunk_size=400, chunk_overlap=100):
    return chunk_text(text, chunk_size, chunk_overlap)
