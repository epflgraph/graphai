from celery import shared_task
from graphai.core.retrieval.retrieval_utils import (
    retrieve_from_es,
    chunk_text
)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='retrieval_10.retrieve', ignore_result=False)
def retrieve_from_es_task(self, embedding_results, text, index_to_search_in, filters=None, limit=10):
    return retrieve_from_es(embedding_results, text, index_to_search_in, filters, limit)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='retrieval_10.chunk', ignore_result=False)
def chunk_text_task(self, text, chunk_size=400, chunk_overlap=100):
    return chunk_text(text, chunk_size, chunk_overlap)
