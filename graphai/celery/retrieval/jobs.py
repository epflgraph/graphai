from celery import chain
from graphai.celery.retrieval.tasks import (
    retrieve_from_es_task,
    chunk_text_task
)
from graphai.celery.embedding.tasks import embed_text_task
from graphai.core.retrieval.retrieval_settings import RETRIEVAL_PARAMS
from graphai.celery.common.jobs import DEFAULT_TIMEOUT


def retrieve_lex_job(text, index_to_search_in, lang_filter=None, limit=10):
    task_list = [
        embed_text_task.s(text, RETRIEVAL_PARAMS[index_to_search_in]['model']),
        retrieve_from_es_task.s(text, index_to_search_in, lang_filter, limit)
    ]
    task = chain(task_list)
    task = task.apply_async(priority=6)
    return task.id


def chunk_text_job(text, chunk_size, chunk_overlap):
    task = chunk_text_task.s(text, chunk_size, chunk_overlap)
    results = task.apply_async(priority=10).get(timeout=DEFAULT_TIMEOUT)
    return results
