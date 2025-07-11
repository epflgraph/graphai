from celery import chain
from graphai.celery.retrieval.tasks import (
    retrieve_from_es_task,
    chunk_text_task,
    anonymize_text_task
)
from graphai.celery.embedding.tasks import embed_text_task
from graphai.core.retrieval.retrieval_settings import RETRIEVAL_PARAMS
from graphai.celery.common.jobs import DEFAULT_TIMEOUT


def retrieve_from_es_job(text, index_to_search_in,
                         filters=None, limit=10, return_scores=False, filter_by_date=False):
    task_list = [
        embed_text_task.s(text, RETRIEVAL_PARAMS.get(index_to_search_in, dict()).get('model', None)),
        retrieve_from_es_task.s(text, index_to_search_in, filters, limit, return_scores, filter_by_date)
    ]
    task = chain(task_list)
    results = task.apply_async(priority=6).get(timeout=DEFAULT_TIMEOUT)
    return results


def chunk_text_job(text, chunk_size, chunk_overlap, one_chunk_per_page=False, one_chunk_per_doc=False):
    task = chunk_text_task.s(text, chunk_size, chunk_overlap, one_chunk_per_page, one_chunk_per_doc)
    results = task.apply_async(priority=10).get(timeout=DEFAULT_TIMEOUT)
    return results


def anonymize_text_job(text, lang):
    task = anonymize_text_task.s(text, lang)
    results = task.apply_async(priority=10).get(timeout=DEFAULT_TIMEOUT)
    return results
