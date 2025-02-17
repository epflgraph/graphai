from celery import chain
from graphai.celery.retrieval.tasks import retrieve_from_es_task
from graphai.celery.embedding.tasks import embed_text_task
from graphai.core.retrieval.retrieval_settings import RETRIEVAL_PARAMS


def retrieve_lex_job(text, index_to_search_in, lang_filter=None, limit=10):
    task_list = [
        embed_text_task.s(text, RETRIEVAL_PARAMS[index_to_search_in]['model']),
        retrieve_from_es_task.s(text, index_to_search_in, lang_filter, limit)
    ]
    task = chain(task_list)
    task = task.apply_async(priority=6)
    return task.id
