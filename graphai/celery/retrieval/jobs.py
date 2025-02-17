from celery import chain
from graphai.celery.retrieval.tasks import retrieve_lex_task
from graphai.celery.embedding.tasks import embed_text_task
from graphai.core.retrieval.retrieval_settings import RETRIEVAL_PARAMS


def retrieve_lex_job(text, lang_filter=None, limit=10):
    task_list = [
        embed_text_task.s(text, RETRIEVAL_PARAMS['lex']['model']),
        retrieve_lex_task.s(text, lang_filter, limit)
    ]
    task = chain(task_list)
    task = task.apply_async(priority=6)
    return task.id
