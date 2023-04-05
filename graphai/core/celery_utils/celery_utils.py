from celery import current_app as current_celery_app
from celery.result import AsyncResult

from graphai.core.interfaces.celery_config import get_settings


def create_celery():
    celery_app = current_celery_app
    settings = get_settings()
    celery_app.config_from_object(settings, namespace='CELERY')
    celery_app.conf.update(task_track_started=True)
    celery_app.conf.update(task_serializer='pickle')
    celery_app.conf.update(result_serializer='pickle')
    celery_app.conf.update(accept_content=['pickle', 'json'])
    celery_app.conf.update(result_expires=200)
    celery_app.conf.update(result_persistent=True)
    celery_app.conf.update(result_extended=True)
    celery_app.conf.update(worker_send_task_events=False)
    celery_app.conf.update(worker_prefetch_multiplier=1)

    return celery_app


def get_celery_task(task_id):
    return AsyncResult(task_id)


def compile_task_results(task_id, task_results=None):
    """
    return task info for the given task_id
    """
    task = get_celery_task(task_id)
    if task_results is None:
        task_results = task.result
    return {'id': task_id, 'name': task.name, 'status': task.status, 'results': task_results}