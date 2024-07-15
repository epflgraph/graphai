import os
from functools import lru_cache

from celery import current_app as current_celery_app
from celery.result import AsyncResult
from kombu import Queue

from graphai.core.interfaces.config import config

DEFAULT_BROKER = "amqp://guest:guest@localhost:5672//"
DEFAULT_BACKEND = "redis://localhost:6379/0"


def route_task(name, args, kwargs, options, task=None, **kw):
    # Naming convention: name of a task follows the `queue.taskname` format. `taskname` may have further dots.
    if '.' in name:
        queue = name.split('.')[0]
        return {'queue': queue}
    return {'queue': 'celery'}


class BaseConfig:

    def __init__(self):
        try:
            print("Reading celery configuration from config")
            self.broker_url = config['celery'].get('broker_url', DEFAULT_BROKER)
            self.result_backend = config['celery'].get('result_backend', DEFAULT_BACKEND)
        except Exception:
            print(
                "The celery configuration could not be found in the config file, using default parameters. "
                "To use different ones, make sure to add a [celery] section with the corresponding parameters."
            )
            self.broker_url = DEFAULT_BROKER
            self.result_backend = DEFAULT_BACKEND

        self.CELERY_WORKER_REDIRECT_STDOUTS: bool = False
        self.CELERY_TASK_QUEUES: list = [
            # default queue
            Queue("celery"),
            # custom queues
            # Concept detection
            Queue("text_10", max_priority=10),
            # Cache lookups
            Queue("caching_6", max_priority=6),
            # Translation
            Queue("text_6", max_priority=6),
            # Video, voice, image
            Queue("video_2", max_priority=2),
            # Ontology
            Queue("ontology_6", max_priority=6),
            # Scraping
            Queue("scraping_6", max_priority=6)
        ]

        self.CELERY_TASK_ROUTES = (route_task,)


class DevelopmentConfig(BaseConfig):
    pass


@lru_cache()
def get_settings():
    config_cls_dict = {
        "development": DevelopmentConfig,
    }
    config_name = os.environ.get("CELERY_CONFIG", "development")
    config_cls = config_cls_dict[config_name]
    return config_cls()


def create_celery():
    """
    Creates a celery app with default settings
    Returns:
        Celery app object
    """
    os.environ["FORKED_BY_MULTIPROCESSING"] = "1"
    if os.name != "nt":
        from billiard import context
        context._force_start_method("spawn")
    celery_app = current_celery_app
    settings = get_settings()
    celery_app.config_from_object(settings, namespace='CELERY')
    celery_app.conf.update(task_track_started=True)
    # Setting serializers to pickle makes them more flexible and faster (when running a local instance of celery)
    celery_app.conf.update(task_serializer='pickle')
    celery_app.conf.update(result_serializer='pickle')
    celery_app.conf.update(accept_content=['pickle', 'json'])
    celery_app.conf.update(result_expires=10800)
    celery_app.conf.update(result_persistent=True)
    celery_app.conf.update(result_extended=True)
    celery_app.conf.update(worker_send_task_events=False)
    celery_app.conf.update(worker_prefetch_multiplier=1)
    celery_app.conf.update(broker_transport_options={'visibility_timeout': 9999999})
    celery_app.conf.update(beat_schedule={
        'cleanup-embedding-model-every-six-hours': {
            'task': 'text_6.clean_up_large_embedding_objects',
            'schedule': 6 * 3600.0
            # Every 6 hours
        },
        'cleanup-whisper-model-every-twentyfour-hours': {
            'task': 'video_2.clean_up_transcription_object',
            'schedule': 24 * 3600.0
            # Every 24 hours
        }
    })

    return celery_app


def get_celery_task(task_id):
    """
    Returns results for the task with the provided task id
    Args:
        task_id: task id

    Returns:
        AsyncResult object that contains task id, name, status, and results
    """
    return AsyncResult(task_id)


def get_task_info(task_id, task_results=None):
    """
    return task info for the given task_id
    """
    task = get_celery_task(task_id)
    if task_results is None:
        task_results = task.result
    return {'id': task_id, 'name': task.name, 'status': task.status, 'results': task_results}
