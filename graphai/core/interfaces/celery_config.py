import os
import configparser
from functools import lru_cache

from celery import current_app as current_celery_app
from celery.result import AsyncResult
from kombu import Queue

from graphai.definitions import CONFIG_DIR

DEFAULT_BROKER = "amqp://guest:guest@localhost:5672//"
DEFAULT_CACHE = "redis://localhost:6379/0"


def route_task(name, args, kwargs, options, task=None, **kw):
    # Naming convention: name of a task follows the `queue.taskname` format. `taskname` may have further dots.
    if '.' in name:
        queue = name.split('.')[0]
        return {'queue': queue}
    return {'queue': 'celery'}


class BaseConfig:

    def __init__(self):
        config_contents = configparser.ConfigParser()
        try:
            print('Reading celery configuration from file')
            config_contents.read(f'{CONFIG_DIR}/celery.ini')
            self.broker_url = config_contents['CELERY'].get('broker_url', fallback=DEFAULT_BROKER)
            self.result_backend = config_contents['CELERY'].get('result_backend', fallback=DEFAULT_CACHE)
        except Exception:
            print(f'Could not read file {CONFIG_DIR}/celery.ini or '
                  f'file does not have section [CELERY], falling back to defaults.')
            self.broker_url = DEFAULT_BROKER
            self.result_backend = DEFAULT_CACHE

        self.CELERY_WORKER_REDIRECT_STDOUTS: bool = False
        self.CELERY_TASK_QUEUES: list = [
            # default queue
            Queue("celery"),
            # custom queues
            # Concept detection
            Queue("text_10", max_priority=10),
            # Translation
            Queue("text_6", max_priority=6),
            # Video, voice, image
            Queue("video_2", max_priority=2),
            # Ontology
            Queue("ontology_6", max_priority=6)
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
    celery_app = current_celery_app
    settings = get_settings()
    celery_app.config_from_object(settings, namespace='CELERY')
    celery_app.conf.update(task_track_started=True)
    # Setting serializers to pickle makes them more flexible and faster (when running a local instance of celery)
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


def get_task_info(task_id, task_results=None):
    """
    return task info for the given task_id
    """
    task = get_celery_task(task_id)
    if task_results is None:
        task_results = task.result
    return {'id': task_id, 'name': task.name, 'status': task.status, 'results': task_results}
