import os
from functools import lru_cache

from celery import current_app as current_celery_app
from celery.result import AsyncResult
from kombu import Queue

from graphai.core.common.config import config

DEFAULT_BROKER = "amqp://guest:guest@localhost:5672//"
DEFAULT_BACKEND = "redis://localhost:6379/0"

# Queue priorities (1 = lowest, 9 = highest)
queue_priorities = {
    "celery": 1,
    "caching": 9,
    "video": 5,
    "image": 7,
    "voice": 5,
    "voice_gpu": 5,
    "translation": 5,
    "translation_gpu": 5,
    "embedding": 8,
    "embedding_gpu": 8,
    "ontology": 4,
    "text": 6,
    "scraping": 3,
    "rag": 8
}


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
            
            #----------------------#
            # Default Celery Queue #
            #----------------------#
            Queue("celery", max_priority=queue_priorities["celery"]), # ⚪ Lowest priority fallback queue

            #---------------#
            # Caching Queue #
            #---------------#
            Queue("caching", max_priority=queue_priorities["caching"]), # 🔴 Highest priority (cache hits)

            #-----------------#
            # Endpoint Queues #
            #-----------------#
            Queue("video"          , max_priority=queue_priorities["video"]),           # /video ................. 🔵 Graph pipeline work
            Queue("image"          , max_priority=queue_priorities["image"]),           # /image ................. 🔴 High priority (for RAG pipeline)
            Queue("voice"          , max_priority=queue_priorities["voice"]),           # /voice ................. 🔵 Graph pipeline work
            Queue("voice_gpu"      , max_priority=queue_priorities["voice_gpu"]),       # /voice (GPU) ........... 🔵 Graph pipeline work (slow tasks)
            Queue("translation"    , max_priority=queue_priorities["translation"]),     # /translation ........... 🔵 Graph pipeline work
            Queue("translation_gpu", max_priority=queue_priorities["translation_gpu"]), # /translation (GPU) ..... 🔵 Graph pipeline work (slow tasks)
            Queue("embedding"      , max_priority=queue_priorities["embedding"]),       # /embedding ............. 🔴 Misson critical (for Chatbot)
            Queue("embedding_gpu"  , max_priority=queue_priorities["embedding_gpu"]),   # /embedding (GPU) ....... 🔴 Misson critical (for Chatbot)
            Queue("ontology"       , max_priority=queue_priorities["ontology"]),        # /ontology .............. 🟢 Used sparingly
            Queue("text"           , max_priority=queue_priorities["text"]),            # /text .................. 🟡 Concept detection (lots of tasks) 
            Queue("scraping"       , max_priority=queue_priorities["scraping"]),        # /scraping .............. 🟢 Used sparingly
            Queue("rag"            , max_priority=queue_priorities["rag"])              # /rag ................... 🔴 Misson critical (for Chatbot)
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
    celery_app.conf.update(worker_send_task_events=True)
    celery_app.conf.update(task_send_sent_event=True)
    celery_app.conf.update(worker_prefetch_multiplier=1)
    celery_app.conf.update(broker_transport_options={'visibility_timeout': 9999999})
    celery_app.conf.update(beat_schedule={
        'cleanup-embedding-model-every-six-hours': {
            'task': 'embedding_gpu.clean_up_large_embedding_objects',
            'schedule': 6 * 3600.0
            # Every 6 hours
        },
        'cleanup-translation-model-every-six-hours': {
            'task': 'translation_gpu.clean_up_translation_object',
            'schedule': 6 * 3600.0
            # Every 6 hours
        },
        'cleanup-whisper-model-every-twentyfour-hours': {
            'task': 'voice_gpu.clean_up_transcription_object',
            'schedule': 24 * 3600.0
            # Every 24 hours
        }
    })
    # Configuring the broker to avoid accidentally-missed heartbeats
    celery_app.conf.update(task_acks_late=True)
    celery_app.conf.update(worker_send_task_events=True)
    celery_app.conf.update(send_events=True)
    celery_app.conf.update(send_sent_event=True)
    celery_app.conf.update(task_track_started=True)
    celery_app.conf.update(redis_socket_keepalive=True)
    celery_app.conf.update(broker_pool_limit=None)
    celery_app.conf.update(broker_connection_timeout=300000)
    celery_app.conf.update(broker_max_retries=None)
    celery_app.conf.update(worker_lost_wait=300000)
    celery_app.conf.update(worker_cancel_long_running_tasks_on_connection_loss=True)
    celery_app.conf.update(broker_connection_retry_on_startup=True)
    celery_app.conf.update(broker_channel_error_retry=True)

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
