import os
from functools import lru_cache
from kombu import Queue


def route_task(name, args, kwargs, options, task=None, **kw):
    if "." in name:
        queue, _ = name.split(".")
        return {"queue": queue}
    return {"queue": "celery"}


class BaseConfig:
    broker_url: str = os.environ.get("CELERY_BROKER_URL", "amqp://guest:guest@localhost:5672//")
    result_backend: str = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
    CELERY_WORKER_REDIRECT_STDOUTS: bool = False

    CELERY_TASK_QUEUES: list = (
        # default queue
        Queue("celery"),
        # custom queue
        Queue("text", max_priority=10),
        Queue("video", max_priority=2),
        Queue("ontology", max_priority=6)
    )

    CELERY_TASK_ROUTES = (route_task,)


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


settings = get_settings()
