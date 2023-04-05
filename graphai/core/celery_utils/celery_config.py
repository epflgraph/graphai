import os
from functools import lru_cache
from kombu import Queue
import configparser
from graphai.definitions import CONFIG_DIR

DEFAULT_BROKER = "amqp://guest:guest@localhost:5672//"
DEFAULT_CACHE = "redis://localhost:6379/0"

def route_task(name, args, kwargs, options, task=None, **kw):
    if "." in name:
        queue, _ = name.split(".")
        return {"queue": queue}
    return {"queue": "celery"}


class BaseConfig:

    def __init__(self):
        db_config = configparser.ConfigParser()
        try:
            print('Reading celery configuration from file')
            db_config.read(f'{CONFIG_DIR}/celery.ini')
            self.broker_url = db_config['CELERY'].get('broker_url', fallback=DEFAULT_BROKER)
            self.result_backend = db_config['CELERY'].get('result_backend', fallback=DEFAULT_CACHE)
        except Exception as e:
            print(f'Could not read file {CONFIG_DIR}/celery.ini or '
                  f'file does not have section [CELERY], falling back to defaults.')
            self.broker_url = DEFAULT_BROKER
            self.result_backend = DEFAULT_CACHE

        self.CELERY_WORKER_REDIRECT_STDOUTS: bool = False
        self.CELERY_TASK_QUEUES: list = [
            # default queue
            Queue("celery"),
            # custom queue
            Queue("text", max_priority=10),
            Queue("video", max_priority=2),
            Queue("ontology", max_priority=6)
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


settings = get_settings()
