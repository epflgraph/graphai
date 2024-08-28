from graphai.celery.common.celery_config import create_celery

celery_instance = create_celery()


def get_n_celery_workers():
    celery_stats = celery_instance.control.inspect().stats()
    celery_host = list(celery_stats)[0]
    print('#' * 60, 'get_n_celery_workers')
    print(celery_stats[celery_host])
    return celery_stats[celery_host]['pool']['max-concurrency']
