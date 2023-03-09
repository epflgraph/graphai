from celery import shared_task, chord, group
import ray
import time
from graphai.api.common.log import log
from graphai.core.utils.time.stopwatch import Stopwatch


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 1},
             name='video.some_dummy', ignore_result=False)
def do_something_in_parallel(self, x):
    time.sleep(x)
    return True


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 1},
             name='video.some_post_dummy', ignore_result=False)
def do_some_callback(self, l):
    return all(l)


def celery_multiproc_example_task(data):
    sw = Stopwatch()
    foo = data.foo
    bar = data.bar
    log(f'Got input parameters ({foo}, {bar})', sw.delta())
    t = (group(do_something_in_parallel.signature(args=[foo], priority=4) for i in range(bar)) |
                        do_some_callback.signature(args=[], priority=4)).apply_async().get()
    log(f'Got all results', sw.delta())
    return {'baz': t}