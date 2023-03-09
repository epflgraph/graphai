from celery import shared_task, chord, group
import time
from graphai.api.common.log import log
from graphai.core.utils.time.stopwatch import Stopwatch


# A task that will have several instances run in parallel
@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 1},
             name='video.example_parallel', ignore_result=False)
def example_parallel_task(self, x):
    time.sleep(x)
    return True


# A task that acts as a callback after a parallel operation
@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 1},
             name='video.example_callback', ignore_result=False)
def example_callback_task(self, l):
    return all(l)


# The function that creates and calls the celery task
def celery_multiproc_example_task(data):
    sw = Stopwatch()
    # Getting the parameters
    foo = data.foo
    bar = data.bar
    log(f'Got input parameters ({foo}, {bar})', sw.delta())
    # Here we run 'bar' instances of the parallel task in a group (which means in parallel), and
    # the results are then collected and fed into the callback task.
    # apply_async() schedules the task and get() blocks on it until completion and returns the results.
    t = (group(example_parallel_task.signature(args=[foo]) for i in range(bar)) |
         example_callback_task.signature(args=[])).apply_async(priority=2).get()
    log(f'Got all results', sw.delta())
    return {'baz': t}