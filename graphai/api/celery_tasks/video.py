from celery import shared_task, chain, group, chord
import ray
import time
from graphai.api.common.log import log
from graphai.core.utils.time.stopwatch import Stopwatch


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 1},
             name='video.multiprocessing_example', ignore_result=False)
def multiproc_example_task(self, data, video_actor_list):
    # Initialize stopwatch to track time
    sw = Stopwatch()

    # Get input parameters
    foo = data.foo
    bar = data.bar
    log(f'Got input parameters ({foo}, {bar})', sw.delta())

    # Execute tasks in parallel
    results = [video_actor_list.get_actor(i).do_something.remote(foo) for i in range(bar)]
    log(f'Dispatched all tasks in parallel to the actors', sw.delta())

    # Wait for the results
    results = ray.get(results)
    log(f'Got all results', sw.delta())

    # Combine the results
    baz = all(results)
    log(f'Finished all tasks', sw.total(), total=True)

    return {'baz': baz}


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 1},
             name='video.some_dummy', ignore_result=False)
def do_something_celery(self, x):
    time.sleep(x)
    return True


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 1},
             name='video.some_post_dummy', ignore_result=False)
def do_some_callback_celery(self, l):
    return all(l)


def celery_multiproc_example_task(data):
    sw = Stopwatch()
    foo = data.foo
    bar = data.bar
    log(f'Got input parameters ({foo}, {bar})', sw.delta())
    t = chord(do_something_celery.s(foo) for i in range(bar))(do_some_callback_celery.s()).get()
    log(f'Got all results', sw.delta())
    return {'baz': t}