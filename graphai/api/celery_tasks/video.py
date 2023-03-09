from celery import shared_task
import ray
import time
from graphai.api.common.log import log
from graphai.core.utils.time.stopwatch import Stopwatch

@ray.remote
def do_something(x):
    # Perform some time-consuming task
    time.sleep(x)
    return True

@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 1},
             name='video.multiprocessing_example', ignore_result=False)
def multiproc_example_task(self, data):
    # Initialize stopwatch to track time
    sw = Stopwatch()

    # Get input parameters
    foo = data.foo
    bar = data.bar
    log(f'Got input parameters ({foo}, {bar})', sw.delta())

    # Execute tasks in parallel
    results = [do_something.remote(foo) for i in range(bar)]
    log(f'Dispatched all tasks in parallel to the actors', sw.delta())

    # Wait for the results
    results = ray.get(results)
    log(f'Got all results', sw.delta())

    # Combine the results
    baz = all(results)
    log(f'Finished all tasks', sw.total(), total=True)

    return {'baz': baz}