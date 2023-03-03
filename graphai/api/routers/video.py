from fastapi import APIRouter
import ray
import time

from graphai.api.schemas.video import *

from graphai.api.common import log

from graphai.core.utils.time.stopwatch import Stopwatch


# Initialise video router
router = APIRouter(
    prefix='/video',
    tags=['video'],
    responses={404: {'description': 'Not found'}}
)


# Define ray actor
@ray.remote
class VideoActor:
    """
    Class representing a ray Actor to perform video tasks in parallel.
    """

    def do_something(self, x):
        # Perform some time-consuming task
        time.sleep(x)

        return True


# Instantiate ray actor list
n_actors = 16
actors = [VideoActor.remote() for i in range(n_actors)]


@router.post('/multiprocessing_example', response_model=MultiprocessingExampleResponse)
async def multiprocessing_example(data: MultiprocessingExampleRequest):

    # Initialize stopwatch to track time
    sw = Stopwatch()

    # Get input parameters
    foo = data.foo
    bar = data.bar
    log(f'Got input parameters ({foo}, {bar})', sw.delta())

    # Execute tasks in parallel
    results = [actors[i % n_actors].do_something.remote(foo) for i in range(bar)]
    log(f'Dispatched all tasks in parallel to the actors', sw.delta())

    # Wait for the results
    results = ray.get(results)
    log(f'Got all results', sw.delta())

    # Combine the results
    baz = all(results)
    log(f'Finished all tasks', sw.total(), total=True)

    return {'baz': baz}
