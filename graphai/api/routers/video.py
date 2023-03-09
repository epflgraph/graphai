import time

from fastapi import APIRouter

import ray

from graphai.api.schemas.video import *

from graphai.api.celery_tasks.video import multiproc_example_task, celery_multiproc_example_task


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


class VideoActorList:

    def __init__(self, n):
        self.n = n
        self.actors = []

    def instantiate_actors(self):
        self.actors = [VideoActor.remote() for i in range(self.n)]

    def free_actors(self):
        self.actors = []

    def get_actor(self, i):
        return self.actors[i % self.n]


# Instantiate ray actor list
video_actor_list = VideoActorList(16)


# @router.post('/multiprocessing_example', response_model=MultiprocessingExampleResponse)
# async def multiprocessing_example(data: MultiprocessingExampleRequest):
#     result = multiproc_example_task.apply_async(args=[data, video_actor_list]).get()
#     return result


@router.post('/multiprocessing_example_purecelery', response_model=MultiprocessingExampleResponse)
async def multiprocessing_example(data: MultiprocessingExampleRequest):
    result = celery_multiproc_example_task(data)
    return result

