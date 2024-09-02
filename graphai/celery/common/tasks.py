from celery import shared_task


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.text_dummy_task', ignore_result=False)
def text_dummy_task(self, results):
    # This task is required for chaining groups due to the peculiarities of celery
    # Whenever there are two groups in one chain of tasks, there need to be at least
    # TWO tasks between them, and this dummy task is simply an f(x)=x function.
    return results


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='video_2.video_dummy_task', ignore_result=False)
def video_dummy_task(self, results):
    # This task is required for chaining groups due to the peculiarities of celery
    # Whenever there are two groups in one chain of tasks, there need to be at least
    # TWO tasks between them, and this dummy task is simply an f(x)=x function.
    return results
