from celery import chain, group

from graphai.api.celery_tasks.video import (
    cache_lookup_retrieve_file_from_url_task,
    retrieve_file_from_url_task,
    retrieve_file_from_url_callback_task
)


def retrieve_url_job(url, force=False, is_playlist=False):
    if not force:
        direct_lookup_job = cache_lookup_retrieve_file_from_url_task.s(url)
        direct_lookup_job = direct_lookup_job.apply_async(priority=6)
        direct_lookup_task_id = direct_lookup_job.id
        # We block on this task since we need its results to decide what to do next
        direct_lookup_results = direct_lookup_job.get(timeout=20)
        # If the cache lookup yielded results, then return the id of the task, otherwise we proceed normally with the
        # computations
        if direct_lookup_results is not None:
            return direct_lookup_task_id

    # Overriding the is_playlist flag if the url ends with m3u8 (playlist) or mp4/mkv/flv/avi/mov (video file)
    if url.endswith('.m3u8'):
        is_playlist = True
    elif any([url.endswith(e) for e in ['.mp4', '.mkv', '.flv', '.avi', '.mov']]):
        is_playlist = False
    # First retrieve the file, and then do the database callback
    task_list = [retrieve_file_from_url_task.s(url, is_playlist, None),
                 retrieve_file_from_url_callback_task.s(url, force)]
    task = chain(task_list)
    task = task.apply_async(priority=2)
    return task.id
