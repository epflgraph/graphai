from celery import chain, group

from graphai.api.celery_tasks.common import ignore_fingerprint_results_callback_task
from graphai.api.celery_tasks.video import (
    cache_lookup_retrieve_file_from_url_task,
    retrieve_file_from_url_task,
    retrieve_file_from_url_callback_task,
    cache_lookup_fingerprint_video_task,
    compute_video_fingerprint_task,
    compute_video_fingerprint_callback_task,
    video_fingerprint_find_closest_retrieve_from_db_task,
    video_fingerprint_find_closest_parallel_task,
    video_fingerprint_find_closest_callback_task,
    retrieve_video_fingerprint_callback_task
)
from graphai.core.common.caching import FingerprintParameters


def get_video_fingerprint_chain_list(token, min_similarity=None, n_jobs=8,
                                     ignore_fp_results=False, results_to_return=None):
    # Retrieve minimum similarity parameter for video fingerprints
    if min_similarity is None:
        fp_parameters = FingerprintParameters()
        min_similarity = fp_parameters.get_min_sim_video()
    # The list of tasks involve video fingerprinting and its callback, followed by fingerprint lookup (preprocess,
    # parallel, callback).
    task_list = [
        compute_video_fingerprint_task.s(token),
        compute_video_fingerprint_callback_task.s(),
        video_fingerprint_find_closest_retrieve_from_db_task.s(),
        group(video_fingerprint_find_closest_parallel_task.s(i, n_jobs, min_similarity)
              for i in range(n_jobs)),
        video_fingerprint_find_closest_callback_task.s()
    ]
    # If the fingerprinting is part of another endpoint, its results are ignored, otherwise they are returned.
    if ignore_fp_results:
        task_list += [ignore_fingerprint_results_callback_task.s(results_to_return)]
    else:
        task_list += [retrieve_video_fingerprint_callback_task.s()]
    return task_list


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


def fingerprint_lookup_job(token):
    direct_lookup_job = cache_lookup_fingerprint_video_task.s(token)
    direct_lookup_job = direct_lookup_job.apply_async(priority=6)
    direct_lookup_task_id = direct_lookup_job.id
    # We block on this task since we need its results to decide what to do next
    direct_lookup_results = direct_lookup_job.get(timeout=20)
    # If the cache lookup yielded results, then return the id of the task, otherwise we proceed normally with the
    # computations
    if direct_lookup_results is not None:
        return direct_lookup_task_id
    return None


def fingerprint_job(token, force):
    ##############
    # Cache lookup
    ##############
    # First, we do the direct cache lookup using the caching queue, but only if force=False
    if not force:
        direct_lookup_task_id = fingerprint_lookup_job(token)
        if direct_lookup_task_id is not None:
            return direct_lookup_task_id

    #################
    # Computation job
    #################
    # This is the fingerprinting job, so ignore_fp_results is False
    task_list = get_video_fingerprint_chain_list(token, ignore_fp_results=False)
    task = chain(task_list)
    task = task.apply_async(priority=2)
    return task.id
