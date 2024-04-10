from celery import chain, group

from graphai.api.celery_tasks.common import ignore_fingerprint_results_callback_task
from graphai.api.celery_tasks.common import video_dummy_task
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
    retrieve_video_fingerprint_callback_task,
    cache_lookup_extract_audio_task,
    extract_audio_task,
    extract_audio_callback_task,
    reextract_cached_audio_task,
    cache_lookup_detect_slides_task,
    extract_and_sample_frames_task,
    compute_noise_level_parallel_task,
    compute_noise_threshold_callback_task,
    compute_slide_transitions_parallel_task,
    compute_slide_transitions_callback_task,
    detect_slides_callback_task,
    reextract_cached_slides_task,
    get_file_task
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


def extract_audio_job(token, force=False, recalculate_cached=False):
    ############################
    # Extract audio cache lookup
    ############################
    # We only do the cache lookup if both force and recalculate_cached flags are False
    # If force=True, we explicitly want to skip the cached results
    # If recalculate_cached=True, we want to re-extract the audio based on cache, and not just return the cached result!
    if not force and not recalculate_cached:
        direct_lookup_job = cache_lookup_extract_audio_task.s(token)
        direct_lookup_job = direct_lookup_job.apply_async(priority=6)
        direct_lookup_task_id = direct_lookup_job.id
        # We block on this task since we need its results to decide what to do next
        direct_lookup_results = direct_lookup_job.get(timeout=20)
        # If the cache lookup yielded results, then return the id of the task, otherwise we proceed normally with the
        # computations
        if direct_lookup_results is not None:
            return direct_lookup_task_id

    ##########################
    # Fingerprint cache lookup
    ##########################
    # Fingerprinting is never forced in non-fingerprinting jobs
    direct_fingerprint_lookup_task_id = fingerprint_lookup_job(token)

    #################
    # (Re)Computation
    #################
    if direct_fingerprint_lookup_task_id is None:
        # If the fingerprint is not cached, we add the fingerprinting chain
        task_list = get_video_fingerprint_chain_list(token, ignore_fp_results=True, results_to_return=token)
        if not recalculate_cached:
            # If the recalculate flag is not set, we add the normal audio extraction chain
            task_list += [
                extract_audio_task.s(),
                extract_audio_callback_task.s(token, force)
            ]
        else:
            # If a recalculation is requested, we add the re-extraction task
            task_list += [reextract_cached_audio_task.s()]
    else:
        # If the fingerprint is already cached
        # Same as above, but with the starting tasks having their full signature since fingerprinting is skipped
        if not recalculate_cached:
            task_list = [
                extract_audio_task.s(token),
                extract_audio_callback_task.s(token, force)
            ]
        else:
            task_list = [reextract_cached_audio_task.s(token)]

    task = chain(task_list)
    task = task.apply_async(priority=2)
    return task.id


def detect_slides_job(token, language, force=False, recalculate_cached=False):
    ############################
    # Detect slides cache lookup
    ############################
    # We only do the cache lookup if both force and recalculate_cached flags are False
    # If force=True, we explicitly want to skip the cached results
    # If recalculate_cached=True, we want to re-extract the audio based on cache, and not just return the cached result!
    if not force and not recalculate_cached:
        direct_lookup_job = cache_lookup_detect_slides_task.s(token)
        direct_lookup_job = direct_lookup_job.apply_async(priority=6)
        direct_lookup_task_id = direct_lookup_job.id
        # We block on this task since we need its results to decide what to do next
        direct_lookup_results = direct_lookup_job.get(timeout=20)
        # If the cache lookup yielded results, then return the id of the task, otherwise we proceed normally with the
        # computations
        if direct_lookup_results is not None:
            return direct_lookup_task_id

    ##########################
    # Fingerprint cache lookup
    ##########################
    # Fingerprinting is never forced in non-fingerprinting jobs
    direct_fingerprint_lookup_task_id = fingerprint_lookup_job(token)

    #################
    # (Re)Computation
    #################
    if not recalculate_cached:
        if direct_fingerprint_lookup_task_id is None:
            # If there's no fingerprint, add the fingerprinting chain
            task_list = get_video_fingerprint_chain_list(token, ignore_fp_results=True, results_to_return=token)
            task_list += [extract_and_sample_frames_task.s()]
        else:
            # Otherwise the first task gets its full signature
            task_list = [extract_and_sample_frames_task.s(token)]
        # Now we add the rest
        n_jobs = 8
        # This is the maximum similarity threshold used for image hashes when finding slide transitions.
        hash_thresh = 0.95
        # The dummy task is there because of a celery peculiarity where a group chord cannot be immediately followed
        # by another group.
        task_list += [group(compute_noise_level_parallel_task.s(i, n_jobs, language) for i in range(n_jobs)),
                      compute_noise_threshold_callback_task.s(hash_thresh),
                      video_dummy_task.s(),
                      group(compute_slide_transitions_parallel_task.s(i, n_jobs, language) for i in range(n_jobs)),
                      compute_slide_transitions_callback_task.s(language),
                      detect_slides_callback_task.s(token, force)]
    else:
        if direct_fingerprint_lookup_task_id is None:
            # If there's no fingerprint, add the fingerprinting chain
            task_list = get_video_fingerprint_chain_list(token, ignore_fp_results=True, results_to_return=token)
            task_list += [reextract_cached_slides_task.s()]
        else:
            # Full signature for first task
            task_list = [reextract_cached_slides_task.s(token)]

    task = chain(task_list)
    task = task.apply_async(priority=2)
    return task.id


def get_file_job(token):
    task = get_file_task.s(token)
    result = task.apply_async(priority=2).get(timeout=300)
    return result
