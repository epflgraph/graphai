from celery import (
    chain,
    group
)


from graphai.celery.common.tasks import video_dummy_task
from graphai.celery.video.tasks import (
    cache_lookup_retrieve_file_from_url_task,
    retrieve_file_from_url_task,
    retrieve_file_from_url_callback_task,
    cache_lookup_fingerprint_video_task,
    compute_video_fingerprint_task,
    compute_video_fingerprint_callback_task,
    video_id_and_duration_fp_lookup_task,
    video_fingerprint_find_closest_retrieve_from_db_task,
    video_fingerprint_find_closest_parallel_task,
    video_fingerprint_find_closest_callback_task,
    retrieve_video_fingerprint_callback_task,
    ignore_video_fingerprint_results_callback_task,
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
    get_file_task, compute_audio_fingerprint_task,
    compute_audio_fingerprint_callback_task,
    audio_fingerprint_find_closest_retrieve_from_db_task,
    audio_fingerprint_find_closest_parallel_task,
    audio_fingerprint_find_closest_direct_task,
    audio_fingerprint_find_closest_callback_task,
    retrieve_audio_fingerprint_callback_task,
    ignore_audio_fingerprint_results_callback_task,
    compute_single_image_fingerprint_task,
    compute_slide_set_fingerprint_task,
    compute_slide_fingerprint_callback_task,
    slide_fingerprint_find_closest_retrieve_from_db_task,
    slide_fingerprint_find_closest_parallel_task,
    slide_fingerprint_find_closest_direct_task,
    slide_fingerprint_find_closest_callback_task,
    retrieve_slide_fingerprint_callback_task,
    ignore_slide_fingerprint_results_callback_task,
    ignore_single_image_fingerprint_results_callback_task
)
from graphai.celery.common.jobs import (
    direct_lookup_generic_job,
    DEFAULT_TIMEOUT
)
from graphai.core.common.caching import FingerprintParameters

DEFAULT_SLIDE_TIMEOUT = 90


def get_video_fingerprint_chain_list(token=None, force=False, min_similarity=None, n_jobs=8,
                                     ignore_fp_results=False):
    assert ignore_fp_results or token is not None
    # Retrieve minimum similarity parameter for video fingerprints
    if min_similarity is None:
        fp_parameters = FingerprintParameters()
        min_similarity = fp_parameters.get_min_sim_video()
    # The list of tasks involve video fingerprinting and its callback, followed by fingerprint lookup (preprocess,
    # parallel, callback).
    if ignore_fp_results:
        task_list = [compute_video_fingerprint_task.s(force)]
    else:
        task_list = [compute_video_fingerprint_task.s({'token': token}, force)]
    task_list += [compute_video_fingerprint_callback_task.s()]
    if ignore_fp_results:
        task_list += [video_id_and_duration_fp_lookup_task.s()]
    task_list += [
        video_fingerprint_find_closest_retrieve_from_db_task.s(),
        group(video_fingerprint_find_closest_parallel_task.s(i, n_jobs, min_similarity)
              for i in range(n_jobs)),
        video_fingerprint_find_closest_callback_task.s()
    ]
    # If the fingerprinting is part of another endpoint, its results are ignored, otherwise they are returned.
    if ignore_fp_results:
        task_list += [ignore_video_fingerprint_results_callback_task.s()]
    else:
        task_list += [retrieve_video_fingerprint_callback_task.s()]
    return task_list


def get_audio_fingerprint_chain_list(token=None, force=False, min_similarity=None, n_jobs=8,
                                     ignore_fp_results=False):
    assert ignore_fp_results or token is not None
    # Loading minimum similarity parameter for audio
    if min_similarity is None:
        fp_parameters = FingerprintParameters()
        min_similarity = fp_parameters.get_min_sim_audio()
    if ignore_fp_results:
        task_list = [compute_audio_fingerprint_task.s(force)]
    else:
        task_list = [compute_audio_fingerprint_task.s({'token': token}, force)]
    task_list += [compute_audio_fingerprint_callback_task.s(force)]
    if min_similarity == 1:
        task_list += [audio_fingerprint_find_closest_direct_task.s()]
    else:
        task_list += [audio_fingerprint_find_closest_retrieve_from_db_task.s(),
                      group(audio_fingerprint_find_closest_parallel_task.s(i, n_jobs, min_similarity) for i in
                            range(n_jobs))
                      ]
    task_list += [audio_fingerprint_find_closest_callback_task.s()]
    if ignore_fp_results:
        task_list += [ignore_audio_fingerprint_results_callback_task.s()]
    else:
        task_list += [retrieve_audio_fingerprint_callback_task.s()]
    return task_list


def get_slide_fingerprint_chain_list(token=None, origin_token=None,
                                     force=False, min_similarity=None, n_jobs=8,
                                     ignore_fp_results=False):
    assert ((token is not None and origin_token is None)
            or (token is None and ignore_fp_results))
    # Loading minimum similarity parameter for image
    if min_similarity is None:
        fp_parameters = FingerprintParameters()
        min_similarity = fp_parameters.get_min_sim_image()
    # The usual fingerprinting task list consists of fingerprinting and its callback, then lookup
    if origin_token is not None:
        task_list = [compute_slide_set_fingerprint_task.s(origin_token)]
    elif token is not None:
        task_list = [compute_single_image_fingerprint_task.s({'token': token})]
    else:
        task_list = [compute_single_image_fingerprint_task.s()]
    task_list += [
        compute_slide_fingerprint_callback_task.s(force)
    ]
    if min_similarity == 1:
        task_list += [slide_fingerprint_find_closest_direct_task.s()]
    else:
        task_list += [
            slide_fingerprint_find_closest_retrieve_from_db_task.s(),
            group(slide_fingerprint_find_closest_parallel_task.s(i, n_jobs, min_similarity) for i in range(n_jobs))
        ]
    task_list += [slide_fingerprint_find_closest_callback_task.s()]
    if ignore_fp_results:
        if origin_token is not None:
            task_list += [ignore_slide_fingerprint_results_callback_task.s()]
        else:
            task_list += [ignore_single_image_fingerprint_results_callback_task.s()]
    else:
        task_list += [retrieve_slide_fingerprint_callback_task.s()]
    return task_list


def retrieve_url_job(url, force=False, is_playlist=False):
    if not force:
        direct_lookup_task_id = direct_lookup_generic_job(cache_lookup_retrieve_file_from_url_task, url,
                                                          False, DEFAULT_TIMEOUT)
        if direct_lookup_task_id is not None:
            return direct_lookup_task_id

    # Overriding the is_playlist flag if the url ends with m3u8 (playlist) or mp4/mkv/flv/avi/mov (video file)
    if url.endswith('.m3u8'):
        is_playlist = True
    elif any([url.endswith(e) for e in ['.mp4', '.mkv', '.flv', '.avi', '.mov']]):
        is_playlist = False
    # First retrieve the file, and then do the database callback
    task_list = [retrieve_file_from_url_task.s(url, is_playlist, None),
                 retrieve_file_from_url_callback_task.s(url)]
    task_list += get_video_fingerprint_chain_list(None, ignore_fp_results=True)
    task = chain(task_list)
    task = task.apply_async(priority=2)
    return task.id


def fingerprint_lookup_job(token):
    return direct_lookup_generic_job(cache_lookup_fingerprint_video_task, token, False, DEFAULT_TIMEOUT)


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
    task_list = get_video_fingerprint_chain_list(token, ignore_fp_results=False, force=force)
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
        direct_lookup_task_id = direct_lookup_generic_job(cache_lookup_extract_audio_task, token,
                                                          False, DEFAULT_TIMEOUT)
        if direct_lookup_task_id is not None:
            return direct_lookup_task_id

    #################
    # (Re)Computation
    #################

    if not recalculate_cached:
        task_list = [
            extract_audio_task.s(token),
            extract_audio_callback_task.s(token, force)
        ]
    else:
        task_list = [reextract_cached_audio_task.s(token)]

    ################
    # Fingerprinting
    ################
    task_list += get_audio_fingerprint_chain_list(None, ignore_fp_results=True)

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
        direct_lookup_task_id = direct_lookup_generic_job(cache_lookup_detect_slides_task, token,
                                                          False, DEFAULT_SLIDE_TIMEOUT)
        if direct_lookup_task_id is not None:
            return direct_lookup_task_id

    #################
    # (Re)Computation
    #################
    if not recalculate_cached:
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
        task_list = [reextract_cached_slides_task.s(token)]

    task_list += get_slide_fingerprint_chain_list(origin_token=token, ignore_fp_results=True)
    task = chain(task_list)
    task = task.apply_async(priority=2)
    return task.id


def get_file_job(token):
    task = get_file_task.s(token)
    result = task.apply_async(priority=2).get(timeout=300)
    return result
