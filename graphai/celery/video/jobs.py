from celery import (
    chain,
    group
)

from graphai.core.common.logging import get_logger

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

logger = get_logger('graphai.celery.video.jobs')

DEFAULT_SLIDE_TIMEOUT = 90


def get_video_fingerprint_chain_list(token=None, force=False, min_similarity=None, n_jobs=8,
                                     ignore_fp_results=False, request_id=None):
    assert ignore_fp_results or token is not None
    # Retrieve minimum similarity parameter for video fingerprints
    if min_similarity is None:
        fp_parameters = FingerprintParameters()
        min_similarity = fp_parameters.get_min_sim_video()
    # The list of tasks involve video fingerprinting and its callback, followed by fingerprint lookup (preprocess,
    # parallel, callback).
    if ignore_fp_results:
        task_list = [compute_video_fingerprint_task.s(force, request_id=request_id)]
    else:
        task_list = [compute_video_fingerprint_task.s({'token': token}, force, request_id=request_id)]
    task_list += [compute_video_fingerprint_callback_task.s(request_id=request_id)]
    if ignore_fp_results:
        task_list += [video_id_and_duration_fp_lookup_task.s(request_id=request_id)]
    task_list += [
        video_fingerprint_find_closest_retrieve_from_db_task.s(request_id=request_id),
        group(video_fingerprint_find_closest_parallel_task.s(i, n_jobs, min_similarity, request_id=request_id)
              for i in range(n_jobs)),
        video_fingerprint_find_closest_callback_task.s(request_id=request_id)
    ]
    # If the fingerprinting is part of another endpoint, its results are ignored, otherwise they are returned.
    if ignore_fp_results:
        task_list += [ignore_video_fingerprint_results_callback_task.s(request_id=request_id)]
    else:
        task_list += [retrieve_video_fingerprint_callback_task.s(request_id=request_id)]
    return task_list


def get_audio_fingerprint_chain_list(token=None, force=False, min_similarity=None, n_jobs=8,
                                     ignore_fp_results=False, request_id=None):
    assert ignore_fp_results or token is not None
    # Loading minimum similarity parameter for audio
    if min_similarity is None:
        fp_parameters = FingerprintParameters()
        min_similarity = fp_parameters.get_min_sim_audio()
    if ignore_fp_results:
        task_list = [compute_audio_fingerprint_task.s(force, request_id=request_id)]
    else:
        task_list = [compute_audio_fingerprint_task.s({'token': token}, force, request_id=request_id)]
    task_list += [compute_audio_fingerprint_callback_task.s(force, request_id=request_id)]
    if min_similarity == 1:
        task_list += [audio_fingerprint_find_closest_direct_task.s(request_id=request_id)]
    else:
        task_list += [audio_fingerprint_find_closest_retrieve_from_db_task.s(request_id=request_id),
                      group(audio_fingerprint_find_closest_parallel_task.s(i, n_jobs, min_similarity, request_id=request_id) for i in
                            range(n_jobs))
                      ]
    task_list += [audio_fingerprint_find_closest_callback_task.s(request_id=request_id)]
    if ignore_fp_results:
        task_list += [ignore_audio_fingerprint_results_callback_task.s(request_id=request_id)]
    else:
        task_list += [retrieve_audio_fingerprint_callback_task.s(request_id=request_id)]
    return task_list


def get_slide_fingerprint_chain_list(token=None, origin_token=None,
                                     force=False, min_similarity=None, n_jobs=8,
                                     ignore_fp_results=False, request_id=None):
    assert ((token is not None and origin_token is None)
            or (token is None and ignore_fp_results))
    # Loading minimum similarity parameter for image
    if min_similarity is None:
        fp_parameters = FingerprintParameters()
        min_similarity = fp_parameters.get_min_sim_image()
    # The usual fingerprinting task list consists of fingerprinting and its callback, then lookup
    if origin_token is not None:
        # This is for when the chain is called for the /video/detect_slides endpoint, fingerprinting a set of slides
        task_list = [compute_slide_set_fingerprint_task.s(origin_token, request_id=request_id)]
    elif token is not None:
        # This is for the /image/compute_fingerprint endpoint
        task_list = [compute_single_image_fingerprint_task.s({'token': token}, request_id=request_id)]
    else:
        # This is for the /image/retrieve_url or /image/upload_file endpoint
        task_list = [compute_single_image_fingerprint_task.s(request_id=request_id)]
    task_list += [
        compute_slide_fingerprint_callback_task.s(force, request_id=request_id)
    ]
    if min_similarity == 1:
        task_list += [slide_fingerprint_find_closest_direct_task.s(request_id=request_id)]
    else:
        task_list += [
            slide_fingerprint_find_closest_retrieve_from_db_task.s(request_id=request_id),
            group(slide_fingerprint_find_closest_parallel_task.s(i, n_jobs, min_similarity, request_id=request_id) for i in range(n_jobs))
        ]
    task_list += [slide_fingerprint_find_closest_callback_task.s(request_id=request_id)]
    if ignore_fp_results:
        if origin_token is not None:
            task_list += [ignore_slide_fingerprint_results_callback_task.s(request_id=request_id)]
        else:
            task_list += [ignore_single_image_fingerprint_results_callback_task.s(request_id=request_id)]
    else:
        task_list += [retrieve_slide_fingerprint_callback_task.s(request_id=request_id)]
    return task_list


def _task_queue(task_fn):
    """Return the Celery queue name from a task's dotted name."""
    name = getattr(task_fn, 'name', '')
    return name.split('.')[0] if name else 'unknown'


def retrieve_url_job(url, force=False, is_playlist=False, request_id=None):
    log = logger.bind(endpoint='/video/retrieve_url', request_id=request_id, url=url, force=force, is_playlist=is_playlist)
    if not force:
        log.debug('🔍 Checking cache for video URL', target_queue=_task_queue(cache_lookup_retrieve_file_from_url_task))
        direct_lookup_task_id = direct_lookup_generic_job(
            cache_lookup_retrieve_file_from_url_task, url, False, DEFAULT_TIMEOUT, request_id=request_id
        )
        if direct_lookup_task_id is not None:
            log.info('✅ Video URL cache hit', cache_task_id=direct_lookup_task_id, target_queue=_task_queue(cache_lookup_retrieve_file_from_url_task))
            return direct_lookup_task_id
        log.info('⏭️ Video URL cache miss; starting retrieval')

    # Overriding the is_playlist flag if the url ends with m3u8 (playlist) or mp4/mkv/flv/avi/mov (video file)
    if url.endswith('.m3u8'):
        is_playlist = True
    elif any([url.endswith(e) for e in ['.mp4', '.mkv', '.flv', '.avi', '.mov']]):
        is_playlist = False
    # First retrieve the file, and then do the database callback
    task_list = [
        retrieve_file_from_url_task.s(url, is_playlist, None, request_id=request_id),
        retrieve_file_from_url_callback_task.s(url, request_id=request_id),
    ]
    task_list += get_video_fingerprint_chain_list(None, ignore_fp_results=True, request_id=request_id)
    task = chain(task_list)
    task = task.apply_async(priority=2)
    log.info('🚀 Submitted video retrieve_url job', task_id=task.id, target_queue='video')
    return task.id


def fingerprint_lookup_job(token, request_id=None):
    return direct_lookup_generic_job(
        cache_lookup_fingerprint_video_task, token, False, DEFAULT_TIMEOUT, request_id=request_id
    )


def fingerprint_job(token, force, request_id=None):
    log = logger.bind(endpoint='/video/calculate_fingerprint', request_id=request_id, token=token, force=force)
    ##############
    # Cache lookup
    ##############
    if not force:
        log.debug('🔍 Checking cache for video fingerprint', target_queue=_task_queue(cache_lookup_fingerprint_video_task))
        direct_lookup_task_id = fingerprint_lookup_job(token, request_id=request_id)
        if direct_lookup_task_id is not None:
            log.info('✅ Video fingerprint cache hit', cache_task_id=direct_lookup_task_id)
            return direct_lookup_task_id
        log.info('⏭️ Video fingerprint cache miss; starting computation')


    #################
    # Computation job
    #################
    task_list = get_video_fingerprint_chain_list(token, ignore_fp_results=False, force=force, request_id=request_id)
    task = chain(task_list)
    task = task.apply_async(priority=2)
    log.info('🚀 Submitted video fingerprint job', task_id=task.id, target_queue='video')
    return task.id


def extract_audio_job(token, force=False, recalculate_cached=False, request_id=None):
    log = logger.bind(
        endpoint='/video/extract_audio',
        request_id=request_id,
        token=token,
        force=force,
        recalculate_cached=recalculate_cached,
    )
    ############################
    # Extract audio cache lookup
    ############################
    if not force and not recalculate_cached:
        log.debug('🔍 Checking cache for audio extraction', target_queue=_task_queue(cache_lookup_extract_audio_task))
        direct_lookup_task_id = direct_lookup_generic_job(
            cache_lookup_extract_audio_task, token, False, DEFAULT_TIMEOUT, request_id=request_id
        )
        if direct_lookup_task_id is not None:
            log.info('✅ Audio extraction cache hit', cache_task_id=direct_lookup_task_id)
            return direct_lookup_task_id
        log.info('⏭️ Audio extraction cache miss; starting extraction')


    #################
    # (Re)Computation
    #################
    if not recalculate_cached:
        task_list = [
            extract_audio_task.s(token, request_id=request_id),
            extract_audio_callback_task.s(token, force, request_id=request_id)
        ]
    else:
        task_list = [reextract_cached_audio_task.s(token, request_id=request_id)]

    ################
    # Fingerprinting
    ################
    task_list += get_audio_fingerprint_chain_list(None, ignore_fp_results=True, request_id=request_id)

    task = chain(task_list)
    task = task.apply_async(priority=2)
    log.info('🚀 Submitted video extract_audio job', task_id=task.id, target_queue='video')
    return task.id


def detect_slides_job(token, language, force=False, recalculate_cached=False, request_id=None, **kwargs):
    log = logger.bind(
        endpoint='/video/detect_slides',
        request_id=request_id,
        token=token,
        force=force,
        recalculate_cached=recalculate_cached,
        language=language,
    )
    ############################
    # Detect slides cache lookup
    ############################
    if not force and not recalculate_cached:
        log.debug('🔍 Checking cache for slide detection', target_queue=_task_queue(cache_lookup_detect_slides_task))
        direct_lookup_task_id = direct_lookup_generic_job(
            cache_lookup_detect_slides_task, token, False, DEFAULT_SLIDE_TIMEOUT, request_id=request_id
        )
        if direct_lookup_task_id is not None:
            log.info('✅ Slide detection cache hit', cache_task_id=direct_lookup_task_id)
            return direct_lookup_task_id
        log.info('⏭️ Slide detection cache miss; starting detection')


    #################
    # (Re)Computation
    #################
    if not recalculate_cached:
        task_list = [extract_and_sample_frames_task.s(token, request_id=request_id)]
        # Now we add the rest
        n_jobs = 8
        # This is the maximum similarity threshold used for image hashes when finding slide transitions.
        default_hash_thresh = 0.95
        default_multiplier = 5
        default_threshold = 0.05
        # The dummy task is there because of a celery peculiarity where a group chord cannot be immediately followed
        # by another group.
        task_list += [group(compute_noise_level_parallel_task.s(i, n_jobs, language, request_id=request_id) for i in range(n_jobs)),
                      compute_noise_threshold_callback_task.s(
                          kwargs.get('hash_thresh', default_hash_thresh),
                          kwargs.get('multiplier', default_multiplier),
                          kwargs.get('default_threshold', default_threshold),
                          request_id=request_id),
                      video_dummy_task.s(request_id=request_id),
                      group(compute_slide_transitions_parallel_task.s(
                          i,
                          n_jobs,
                          language,
                          True if i > 0 else kwargs.get('include_first', True),
                          True if i < n_jobs - 1 else kwargs.get('include_last', True),
                          request_id=request_id
                      ) for i in range(n_jobs)),
                      compute_slide_transitions_callback_task.s(language, request_id=request_id),
                      detect_slides_callback_task.s(token, force, request_id=request_id)]
    else:
        task_list = [reextract_cached_slides_task.s(token, request_id=request_id)]

    task_list += get_slide_fingerprint_chain_list(origin_token=token, ignore_fp_results=True, request_id=request_id)
    task = chain(task_list)
    task = task.apply_async(priority=2)
    log.info('🚀 Submitted video detect_slides job', task_id=task.id, target_queue='video')
    return task.id


def get_file_job(token, request_id=None):
    log = logger.bind(endpoint='/video/get_file', request_id=request_id, token=token)
    log.info('📁 Fetching video file path')
    task = get_file_task.s(token, request_id=request_id)
    result = task.apply_async(priority=2).get(timeout=300)
    log.info('✅ Video file path fetched', file_path=result)
    return result

