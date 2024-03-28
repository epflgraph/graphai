from celery import group, chain

from graphai.api.celery_tasks.translation import (
    translate_text_task,
    translate_text_callback_task,
    detect_text_language_task,
    cache_lookup_translation_text_fingerprint_task,
    cache_lookup_translate_text_task,
    compute_translation_text_fingerprint_task,
    compute_translation_text_fingerprint_callback_task,
    translation_text_fingerprint_find_closest_retrieve_from_db_task,
    translation_text_fingerprint_find_closest_parallel_task,
    translation_text_fingerprint_find_closest_direct_task,
    translation_text_fingerprint_find_closest_callback_task,
    translation_retrieve_text_fingerprint_callback_task,
)

from graphai.api.celery_tasks.common import (
    ignore_fingerprint_results_callback_task,
)

from graphai.core.common.text_utils import (
    generate_src_tgt_dict,
    generate_translation_text_token,
    translation_list_to_text
)

from graphai.core.common.caching import FingerprintParameters


def get_translation_text_fingerprint_chain_list(token, text, src, tgt, min_similarity=None, n_jobs=8,
                                                ignore_fp_results=False, results_to_return=None):
    # Loading min similarity parameter for text
    if min_similarity is None:
        fp_parameters = FingerprintParameters()
        min_similarity = fp_parameters.get_min_sim_text()
    print(min_similarity)
    print(min_similarity == 1.0)

    # Generating the equality condition dictionary
    equality_conditions = generate_src_tgt_dict(src, tgt)
    # The tasks are fingerprinting and callback, then lookup. The lookup is only among cache rows that satisfy the
    # equality conditions (source and target languages).
    task_list = [
        compute_translation_text_fingerprint_task.s(token, text),
        compute_translation_text_fingerprint_callback_task.s(text, src, tgt)
    ]
    if min_similarity == 1.0:
        task_list += [translation_text_fingerprint_find_closest_direct_task.s(equality_conditions)]
    else:
        task_list += [
            translation_text_fingerprint_find_closest_retrieve_from_db_task.s(equality_conditions),
            group(translation_text_fingerprint_find_closest_parallel_task.s(i, n_jobs, equality_conditions, min_similarity)
                  for i in range(n_jobs))
        ]
    task_list += [translation_text_fingerprint_find_closest_callback_task.s()]
    if ignore_fp_results:
        task_list += [ignore_fingerprint_results_callback_task.s(results_to_return)]
    else:
        task_list += [translation_retrieve_text_fingerprint_callback_task.s()]
    return task_list


def fingerprint_lookup_job(token):
    direct_lookup_job = cache_lookup_translation_text_fingerprint_task.s(token)
    direct_lookup_job = direct_lookup_job.apply_async(priority=6)
    direct_lookup_task_id = direct_lookup_job.id
    # We block on this task since we need its results to decide what to do next
    direct_lookup_results = direct_lookup_job.get(timeout=20)
    # If the cache lookup yielded results, then return the id of the task, otherwise we proceed normally with the
    # computations
    if direct_lookup_results is not None:
        return direct_lookup_task_id
    return None


def fingerprint_job(text, src, tgt, force):
    token = generate_translation_text_token(text, src, tgt)
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
    text = translation_list_to_text(text)
    task_list = get_translation_text_fingerprint_chain_list(token, text, src, tgt,
                                                            ignore_fp_results=False)
    task = chain(task_list)
    task = task.apply_async(priority=6)
    return task.id


def translation_job(text, src, tgt, force):
    token = generate_translation_text_token(text, src, tgt)
    return_list = isinstance(text, list)
    text = translation_list_to_text(text)
    ##########################
    # Translation cache lookup
    ##########################
    if not force:
        direct_lookup_job = cache_lookup_translate_text_task.s(token, return_list)
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
    # Now we do a fingerprint lookup to see if we can skip the fingerprinting chain or if it needs to be included
    direct_fingerprint_lookup_task_id = fingerprint_lookup_job(token)
    if direct_fingerprint_lookup_task_id is None:
        # If the result is None, it means that there is no cached fingerprint, so we need to do fingerprinting
        # The first task, "translate_text_task", will only have 'src' and 'tgt' in its signature
        # because the first argument (which is 'text') will come from the preceding task chain.
        task_list = get_translation_text_fingerprint_chain_list(token, text, src, tgt,
                                                                ignore_fp_results=True, results_to_return=text)
        task_list += [translate_text_task.s(src, tgt)]
    else:
        # We end up here if the fingerprint results are already cached. Remember that setting force=True
        # in the translation endpoint only forces a re-translation, NOT a re-fingerprinting!

        # If "translate_text_task" is the first task in the chain, then it'll have the full signature.
        task_list = [translate_text_task.s(text, src, tgt)]
    #########################
    # Rest of computation job
    #########################
    task_list += [
        translate_text_callback_task.s(token, text, src, tgt, force, return_list)
    ]
    task = chain(task_list)
    task = task.apply_async(priority=6)
    return task.id


def detect_text_language_job(text):
    text = translation_list_to_text(text)
    # The only task is language detection because this task does not go through the caching logic
    task = (detect_text_language_task.s(text)).apply_async(priority=6)
    return task.id
