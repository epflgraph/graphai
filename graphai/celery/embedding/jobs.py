from celery import chain, group

from graphai.celery.embedding.tasks import (
    cache_lookup_embedding_text_fingerprint_task,
    compute_embedding_text_fingerprint_task,
    compute_embedding_text_fingerprint_callback_task,
    cache_lookup_embedding_text_task,
    cache_lookup_embedding_text_using_fingerprint_task,
    embed_text_task,
    embed_text_callback_task,
    embedding_text_list_fingerprint_parallel_task,
    embedding_text_list_fingerprint_callback_task,
    embedding_text_list_embed_parallel_task,
    embedding_text_list_embed_callback_task
)

from graphai.celery.common.tasks import text_dummy_task

from graphai.celery.common.jobs import (
    direct_lookup_generic_job,
    DEFAULT_TIMEOUT
)

from graphai.core.embedding.embedding import generate_embedding_text_token


def get_embedding_text_fingerprint_chain_list(token, text, model_type):
    # The tasks are fingerprinting and callback. The callback contains a lookup as well.
    task_list = [
        compute_embedding_text_fingerprint_task.s(token, text),
        compute_embedding_text_fingerprint_callback_task.s(text, model_type)
    ]
    return task_list


def fingerprint_lookup_job(token, return_results=False):
    return direct_lookup_generic_job(cache_lookup_embedding_text_fingerprint_task, token,
                                     return_results, DEFAULT_TIMEOUT)


def fingerprint_compute_job(token, text, model_type, asynchronous=True):
    task_list = get_embedding_text_fingerprint_chain_list(token, text, model_type)
    task = chain(task_list)
    task = task.apply_async(priority=6)
    task_id = task.id
    if asynchronous:
        return task_id
    else:
        results = task.get(timeout=20)
        if results is not None:
            return results
        else:
            return None


def fingerprint_job(text, model_type, force):
    token = generate_embedding_text_token(text, model_type)
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
    return fingerprint_compute_job(token, text, model_type, asynchronous=True)


def embedding_job(text, model_type, force):
    if isinstance(text, str):
        token = generate_embedding_text_token(text, model_type)
        ########################
        # Embedding cache lookup
        ########################
        if not force:
            direct_lookup_task_id = direct_lookup_generic_job(cache_lookup_embedding_text_task, token,
                                                              False, DEFAULT_TIMEOUT, model_type)
            if direct_lookup_task_id is not None:
                return direct_lookup_task_id

        ####################################
        # Fingerprinting and fp-based lookup
        ####################################
        current_fingerprint = None
        # Fingerprint cache lookup
        direct_fingerprint_lookup_results = fingerprint_lookup_job(token, return_results=True)
        if direct_fingerprint_lookup_results is not None:
            current_fingerprint = direct_fingerprint_lookup_results['result']
        # If the fingerprint was not cached, fingerprinting
        if current_fingerprint is None:
            fp_computation_results = fingerprint_compute_job(token, text, model_type, asynchronous=False)
            current_fingerprint = fp_computation_results['result']
        # Fingerprint-based lookup
        if not force and current_fingerprint is not None:
            fp_based_lookup_task_id = direct_lookup_generic_job(cache_lookup_embedding_text_using_fingerprint_task,
                                                                token, False, DEFAULT_TIMEOUT,
                                                                current_fingerprint, model_type)
            if fp_based_lookup_task_id is not None:
                return fp_based_lookup_task_id
        # If we're here, both lookups have failed, so it's time for the actual computation
        #################
        # Computation job
        #################
        task_list = [
            embed_text_task.s(text, model_type),
            embed_text_callback_task.s(token, text, model_type, force)
        ]
    else:
        tokens = [generate_embedding_text_token(x, model_type) for x in text]
        task_list = [
            group(embedding_text_list_fingerprint_parallel_task.s(tokens, text, i, 8) for i in range(8)),
            embedding_text_list_fingerprint_callback_task.s(model_type),
            text_dummy_task.s(),
            group(embedding_text_list_embed_parallel_task.s(model_type, i, 8, force) for i in range(8)),
            embedding_text_list_embed_callback_task.s(model_type, force),
            text_dummy_task.s()
        ]
    task = chain(task_list)
    task = task.apply_async(priority=6)
    return task.id
