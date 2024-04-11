def direct_lookup_generic_job(task_fn, token, *args):
    """
    Launches a direct cache lookup job
    Args:
        token: The token to look up in the cache
        task_fn: The task function of the cache lookup

    Returns:
        The id of the cache lookup task in case of a cache hit, None in case of a miss
    """
    args = list(args)
    if len(args) == 0:
        direct_lookup_job = task_fn.s(token)
    else:
        direct_lookup_job = task_fn.s(token, *args)
    direct_lookup_job = direct_lookup_job.apply_async(priority=6)
    direct_lookup_task_id = direct_lookup_job.id
    # We block on this task since we need its results to decide what to do next
    direct_lookup_results = direct_lookup_job.get(timeout=20)
    # If the cache lookup yielded results, then return the id of the task, otherwise we proceed normally with the
    # computations
    if direct_lookup_results is not None:
        return direct_lookup_task_id
    return None
