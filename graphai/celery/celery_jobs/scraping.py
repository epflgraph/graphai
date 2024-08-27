from celery import group, chain

from graphai.celery.celery_tasks.scraping import (
    cache_lookup_get_sublinks_task,
    initialize_url_and_get_sublinks_task,
    scraping_sublinks_callback_task,
    cache_lookup_process_all_sublinks_task,
    process_all_scraping_sublinks_parallel_task,
    sublink_parallel_processing_merge_callback_task,
    remove_junk_scraping_parallel_task,
    extract_scraping_content_callback_task,
    scraping_dummy_task
)

from graphai.celery.celery_jobs.common import direct_lookup_generic_job, DEFAULT_TIMEOUT

from graphai.core.scraping.scraping import create_base_url_token


def sublink_lookup_job(token, return_results=False):
    return direct_lookup_generic_job(cache_lookup_get_sublinks_task, token, return_results, DEFAULT_TIMEOUT)


def extract_sublinks_job(url, force=False):
    token = create_base_url_token(url)
    #################################
    # Sublink extraction cache lookup
    #################################
    if not force:
        direct_lookup_task_id = sublink_lookup_job(token)
        if direct_lookup_task_id is not None:
            return direct_lookup_task_id
    #################
    # Computation job
    #################
    task_list = [
        initialize_url_and_get_sublinks_task.s(token, url),
        scraping_sublinks_callback_task.s()
    ]
    task = chain(task_list)
    task = task.apply_async(priority=6)
    return task.id


def extract_content_job(url, force=False, headers=False, long_patterns=False):
    n_jobs = 8
    token = create_base_url_token(url)
    #################################
    # Content extraction cache lookup
    #################################
    if not force:
        direct_lookup_task_id = direct_lookup_generic_job(cache_lookup_process_all_sublinks_task,
                                                          token, False, DEFAULT_TIMEOUT,
                                                          headers, long_patterns)
        if direct_lookup_task_id is not None:
            return direct_lookup_task_id
    #################################
    # Sublink extraction cache lookup
    #################################
    # Here we define the chain of tasks that the job will have to start with if either:
    # 1. force=True
    # 2. There are no cached results
    task_list_with_sublink_extraction = [
        initialize_url_and_get_sublinks_task.s(token, url),
        scraping_sublinks_callback_task.s(),
        scraping_dummy_task.s()
    ]
    if force:
        task_list = task_list_with_sublink_extraction
    else:
        direct_sublinks_lookup_results = sublink_lookup_job(token, return_results=True)
        # Here we use the scraping dummy task to keep the rest of the chain (which starts with a group) simple
        if direct_sublinks_lookup_results is None:
            # If the sublinks haven't been extracted before, we need to extract them
            task_list = task_list_with_sublink_extraction
        else:
            # If the sublinks are cached, then the dummy task will simply pass along the cached results
            task_list = [
                scraping_dummy_task.s(direct_sublinks_lookup_results)
            ]
    #################
    # Computation job
    #################
    task_list += [
        group(process_all_scraping_sublinks_parallel_task.s(i, n_jobs) for i in range(n_jobs)),
        sublink_parallel_processing_merge_callback_task.s(),
        scraping_dummy_task.s(),
        group(remove_junk_scraping_parallel_task.s(i, n_jobs, headers, long_patterns) for i in range(n_jobs)),
        sublink_parallel_processing_merge_callback_task.s(),
        extract_scraping_content_callback_task.s(headers, long_patterns)
    ]
    task = chain(task_list)
    task = task.apply_async(priority=6)
    return task.id
