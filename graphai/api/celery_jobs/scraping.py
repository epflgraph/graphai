from celery import group, chain

from graphai.api.celery_tasks.scraping import (
    initialize_url_and_get_sublinks_task,
    scraping_sublinks_callback_task,
    process_all_scraping_sublinks_preprocess_task,
    process_all_scraping_sublinks_parallel_task,
    process_all_scraping_sublinks_callback_task,
    remove_junk_scraping_parallel_task,
    remove_junk_scraping_callback_task,
    extract_scraping_content_callback_task,
    scraping_dummy_task
)

from graphai.core.common.scraping import create_base_url_token


def extract_sublinks_job(url, force=False):
    token = create_base_url_token(url)
    task_list = [
        initialize_url_and_get_sublinks_task.s(token, url, force),
        scraping_sublinks_callback_task.s()
    ]
    task = chain(task_list)
    task = task.apply_async(priority=6)
    return task.id


def extract_content_job(url, force=False, headers=False, long_patterns=False):
    n_jobs = 8
    token = create_base_url_token(url)
    task_list = [
        initialize_url_and_get_sublinks_task.s(token, url, force),
        scraping_sublinks_callback_task.s(),
        process_all_scraping_sublinks_preprocess_task.s(headers, long_patterns, force),
        group(process_all_scraping_sublinks_parallel_task.s(i, n_jobs) for i in range(n_jobs)),
        process_all_scraping_sublinks_callback_task.s(),
        scraping_dummy_task.s(),
        group(remove_junk_scraping_parallel_task.s(i, n_jobs, headers, long_patterns) for i in range(n_jobs)),
        remove_junk_scraping_callback_task.s(),
        extract_scraping_content_callback_task.s(headers, long_patterns)
    ]
    task = chain(task_list)
    task = task.apply_async(priority=6)
    return task.id
