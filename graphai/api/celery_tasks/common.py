from graphai.core.celery_utils.celery_utils import get_celery_task


def format_results(id, name, status, result):
    return {
        "task_id": id,
        "task_name": name,
        "task_status": status,
        "task_result": result
    }


def compile_task_results(task_id, task_results=None):
    """
    return task info for the given task_id
    """
    task = get_celery_task(task_id)
    if task_results is None:
        task_results = task.result
    return format_results(task_id, task.name, task.status, task_results)

