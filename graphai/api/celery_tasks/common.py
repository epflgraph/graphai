from graphai.core.celery_utils.celery_utils import get_celery_task


def format_results(id, name, status, result):
    return {
        "task_id": id,
        "task_name": name,
        "task_status": status,
        "task_result": result
    }


def get_task_results(task_id):
    """
    return task info for the given task_id
    """
    task_result = get_celery_task(task_id)
    result = format_results(task_id, task_result.name, task_result.status, task_result.result)
    print(result)
    return result