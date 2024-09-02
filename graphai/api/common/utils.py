def format_api_results(id, name, status, result):
    """
    Formats results coming from celery into the common output format of the API
    Args:
        id: Id of the task
        name: Name of the task
        status: Task status
        result: Task results

    Returns:
        Appropriately formatted results dictionary
    """
    return {
        "task_id": id,
        "task_name": name,
        "task_status": status,
        "task_result": result
    }
