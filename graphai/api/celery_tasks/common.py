def format_api_results(id, name, status, result):
    return {
        "task_id": id,
        "task_name": name,
        "task_status": status,
        "task_result": result
    }
