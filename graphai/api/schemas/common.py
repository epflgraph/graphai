from pydantic import BaseModel, Field, Json
from typing import List, Dict, Union
import abc

class TaskIDResponse(BaseModel):
    """
    Object representing the output of the /ontology/tree endpoint.
    """

    task_id: str = Field(
        ...,
        title="Task ID",
        description="ID of the task created as a response to an API request"
    )

# This class follows the response model of our celery get_task_info function:
# task_id, task_status, and task_result (the latter of which is implemented by child classes).
class TaskStatusResponse(BaseModel, abc.ABC):
    task_id: str = Field(
        title="Task ID",
        description="ID of the task created as a response to an API request"
    )
    task_status: str = Field(
        title="Task status",
        description="Status of the task"
    )
    task_result: Json[BaseModel]

