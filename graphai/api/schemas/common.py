from pydantic import BaseModel, Field
from typing import List
import abc

class TaskIDResponse(BaseModel):
    """
    Object representing the output of the /ontology/tree endpoint.
    """

    TaskID: str = Field(
        ...,
        title="Task ID",
        description="ID of the task created as a response to an API request"
    )

class TaskStatusResponse(BaseModel, abc.ABC):
    task_id: str = Field(
        title="Task ID",
        description="ID of the task created as a response to an API request"
    )
    task_status: str = Field(
        title="Task status",
        description="Status of the task"
    )

