import abc

from pydantic import BaseModel, Field, Json
from typing import Union, List


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
# task_id, task_name, task_status, and task_result (the latter of which must be overwritten by child classes).
#
# Create one child of this class per endpoint. Be sure to override the `task_result` attribute.
class TaskStatusResponse(BaseModel, abc.ABC):
    task_id: str = Field(
        title="Task ID",
        description="ID of the task created as a response to an API request"
    )
    task_name: Union[str, None] = Field(
        title="Task name",
        description="Name of the task"
    )
    task_status: str = Field(
        title="Task status",
        description="Status of the task"
    )
    task_result: Json[BaseModel]


class FileRequest(BaseModel):
    token: str = Field(
        title="File name",
        description="The name of the file to be downloaded (received as a response from another endpoint)."
    )


class TokenStatus(BaseModel):
    active: bool = Field(
        title="Token active",
        description="Whether the token's file is available (which makes calculations possible, otherwise only cached "
                    "results can be returned for this token)."
    )

    cached: List[str] = Field(
        title="Cached results",
        description="List of endpoints whose results have already been cached for this token"
    )
