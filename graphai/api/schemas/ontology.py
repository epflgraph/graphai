from pydantic import BaseModel, Field
from typing import List, Dict
from .common import TaskStatusResponse


class TreeResponseElem(BaseModel):
    """
    Object representing the output of the /ontology/tree endpoint.
    """

    ChildCategoryID: int = Field(
        ...,
        title="Child category ID",
        description="ID of the child category"
    )

    ParentCategoryID: int = Field(
        ...,
        title="Parent category ID",
        description="ID of the parent category"
    )


class TreeResponse(TaskStatusResponse):
    task_result: List[TreeResponseElem] = Field(
        title="Ontology tree results",
        description="The child-parent relationships of the ontology's predefined tree as a list of dicts."
    )
