from pydantic import BaseModel, Field
from typing import Union, Literal, List, Dict

from graphai.api.common.schemas import TaskStatusResponse


class LexRetrievalRequest(BaseModel):
    text: str = Field(
        title="Text",
        description="Text to search for"
    )

    lang: Literal['en', 'fr', None] = Field(
        title="Language filter",
        description="Only retrieves documents that are originally in the provided language. "
                    "If left empty, all documents will be searched.",
        default=None
    )

    limit: int = Field(
        title="Limit",
        description="Number of search results to return",
        default=10
    )


class LexRetrievalTaskResponse(BaseModel):
    result: Union[List[dict], None] = Field(
        title="Retrieval results",
        description="Retrieval results"
    )

    successful: bool = Field(
        title="Success flag",
        description="Whether or not the retrieval was successful"
    )


class LexRetrievalResponse(TaskStatusResponse):
    task_result: Union[LexRetrievalTaskResponse, None] = Field(
        title="Retrieval response",
        description="A dict containing the result of the LEX index retrieval."
    )
