from pydantic import BaseModel, Field
from typing import Union, Literal, List, Dict

from graphai.api.common.schemas import TaskStatusResponse


class RetrievalRequest(BaseModel):
    text: str = Field(
        title="Text",
        description="Text to search for"
    )

    index: Literal['lex'] = Field(
        title="Index",
        description="Index to search in.",
        default='lex'
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


class RetrievalTaskResponse(BaseModel):
    n_results: int = Field(
        title="Number of results"
    )
    result: Union[List[dict], None] = Field(
        title="Retrieval results",
        description="Retrieval results"
    )

    successful: bool = Field(
        title="Success flag",
        description="Whether or not the retrieval was successful"
    )


class RetrievalResponse(TaskStatusResponse):
    task_result: Union[RetrievalTaskResponse, None] = Field(
        title="Retrieval response",
        description="A dict containing the result of the ES index retrieval."
    )
