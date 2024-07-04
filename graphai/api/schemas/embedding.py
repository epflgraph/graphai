from pydantic import BaseModel, Field
from typing import Union, Literal

from graphai.api.schemas.common import TaskStatusResponse


class EmbeddingRequest(BaseModel):
    text: str = Field(
        title="Text",
        description="Text to embed."
    )

    model_type: Literal['light'] = Field(
        title="Model type",
        description="Language of the provided text",
        default='light'
    )

    force: bool = Field(
        title="Force recomputation",
        default=False
    )


class EmbeddingTaskResponse(BaseModel):
    result: Union[str, None] = Field(
        None,
        title="Embedding results",
        description="Embedding text"
    )

    successful: bool = Field(
        title="Success flag",
        description="Whether or not the embedding was successful"
    )

    fresh: bool = Field(
        title="Freshness flag",
        description="Whether or not the result is fresh"
    )

    device: Union[str, None] = Field(
        None,
        title="Device",
        description="The device used ('cuda' or 'cpu') for the embedding. `None` in case of cache hit or failure."
    )


class EmbeddingResponse(TaskStatusResponse):
    task_result: Union[EmbeddingTaskResponse, None] = Field(
        title="Embedding response",
        description="A dict containing the resulting embedding of the original text and a success flag."
    )
