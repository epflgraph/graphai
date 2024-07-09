from pydantic import BaseModel, Field
from typing import Union, Literal

from graphai.api.schemas.common import TaskStatusResponse


class EmbeddingRequest(BaseModel):
    text: str = Field(
        title="Text",
        description="Text to embed."
    )

    model_type: Literal['all-MiniLM-L12-v2', 'Solon-embeddings-large-0.1'] = Field(
        title="Model type",
        description="Type of model to use",
        default='all-MiniLM-L12-v2'
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

    text_too_large: bool = Field(
        title="Text too large",
        description="This boolean flag is true if the text provided for embedding is too long "
                    "(depends on model, the limit for the default model is 128 tokens)."
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

    model_type: Literal['all-MiniLM-L12-v2'] = Field(
        title="Model type",
        description="Type of model that was used",
        default='all-MiniLM-L12-v2'
    )


class EmbeddingResponse(TaskStatusResponse):
    task_result: Union[EmbeddingTaskResponse, None] = Field(
        title="Embedding response",
        description="A dict containing the resulting embedding of the original text and a success flag."
    )
