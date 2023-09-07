from pydantic import BaseModel, Field
from typing import Union, Literal, Dict, List

from graphai.api.schemas.common import TaskStatusResponse


class GetSublinksRequest(BaseModel):
    url: str = Field(
        title="URL",
        description="Text to summarize. Can be one string or a string to string dictionary."
    )

    force: bool = Field(
        title="Force recomputation",
        default=False
    )


class GetSublinksTaskResponse(BaseModel):
    token: Union[str, None] = Field(
        title="Token",
        description="The cleaned base URL which doubles as the token"
    )

    validated_url: Union[str, None] = Field(
        title="Validated URL",
        description="The validated base url"
    )

    sublinks: List[str] = Field(
        title="Sublinks",
        description="List of page sublinks"
    )

    successful: bool = Field(
        title="Success flag",
        description="Whether or not the sublink extraction was successful"
    )

    fresh: bool = Field(
        title="Freshness flag",
        description="Whether or not the result is fresh"
    )


class GetSublinksResponse(TaskStatusResponse):
    task_result: Union[GetSublinksTaskResponse, None] = Field(
        title="Sublink extraction response",
        description="A dict containing the resulting extracted sublinks and a success flag."
    )
