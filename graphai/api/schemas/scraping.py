from pydantic import BaseModel, Field
from typing import Union, Dict, List

from graphai.api.schemas.common import TaskStatusResponse


class GetSublinksRequest(BaseModel):
    url: str = Field(
        title="URL",
        description="URL to scrape."
    )

    force: bool = Field(
        title="Force recomputation",
        default=False
    )


class GetSublinksTaskResponse(BaseModel):
    token: Union[str, None] = Field(
        None, title="Token",
        description="The cleaned base URL which doubles as the token"
    )

    validated_url: Union[str, None] = Field(
        None, title="Validated URL",
        description="The validated base url"
    )

    sublinks: Union[List[str], None] = Field(
        None, title="Sublinks",
        description="List of page sublinks"
    )

    status_msg: str = Field(
        title="Status message",
        description="Message indicating the status of the response"
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


class ExtractContentRequest(GetSublinksRequest):
    remove_headers: bool = Field(
        title="Remove headers",
        description="Flag that determines whether headers are removed. False by default.",
        default=False
    )

    remove_long_patterns: bool = Field(
        title="Remove long patterns",
        description="Flag that determines whether long patterns are removed. False by default.",
        default=False
    )


class ExtractContentTaskResponse(GetSublinksTaskResponse):
    data: Union[Dict[str, Dict[str, Union[str, None]]], None] = Field(
        None, title="Extracted content",
        description="Dictionary mapping each sublink to content extracted from that sublink"
    )


class ExtractContentResponse(TaskStatusResponse):
    task_result: Union[ExtractContentTaskResponse, None] = Field(
        title="Content extraction response",
        description="A dict containing the resulting extracted content and a success flag."
    )
