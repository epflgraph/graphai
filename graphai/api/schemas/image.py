from pydantic import BaseModel, Field
from typing import Union
from .common import TaskStatusResponse

class ImageFingerprintRequest(BaseModel):
    token: str = Field(
        title="Token",
        description="The token that identifies the requested file"
    )

    force: bool = Field(
        title="Force",
        description="Whether to force a recomputation",
        default=False
    )


class ImageFingerprintTaskResponse(BaseModel):
    result: Union[str, None] = Field(
        title="Fingerprint",
        description="Fingerprint of the image file."
    )

    fresh: bool = Field(
        title="Freshness flag",
        description="Whether the result was computed freshly or an existing cached result was returned."
    )

    successful: bool = Field(
        title="Success flag",
        description="Whether the computation was successful."
    )


class ImageFingerprintResponse(TaskStatusResponse):
    task_result: Union[ImageFingerprintTaskResponse, None] = Field(
        title="Audio transcription response",
        description="A dict containing the resulting image fingerprint and a freshness flag."
    )