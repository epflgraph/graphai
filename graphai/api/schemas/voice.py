from pydantic import BaseModel, Field
from typing import List, Union
from .common import FileCachableComputationResponse, TaskStatusResponse, OngoingTaskResponse


class AudioFingerprintRequest(BaseModel):
    token: str = Field(
        title="Token",
        description="The token that identifies the requested file"
    )

    force: bool = Field(
        title="Force",
        description="Whether to force a recomputation",
        default=False
    )

    threshold: float = Field(
        title="Silence threshold",
        description="Threshold for silence removal at the beginning and end of the video",
        default=0.0
    )


class AudioFingerprintTaskResponse(BaseModel):
    result: Union[str, None] = Field(
        title="Fingerprint",
        description="The fingerprint of the requested audio file"
    )

    fresh: bool = Field(
        title="Freshness flag",
        description="Whether the result was computed freshly or an existing cached result was returned."
    )

    duration_nosilence: float = Field(
        title="Silence-less duration",
        description="Length of audio with beginning and ending silence removed. This value is an approximation "
                    "based on the audio file's bitrate."
    )


class AudioFingerprintResponse(TaskStatusResponse):
    task_result: Union[AudioFingerprintTaskResponse, OngoingTaskResponse, None] = Field(
        title="Audio fingerprinting response",
        description="A dict containing the resulting audio fingerprint, a freshness flag, and no-silence audio length."
    )