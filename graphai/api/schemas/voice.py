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

    remove_silence: bool = Field(
        title="Remove silence",
        description="Boolean flag for choosing whether or not to remove silence from the beginning and the "
                    "end of the audio. False by default, setting it to True will reduce performance by a factor of ~2.",
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

    duration: float = Field(
        title="Duration of audio file",
        description="Length of audio. If the remove_silence flag was set, this is the length of the audio "
                    "with beginning and ending silence removed. This value is an approximation "
                    "based on the audio file's bitrate."
    )

    fp_nosilence: int = Field(
        title="Fingerprint with no beginning/end silence"
    )


class AudioFingerprintResponse(TaskStatusResponse):
    task_result: Union[AudioFingerprintTaskResponse, OngoingTaskResponse, None] = Field(
        title="Audio fingerprinting response",
        description="A dict containing the resulting audio fingerprint, a freshness flag, and no-silence audio length."
    )