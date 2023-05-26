from pydantic import BaseModel, Field
from typing import Union, Literal, Dict

from graphai.api.schemas.common import TaskStatusResponse


class RetrieveURLRequest(BaseModel):
    url: str = Field(
        title="URL",
        description="The URL to be retrieved and stored for further processing"
    )

    kaltura: bool = Field(
        title="Kaltura url",
        description="A boolean variable indicating whether the provided URL is from Kaltura. True by default.",
        default=True
    )

    timeout: int = Field(
        title="Kaltura timeout",
        description="Timeout in seconds for Kaltura URLs, default 120s. Cannot be more than 480s.",
        default=240
    )


class RetrieveURLResponseInner(BaseModel):
    token: Union[str, None] = Field(
        title="Token",
        description="Result token, null if task has failed"
    )

    fresh: bool = Field(
        title="Freshness flag",
        description="False if the URI had already been retrieved before, True if not and if retrieval was successful"
    )

    successful: bool = Field(
        title="Success flag",
        description="True if task successful, False otherwise"
    )


class RetrieveURLResponse(TaskStatusResponse):
    task_result: Union[RetrieveURLResponseInner, None] = Field(
        title="File retrieval response",
        description="A dict containing a flag for whether the retrieval was successful, plus a token "
                    "that refers to the now-retrieved file if so (and null if not)."
    )


class VideoFingerprintRequest(BaseModel):
    token: str = Field(
        title="Token",
        description="Token of video to fingerprint"
    )

    force: bool = Field(
        title="Force recomputation",
        default=False
    )


class VideoFingerprintTaskResponse(BaseModel):
    result: Union[str, None] = Field(
        title="Fingerprint",
        description="Fingerprint of the provided video."
    )

    fresh: bool = Field(
        title="Freshness flag",
        description="Whether the result was computed freshly or an existing cached result was returned."
    )

    closest_token: Union[str, None] = Field(
        title="Closest token",
        description="The token of the most similar existing video that the fingerprint lookup was able to find. Equal "
                    "to original token if the most similar existing video did not satisfy the minimum similarity "
                    "threshold."
    )

    successful: bool = Field(
        title="Success flag",
        description="Whether the computation was successful."
    )


class VideoFingerprintResponse(TaskStatusResponse):
    task_result: Union[VideoFingerprintTaskResponse, None] = Field(
        title="Video fingerprinting response",
        description="A dict containing the resulting video fingerprint and a freshness flag."
    )


class ExtractAudioRequest(BaseModel):
    token: str = Field(
        title="Token",
        description="The token that identifies the requested file"
    )

    force: bool = Field(
        title="Force",
        description="Whether to force a recomputation",
        default=False
    )


class ExtractAudioTaskResponse(BaseModel):
    token: Union[str, None] = Field(
        title="Token",
        description="Result token, null if task has failed"
    )
    successful: bool = Field(
        title="Success flag",
        description="True if task successful, False otherwise"
    )
    fresh: bool = Field(
        title="Freshness flag",
        description="Whether the result was computed freshly or an existing cached result was returned."
    )
    duration: float = Field(
        title="Audio duration",
        description="Duration of audio based on the length of its video file. This value is exact as it is based "
                    "on video metadata."
    )


class ExtractAudioResponse(TaskStatusResponse):
    task_result: Union[ExtractAudioTaskResponse, None] = Field(
        title="Extract audio response",
        description="A dict containing a flag for whether the audio extraction was successful, a freshness flag, "
                    "a token that refers to the now-computed file if so (and null if not), and the duration of "
                    "the audio file."
    )


class DetectSlidesRequest(BaseModel):
    token: str = Field(
        title="Token",
        description="The token that identifies the requested file"
    )

    force: bool = Field(
        title="Force",
        description="Whether to force a recomputation",
        default=False
    )

    language: Literal['en', 'fr'] = Field(
        title="Language",
        description="Language of the video",
        default="en"
    )


class SlideTokenAndTimeStamp(BaseModel):
    token: str = Field(
        title="Slide token"
    )

    timestamp: int = Field(
        title="Slide timestamp"
    )


class DetectSlidesTaskResponse(BaseModel):
    slide_tokens: Union[Dict[int, SlideTokenAndTimeStamp], None] = Field(
        title="Slide tokens",
        description="Tokens of the detected slides and their timestamps"
    )

    successful: bool = Field(
        title="Success flag",
        description="True if task successful, False otherwise"
    )

    fresh: bool = Field(
        title="Freshness flag",
        description="Whether the result was computed freshly or an existing cached result was returned."
    )


class DetectSlidesResponse(TaskStatusResponse):
    task_result: Union[DetectSlidesTaskResponse, None] = Field(
        title="Detect slides response",
        description="A dict containing a flag for whether the slide detection was successful and a freshness flag, "
                    "a token that refers to the now-computed file if so (and null if not), "
                    "and the number of detected slides."
    )
