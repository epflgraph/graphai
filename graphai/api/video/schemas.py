from pydantic import BaseModel, Field
from typing import Union, Literal, Dict, List

from graphai.api.common.schemas import (
    TaskStatusResponse,
    TokenStatus
)


class RetrieveURLRequest(BaseModel):
    url: str = Field(
        title="URL",
        description="The URL to be retrieved and stored for further processing"
    )

    force: bool = Field(
        title="Force redownload",
        default=False
    )

    playlist: bool = Field(
        title="Playlist flag",
        description="A boolean variable indicating whether the provided URL is an m3u8 playlist, rather "
                    "than a video file (like an .mp4 file). Video URLs from Kaltura, for example, "
                    "are m3u8 playlists. False by default.",
        default=False
    )


class VideoStreamInfo(BaseModel):
    codec_type: Literal['video', 'audio'] = Field(
        title="Codec type",
        description="Allowed values are 'video' and 'audio'"
    )

    codec_name: Union[str, None] = Field(
        title="Codec name"
    )

    duration: Union[float, None] = Field(
        title="Duration"
    )

    bit_rate: Union[int, None] = Field(
        title="Bit rate"
    )

    sample_rate: Union[int, None] = Field(
        title="Sample rate"
    )

    resolution: Union[str, None] = Field(
        title="Resolution",
        description="For video streams only"
    )


class VideoTokenStatus(TokenStatus):
    streams: Union[List[VideoStreamInfo], None] = Field(
        default=None, title="Available streams",
        description="List of streams available in the video token, including their duration, "
                    "bitrate, resolution, codec type and name."
    )


class RetrieveURLResponseInner(BaseModel):
    token: Union[str, None] = Field(
        None, title="Token",
        description="Result token, null if task has failed"
    )

    token_status: Union[VideoTokenStatus, None] = Field(
        title="Token status",
        description="Status of the returned token",
        default=None
    )

    token_size: Union[int, None] = Field(
        title="Token size",
        description="Size of the returned token",
        default=None
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
        None, title="Fingerprint",
        description="Fingerprint of the provided video."
    )

    fresh: bool = Field(
        title="Freshness flag",
        description="Whether the result was computed freshly or an existing cached result was returned."
    )

    closest_token: Union[str, None] = Field(
        None, title="Closest token",
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

    recalculate_cached: bool = Field(
        title="Only perform a cache recalculation",
        description="If set to True, this flag OVERRIDES the 'force' flag, assumes that this video token is active and"
                    "that the audio for this video have been previously computed and cached, and recreates "
                    "the audio file. Will fail with null results if the video token is inactive, or if its audio "
                    "has not previously been computed and cached in the database.",
        default=False
    )

    force: bool = Field(
        title="Force recompute",
        description="Whether to force a recomputation under any circumstances. False by default.",
        default=False
    )


class ExtractAudioTaskResponse(BaseModel):
    token: Union[str, None] = Field(
        None, title="Token",
        description="Result token, null if task has failed"
    )

    token_status: Union[TokenStatus, None] = Field(
        title="Token status",
        description="Status of the returned token",
        default=None
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


class SlideDetectionParams(BaseModel):
    hash_thresh: float = Field(
        title="Hash threshold",
        description="The maximum hash-based similarity two images can have without being considered identical. "
                    "Value between 0 and 1. Higher values are more permissive.",
        default=0.95
    )

    multiplier: float = Field(
        title="OCR noise multiplier",
        description="The multiplier used in computing the OCR noise threshold. Differences below the threshold are "
                    "treated as nonexistent. Lower values are more permissive.",
        default=5.0
    )

    default_threshold: float = Field(
        title="OCR default noise threshold",
        description="The default value used for the OCR noise threshold when its value cannot be computed. "
                    "Differences below the threshold are treated as nonexistent. Lower values are more permissive.",
        default=0.05
    )

    include_first: bool = Field(
        title="Force-include first slide",
        default=True
    )

    include_last: bool = Field(
        title="Force-include last slide",
        default=True
    )


class DetectSlidesRequest(BaseModel):
    token: str = Field(
        title="Token",
        description="The token that identifies the requested file"
    )

    recalculate_cached: bool = Field(
        title="Only perform a cache recalculation",
        description="If set to True, this flag OVERRIDES the 'force' flag, assumes that this video token is active and"
                    "that the slides for this video have been previously computed and cached, "
                    "and recreates the slide files by extracting video frames and then only keeping the "
                    "timestamps indicated in the cache. Will fail with null results if the video token is "
                    "inactive, or if its slides have not previously been computed and cached in the database.",
        default=False
    )

    force: bool = Field(
        title="Force",
        description="Whether to force a recomputation. False by default.",
        default=False
    )

    language: Literal['en', 'fr'] = Field(
        title="Language",
        description="Language of the video",
        default="en"
    )

    parameters: SlideDetectionParams = Field(
        title="Parameters (advanced users only)",
        default=SlideDetectionParams(hash_thresh=0.95, multiplier=5, default_threshold=0.05,
                                     include_first=True, include_last=True)
    )


class SlideTokenAndTimeStamp(BaseModel):
    token: str = Field(
        title="Slide token"
    )

    token_status: Union[TokenStatus, None] = Field(
        title="Token status",
        description="Status of the returned token",
        default=None
    )

    timestamp: int = Field(
        title="Slide timestamp"
    )


class DetectSlidesTaskResponse(BaseModel):
    slide_tokens: Union[Dict[int, SlideTokenAndTimeStamp], None] = Field(
        None, title="Slide tokens",
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
