from pydantic import BaseModel, Field, Json
from typing import Union, Any

from graphai.api.schemas.common import TaskStatusResponse


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

    closest_token: Union[str, None] = Field(
        title="Closest token",
        description="The token of the most similar existing audio that the fingerprint lookup was able to find. Equal "
                    "to original token if the most similar existing audio did not satisfy the minimum similarity "
                    "threshold."
    )

    closest_token_origin: Union[str, None] = Field(
        title="Original token of the closest token",
        description="The token of video that the closest audio token originated from."
    )

    duration: float = Field(
        title="Duration of audio file",
        description="Length of audio. If the remove_silence flag was set, this is the length of the audio "
                    "with beginning and ending silence removed. This value is an approximation "
                    "based on the audio file's bitrate."
    )

    fp_nosilence: bool = Field(
        title="Fingerprint with no beginning/end silence"
    )

    successful: bool = Field(
        title="Success flag",
        description="Whether the computation was successful."
    )


class AudioFingerprintResponse(TaskStatusResponse):
    task_result: Union[AudioFingerprintTaskResponse, None] = Field(
        title="Audio fingerprinting response",
        description="A dict containing the resulting audio fingerprint, a freshness flag, and no-silence audio length."
    )


class AudioDetectLanguageRequest(BaseModel):
    token: str = Field(
        title="Token",
        description="The token that identifies the requested file"
    )

    force: bool = Field(
        title="Force",
        description="Whether to force a recomputation",
        default=False
    )


class AudioDetectLanguageTaskResponse(BaseModel):
    language: Union[str, None] = Field(
        title="Language",
        description="Language of the audio file."
    )

    fresh: bool = Field(
        title="Freshness flag",
        description="Whether the result was computed freshly or an existing cached result was returned."
    )


class AudioDetectLanguageResponse(TaskStatusResponse):
    task_result: Union[AudioDetectLanguageTaskResponse, None] = Field(
        title="Audio transcription response",
        description="A dict containing the resulting audio language and a freshness flag."
    )


class AudioTranscriptionRequest(BaseModel):
    token: str = Field(
        title="Token",
        description="The token that identifies the requested file"
    )

    force: bool = Field(
        title="Force",
        description="Whether to force a recomputation",
        default=False
    )

    force_lang: str = Field(
        title="Force language",
        description="If set, it tells Whisper to transcribe the audio into the indicated language. Language detection "
                    "is automatic otherwise.",
        default=None
    )


class AudioTranscriptionTaskResponse(BaseModel):
    transcript_results: Union[str, None] = Field(
        title="Transcript",
        description="The transcript of the requested audio file"
    )

    subtitle_results: Union[Json[Any], None] = Field(
        title="Subtitles",
        description="Timestamped transcript of the requested audio file."
    )

    language: Union[str, None] = Field(
        title="Language",
        description="Language of the audio file."
    )

    fresh: bool = Field(
        title="Freshness flag",
        description="Whether the result was computed freshly or an existing cached result was returned."
    )


class AudioTranscriptionResponse(TaskStatusResponse):
    task_result: Union[AudioTranscriptionTaskResponse, None] = Field(
        title="Audio transcription response",
        description="A dict containing the resulting audio transcript, its subtitles (timesstamped transcript), "
                    "the language of the audio, and a freshness flag."
    )
