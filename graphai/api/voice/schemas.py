from pydantic import BaseModel, Field, Json
from typing import Union, Any

from graphai.api.common.schemas import TaskStatusResponse


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


class AudioFingerprintTaskResponse(BaseModel):
    result: Union[str, None] = Field(
        None, title="Fingerprint",
        description="The fingerprint of the requested audio file"
    )

    fresh: bool = Field(
        title="Freshness flag",
        description="Whether the result was computed freshly or an existing cached result was returned."
    )

    closest_token: Union[str, None] = Field(
        None, title="Closest token",
        description="The token of the most similar existing audio that the fingerprint lookup was able to find. Equal "
                    "to original token if the most similar existing audio did not satisfy the minimum similarity "
                    "threshold."
    )

    closest_token_origin: Union[str, None] = Field(
        None, title="Original token of the closest token",
        description="The token of video that the closest audio token originated from."
    )

    duration: float = Field(
        title="Duration of audio file",
        description="Length of audio. If the remove_silence flag was set, this is the length of the audio "
                    "with beginning and ending silence removed. This value is an approximation "
                    "based on the audio file's bitrate."
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
        None, title="Language",
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

    strict: bool = Field(
        title="Strict silence detection",
        description="If set, reduces the threshold (from 0.6 to 0.5) for a segment being detected as silent. "
                    "If the 'no-speech probability' for a segment is above that threshold and the model's "
                    "confidence in its predicted text is low, the segment is treated as silent. ",
        default=False
    )


class AudioTranscriptionTaskResponse(BaseModel):
    transcript_results: Union[str, None] = Field(
        None, title="Transcript",
        description="The transcript of the requested audio file"
    )

    subtitle_results: Union[Json[Any], None] = Field(
        None, title="Subtitles",
        description="Timestamped transcript of the requested audio file."
    )

    language: Union[str, None] = Field(
        None, title="Language",
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
