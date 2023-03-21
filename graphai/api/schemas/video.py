from pydantic import BaseModel, Field
from typing import List, Dict, Union
from .common import TaskStatusResponse


class MultiprocessingExampleRequest(BaseModel):
    """
    Object representing the input of the /video/multiprocessing_example endpoint.
    """

    foo: int = Field(
        ...,
        title="Foo",
        description="First parameter"
    )

    bar: int = Field(
        ...,
        title="Bar",
        description="Second parameter"
    )


class MultiprocessingExampleResponse(BaseModel):
    """
    Object representing the output of the /video/multiprocessing_example endpoint.
    """

    baz: bool = Field(
        ...,
        title="Baz",
        description="Output parameter"
    )


class RetrieveURLRequest(BaseModel):
    url: str = Field(
        title="URL",
        description="The URL to be retrieved and stored for further processing"
    )


class RetrieveURLResponseInner(BaseModel):
    token: Union[str, None] = Field(
        title="Token",
        description="Result token, null if task has failed"
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


class PerformFileCachableComputationResponse(BaseModel):
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


class ComputeSignatureRequest(BaseModel):
    token: str = Field(
        title="Token",
        description="The token that identifies the requested file"
    )

    force: bool = Field(
        title="Force",
        description="Whether to force a recomputation",
        default=False
    )


class ComputeSignatureResponse(TaskStatusResponse):
    task_result: Union[PerformFileCachableComputationResponse, None] = Field(
        title="Calculate fingerprint response",
        description="A dict containing a flag for whether the fingerprint calculation was successful and "
                    "a freshness flag, plus a token that refers to the now-computed file if so (and null if not)."
    )


class FileRequest(BaseModel):
    token: str = Field(
        title="File name",
        description="The name of the file to be downloaded (received as a response from another endpoint)."
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


class ExtractAudioTaskResponse(PerformFileCachableComputationResponse):
    duration: float = Field(
        title="Audio duration",
        description="Duration of audio based on the length of its video file"
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


class DetectSlidesTaskResponse(PerformFileCachableComputationResponse):
    n_slides: int = Field(
        title="# of slides",
        description="Number of detected slides in given video file"
    )

    files: List[str] = Field(
        title="Names of slide files",
        description="The names of the slide files extracted from video"
    )


class DetectSlidesResponse(TaskStatusResponse):
    task_result: Union[DetectSlidesTaskResponse, None] = Field(
        title="Detect slides response",
        description="A dict containing a flag for whether the slide detection was successful and a freshness flag, "
                    "a token that refers to the now-computed file if so (and null if not), "
                    "and the number of detected slides."
    )