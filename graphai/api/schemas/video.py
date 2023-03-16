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


class RetrieveURLResponse(TaskStatusResponse):
    task_result: Union[Dict[str, Union[str, bool, None]], None] = Field(
        title="File retrieval response",
        description="A dict containing a flag for whether the retrieval was successful, plus a token "
                    "that refers to the now-retrieved file if so (and null if not)."
    )


class ComputeSignatureRequest(BaseModel):
    token: str = Field(
        title="Token",
        description="The token that identifies the requested file"
    )


class ComputeSignatureResponse(TaskStatusResponse):
    task_result: Union[Dict[str, Union[str, bool, None]], None] = Field(
        title="Calculate fingerprint response",
        description="A dict containing a flag for whether the fingerprint calculation was successful, plus a token "
                    "that refers to the now-retrieved file if so (and null if not)."
    )


class FileRequest(BaseModel):
    filename: str = Field(
        title="File name",
        description="The name of the file to be downloaded (received as a response from another endpoint)."
    )