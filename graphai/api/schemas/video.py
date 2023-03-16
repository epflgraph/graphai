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


class RetrieveFileRequest(BaseModel):
    url: str = Field(
        ...,
        title="URL",
        description="The URL to be retrieved and stored for further processing"
    )


class RetrieveFileResponse(TaskStatusResponse):
    task_result: Union[Dict[str, Union[str, bool, None]], None] = Field(
        title="File retrieval response",
        description="A dict containing a flag for whether or not the retrieval was successful, plus a token "
                    "that refers to the now-retrieved file if it was (and null if it wasn't)."
    )