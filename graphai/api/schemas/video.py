from pydantic import BaseModel, Field


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
