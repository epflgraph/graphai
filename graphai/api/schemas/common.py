from pydantic import BaseModel, Field

class TaskIDResponse(BaseModel):
    """
    Object representing the output of the /ontology/tree endpoint.
    """

    TaskID: str = Field(
        ...,
        title="Task ID",
        description="ID of the task created as a response to an API request"
    )

