from pydantic import BaseModel, Field


class ScoresResult(BaseModel):
    """
    Object containing the graph score for a combination of source and target Wikipedia pages.
    """
    source_page_id: int = Field(
        ...,
        title="Source Page ID",
        description="IDs of the source Wikipedia page",
        example=1
    )

    target_page_id: int = Field(
        ...,
        title="Target Page ID",
        description="IDs of the target Wikipedia page",
        example=2
    )

    score: float = Field(
        ...,
        title="Score",
        description="Graph score associated to the source and target Wikipedia pages",
        example=0.21348491328739575
    )
