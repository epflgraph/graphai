from pydantic import BaseModel, Field
from typing import List


class ScoresData(BaseModel):
    """
    Object containing the source and target page ids over which to compute the graph scores.
    """
    source_page_ids: List[int] = Field(
        ...,
        title="Source Page IDs",
        description="List of IDs of source Wikipedia pages",
        example=[1, 2, 3, 4]
    )
    target_page_ids: List[int] = Field(
        ...,
        title="Target Page IDs",
        description="List of IDs of target Wikipedia pages",
        example=[1, 2, 3, 4]
    )


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
