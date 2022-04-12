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
