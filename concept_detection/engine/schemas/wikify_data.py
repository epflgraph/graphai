from pydantic import BaseModel, Field
from typing import List, Optional


class WikifyData(BaseModel):
    """
    Object containing the raw text to be wikified and the list of anchor page ids to define the search space in the graph.
    """
    raw_text: str = Field(
        ...,
        title="Raw Text",
        description="Raw text to be wikified",
        example="To draw a straight line from any point to any point.\nTo produce a finite straight line continuously in a straight line.\nTo describe a circle with any center and radius.\nThat all right angles equal one another.\nThat, if a straight line falling on two straight lines makes the interior angles on the same side less than two right angles, the two straight lines, if produced indefinitely, meet on that side on which are the angles less than the two right angles."
    )
    anchor_page_ids: Optional[List[int]] = Field(
        [],
        title="Anchor Page IDs",
        description="List of IDs of Wikipedia pages definining the search space in the graph",
        example=[18973446, 9417, 946975]
    )


class WikifyDataKeywords(BaseModel):
    """
    Object containing the keywords to be wikified and the list of anchor page ids to define the search space in the graph.
    """
    keyword_list: List[str] = Field(
        ...,
        title="Keyword List",
        description="Keywords to be wikified",
        example=["straight line", "describe a circle", "right angles", "straight line falling", "interior angles"]
    )
    anchor_page_ids: Optional[List[int]] = Field(
        [],
        title="Anchor Page IDs",
        description="List of IDs of Wikipedia pages definining the search space in the graph",
        example=[18973446, 9417, 946975]
    )
