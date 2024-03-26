from pydantic import BaseModel, Field
from typing import List


class SlideConceptsMap(BaseModel):
    number: int = Field(
        title="Slide Number"
    )

    concepts: List[str] = Field(
        title="Concepts",
        description="List of concepts for this slide"
    )


class SlideSubsetRequest(BaseModel):
    slides: List[SlideConceptsMap] = Field(
        title="Slides list",
        description="A list of dictionaries with two fields: an integer 'number' that indicates the slide number, "
                    "and 'concepts', the list containing the slide's concepts."
    )

    coverage: float = Field(
        title="Coverage",
        description="What proportion of the concepts to cover.",
        default=1.0
    )

    min_freq: int = Field(
        title="Minimum frequency",
        description="Minimum number of occurrences a concept must have in all the slides combined in order not to "
                    "be removed as noise.",
        default=2
    )


class SlideSubsetResponse(BaseModel):
    subset: List[int] = Field(
        title="Optimal subset",
        description="The Slide Numbers of the slides that were chosen as part of the "
                    "optimal set cover for the concepts."
    )
