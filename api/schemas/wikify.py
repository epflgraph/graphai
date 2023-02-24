from pydantic import BaseModel, Field, root_validator
from typing import List, Optional


class KeywordsRequest(BaseModel):
    """
    Object containing the raw text to extract keywords.
    """
    raw_text: str = Field(
        ...,
        title="Raw Text",
        description="Raw text to be wikified",
        example="To draw a straight line from any point to any point.\nTo produce a finite straight line continuously in a straight line.\nTo describe a circle with any center and radius.\nThat all right angles equal one another.\nThat, if a straight line falling on two straight lines makes the interior angles on the same side less than two right angles, the two straight lines, if produced indefinitely, meet on that side on which are the angles less than the two right angles."
    )


class WikifyRequest(BaseModel):
    """
    Object containing the information to be wikified and the list of anchor page ids to define the search space in the graph.
    """
    raw_text: Optional[str] = Field(
        None,
        title="Raw Text",
        description="Raw text to be wikified",
        example="To draw a straight line from any point to any point.\nTo produce a finite straight line continuously in a straight line.\nTo describe a circle with any center and radius.\nThat all right angles equal one another.\nThat, if a straight line falling on two straight lines makes the interior angles on the same side less than two right angles, the two straight lines, if produced indefinitely, meet on that side on which are the angles less than the two right angles."
    )

    keyword_list: Optional[List[str]] = Field(
        [],
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

    @root_validator
    def has_input(cls, values):
        if 'raw_text' not in values and 'keyword_list' not in values:
            raise ValueError('At least one of {raw_text, keyword_list} must be provided.')
        return values


class WikifyResponseElem(BaseModel):
    """
    Object representing each of the wikify results. It consists of a set of keywords, a Wikipedia page and several
    scores which measure the degree of relevance of the result with respect to the text.
    """

    PageID: int = Field(
        ...,
        title="Page ID",
        description="ID of the Wiki page"
    )

    PageTitle: str = Field(
        ...,
        title="Page Title",
        description="Title of the Wiki page"
    )

    SearchScore: float = Field(
        ...,
        title="Search score",
        description="Score that measures the relevance of the given page for the given keywords"
    )

    LevenshteinScore: float = Field(
        ...,
        title="Levenshtein score",
        description="Score that measures the Levenshtein similarity of the given page title and the given set of keywords"
    )

    OntologyLocalScore: float = Field(
        ...,
        title="OntologyLocalScore",
        description="Score that measures the relevance of the given page with respect to the rest according to the ontology"
    )

    OntologyGlobalScore: float = Field(
        ...,
        title="OntologyGlobalScore",
        description="Score that measures the relevance of the given page with respect to the rest according to the ontology"
    )

    GraphScore: float = Field(
        ...,
        title="Graph score",
        description="Score that measures the relevance of the given page with respect to the rest according to the graph"
    )

    KeywordsScore: float = Field(
        ...,
        title="KeywordsScore",
        description="Score that measures the relevance of the given page with respect to the rest according to the graph"
    )

    MixedScore: float = Field(
        ...,
        title="Mixed score",
        description="Score that aggregates the other scores"
    )

    class Config:
        schema_extra = {
            'examples': [],
            'examples': [
                {
                    "PageID": 1196,
                    "PageTitle": "Angle",
                    "SearchScore": 0.8055555555555556,
                    "LevenshteinScore": 1,
                    "OntologyLocalScore": 1,
                    "OntologyGlobalScore": 1,
                    "GraphScore": 0.44735199351890065,
                    "KeywordsScore": 1,
                    "MixedScore": 0.9058463104630012
                },
                {
                    "PageID": 152547,
                    "PageTitle": "Bisection",
                    "SearchScore": 0.3333333333333333,
                    "LevenshteinScore": 0.16661245633224378,
                    "OntologyLocalScore": 0.6666666666666667,
                    "OntologyGlobalScore": 0.5,
                    "GraphScore": 0.22717425120724127,
                    "KeywordsScore": 0.609271523178808,
                    "MixedScore": 0.44715741719086977
                },
                {
                    "PageID": 23972889,
                    "PageTitle": "Distance_from_a_point_to_a_line",
                    "SearchScore": 0.16666666666666669,
                    "LevenshteinScore": 0.2134432477817427,
                    "OntologyLocalScore": 0.21885521885521886,
                    "OntologyGlobalScore": 0.19791666666666666,
                    "GraphScore": 0.003590346636521923,
                    "KeywordsScore": 0.19828593689131283,
                    "MixedScore": 0.17781458572659029
                },
                {
                    "PageID": 946975,
                    "PageTitle": "Line_(geometry)",
                    "SearchScore": 1,
                    "LevenshteinScore": 0.33567080901262164,
                    "OntologyLocalScore": 0.8232323232323234,
                    "OntologyGlobalScore": 0.7916666666666666,
                    "GraphScore": 1,
                    "KeywordsScore": 0.7855473315153876,
                    "MixedScore": 0.7886663359580247
                },
                {
                    "PageID": 22634860,
                    "PageTitle": "Line_segment",
                    "SearchScore": 0.5555555555555556,
                    "LevenshteinScore": 0.3582111602475977,
                    "OntologyLocalScore": 0.43097643097643096,
                    "OntologyGlobalScore": 0.75,
                    "GraphScore": 0.42461810549112244,
                    "KeywordsScore": 0.6026490066225165,
                    "MixedScore": 0.5277457623305826
                },
                {
                    "PageID": 2175469,
                    "PageTitle": "Non-line-of-sight_propagation",
                    "SearchScore": 0.19444444444444448,
                    "LevenshteinScore": 0.39019342923035233,
                    "OntologyLocalScore": 0.21212121212121213,
                    "OntologyGlobalScore": 0.11647727272727273,
                    "GraphScore": 0.01572170017309725,
                    "KeywordsScore": 0.5087650954421504,
                    "MixedScore": 0.2950855110143057
                },
                {
                    "PageID": 133496,
                    "PageTitle": "Parallelogram",
                    "SearchScore": 0.3055555555555556,
                    "LevenshteinScore": 0.13627674782665528,
                    "OntologyLocalScore": 0.6666666666666667,
                    "OntologyGlobalScore": 0.5,
                    "GraphScore": 0.14484503981375635,
                    "KeywordsScore": 0.609271523178808,
                    "MixedScore": 0.4288185842201275
                },
                {
                    "PageID": 76944,
                    "PageTitle": "Perpendicular",
                    "SearchScore": 0.19444444444444445,
                    "LevenshteinScore": 0.039652736888335365,
                    "OntologyLocalScore": 0.21212121212121213,
                    "OntologyGlobalScore": 0.25,
                    "GraphScore": 0.15017752515904018,
                    "KeywordsScore": 0.2584729255940787,
                    "MixedScore": 0.19421461143444865
                },
                {
                    "PageID": 25278,
                    "PageTitle": "Quadrilateral",
                    "SearchScore": 0.02777777777777779,
                    "LevenshteinScore": 0.15367913840285474,
                    "OntologyLocalScore": 0.36363636363636365,
                    "OntologyGlobalScore": 0.25,
                    "GraphScore": 0.12360559235107477,
                    "KeywordsScore": 0.3307362680171406,
                    "MixedScore": 0.21973432050168798
                },
                {
                    "PageID": 76956,
                    "PageTitle": "Right_angle",
                    "SearchScore": 0.4166666666666667,
                    "LevenshteinScore": 0.4451948682556247,
                    "OntologyLocalScore": 0.42424242424242425,
                    "OntologyGlobalScore": 0.5,
                    "GraphScore": 0.12678424818870415,
                    "KeywordsScore": 0.4107908063887807,
                    "MixedScore": 0.3996645939435453
                },
                {
                    "PageID": 94102,
                    "PageTitle": "Solid_angle",
                    "SearchScore": 0.3888888888888889,
                    "LevenshteinScore": 0.6912823774628103,
                    "OntologyLocalScore": 0.196969696969697,
                    "OntologyGlobalScore": 0.15909090909090912,
                    "GraphScore": 0.08695255651117278,
                    "KeywordsScore": 0.7415270744059212,
                    "MixedScore": 0.4580780578246384
                },
                {
                    "PageID": 31482,
                    "PageTitle": "Tangent",
                    "SearchScore": 0.4722222222222222,
                    "LevenshteinScore": 0.29631455223594527,
                    "OntologyLocalScore": 0.6414141414141414,
                    "OntologyGlobalScore": 0.59375,
                    "GraphScore": 0.3573397570845174,
                    "KeywordsScore": 0.5087650954421504,
                    "MixedScore": 0.48284225283305426
                },
                {
                    "PageID": 30654,
                    "PageTitle": "Triangle",
                    "SearchScore": 0.16666666666666669,
                    "LevenshteinScore": 0.5717479784842091,
                    "OntologyLocalScore": 0.6666666666666667,
                    "OntologyGlobalScore": 0.5,
                    "GraphScore": 0.2535403349997643,
                    "KeywordsScore": 0.609271523178808,
                    "MixedScore": 0.47723102055958355
                }
            ]
        }


WikifyResponse = List[WikifyResponseElem]
