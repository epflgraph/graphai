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


class WikifyResult(BaseModel):
    """
    Object representing each of the wikify results. It consists of a set of keywords, a Wikipedia page and several
    scores which measure the degree of relevance of the result with respect to the text.
    """
    keywords: str = Field(
        ...,
        title="Keywords",
        description="Keywords extracted from the text associated to this result"
    )

    page_id: int = Field(
        ...,
        title="Page ID",
        description="ID of the Wiki page"
    )

    page_title: str = Field(
        ...,
        title="Page Title",
        description="Title of the Wiki page"
    )

    page_title_0: str = Field(
        ...,
        title="Page Title 0",
        description="Title of the Wiki page 0"
    )

    searchrank: int = Field(
        ...,
        title="Searchrank",
        description="Position in which this Wiki page appears in the list of search results when searching for the keywords"
    )

    median_graph_score: float = Field(
        ...,
        title="Median graph score",
        description="Median of the graph scores obtained by this Wiki page with respect to each of the provided anchor pages"
    )

    search_graph_ratio: float = Field(
        ...,
        title="Search-graph ratio",
        description="Ratio between the search and graph scores"
    )

    levenshtein_score: float = Field(
        ...,
        title="Levenshtein score",
        description="Levenshtein score of this page's title with respect to the keywords"
    )

    mixed_score: float = Field(
        ...,
        title="Mixed score",
        description="Combination of all scores"
    )

    class Config:
        schema_extra = {
            'examples': [
                {
                    "keywords": "angles",
                    "page_id": 1196,
                    "page_title": "Angle",
                    "searchrank": 2,
                    "median_graph_score": 0.5830005813291462,
                    "search_graph_ratio": 0.2915002906645731,
                    "levenshtein_score": 0.9090909090909091,
                    "mixed_score": 0.28085506517034586
                },
                {
                    "keywords": "angles",
                    "page_id": 76956,
                    "page_title": "Right_angle",
                    "searchrank": 4,
                    "median_graph_score": 0.48551476089734247,
                    "search_graph_ratio": 0.12137869022433562,
                    "levenshtein_score": 0.5882352941176471,
                    "mixed_score": 0.08126190097848535
                },
                {
                    "keywords": "angles equal",
                    "page_id": 133496,
                    "page_title": "Parallelogram",
                    "searchrank": 1,
                    "median_graph_score": 0.394037640426413,
                    "search_graph_ratio": 0.394037640426413,
                    "levenshtein_score": 0.32,
                    "mixed_score": 0.07547607718181548
                },
                {
                    "keywords": "angles equal",
                    "page_id": 1196,
                    "page_title": "Angle",
                    "searchrank": 2,
                    "median_graph_score": 0.5830005813291462,
                    "search_graph_ratio": 0.2915002906645731,
                    "levenshtein_score": 0.5882352941176471,
                    "mixed_score": 0.19515672571028433
                },
                {
                    "keywords": "angles equal",
                    "page_id": 91111,
                    "page_title": "Angle_trisection",
                    "searchrank": 3,
                    "median_graph_score": 0.3476123143066503,
                    "search_graph_ratio": 0.1158707714355501,
                    "levenshtein_score": 0.5,
                    "mixed_score": 0.05793538571777505
                },
                {
                    "keywords": "finite straight line",
                    "page_id": 946975,
                    "page_title": "Line_(geometry)",
                    "searchrank": 1,
                    "median_graph_score": 0.5915461851792454,
                    "search_graph_ratio": 0.5915461851792454,
                    "levenshtein_score": 0.34285714285714286,
                    "mixed_score": 0.13100744170621825
                },
                {
                    "keywords": "finite straight line",
                    "page_id": 22634860,
                    "page_title": "Line_segment",
                    "searchrank": 2,
                    "median_graph_score": 0.5379522375359874,
                    "search_graph_ratio": 0.2689761187679937,
                    "levenshtein_score": 0.4375,
                    "mixed_score": 0.10154942377039773
                },
                {
                    "keywords": "finite straight line",
                    "page_id": 9417,
                    "page_title": "Euclidean_geometry",
                    "searchrank": 6,
                    "median_graph_score": 0.6011022594685048,
                    "search_graph_ratio": 0.10018370991141747,
                    "levenshtein_score": 0.2631578947368421,
                    "mixed_score": 0.013094543912737368
                },
                {
                    "keywords": "interior angles",
                    "page_id": 13295107,
                    "page_title": "Transversal_(geometry)",
                    "searchrank": 2,
                    "median_graph_score": 0.3276581414375417,
                    "search_graph_ratio": 0.16382907071877084,
                    "levenshtein_score": 0.32432432432432434,
                    "mixed_score": 0.0322677338136873
                },
                {
                    "keywords": "interior angles",
                    "page_id": 1196,
                    "page_title": "Angle",
                    "searchrank": 3,
                    "median_graph_score": 0.5830005813291462,
                    "search_graph_ratio": 0.1943335271097154,
                    "levenshtein_score": 0.5,
                    "mixed_score": 0.0971667635548577
                },
                {
                    "keywords": "interior angles",
                    "page_id": 30654,
                    "page_title": "Triangle",
                    "searchrank": 4,
                    "median_graph_score": 0.5166070436540483,
                    "search_graph_ratio": 0.12915176091351208,
                    "levenshtein_score": 0.6956521739130435,
                    "mixed_score": 0.10682151707307275
                },
                {
                    "keywords": "straight line",
                    "page_id": 946975,
                    "page_title": "Line_(geometry)",
                    "searchrank": 1,
                    "median_graph_score": 0.5915461851792454,
                    "search_graph_ratio": 0.5915461851792454,
                    "levenshtein_score": 0.2857142857142857,
                    "mixed_score": 0.0902750735129564
                },
                {
                    "keywords": "straight line",
                    "page_id": 22634860,
                    "page_title": "Line_segment",
                    "searchrank": 3,
                    "median_graph_score": 0.5379522375359874,
                    "search_graph_ratio": 0.1793174125119958,
                    "levenshtein_score": 0.32,
                    "mixed_score": 0.03434741628275069
                },
                {
                    "keywords": "straight line falling",
                    "page_id": 20110824,
                    "page_title": "Infinity",
                    "searchrank": 2,
                    "median_graph_score": 0.5035806444244773,
                    "search_graph_ratio": 0.2517903222122386,
                    "levenshtein_score": 0.3448275862068966,
                    "mixed_score": 0.056450380072210773
                },
                {
                    "keywords": "straight line falling",
                    "page_id": 33731493,
                    "page_title": "Parallel_postulate",
                    "searchrank": 3,
                    "median_graph_score": 0.5147619767637447,
                    "search_graph_ratio": 0.1715873255879149,
                    "levenshtein_score": 0.3076923076923077,
                    "mixed_score": 0.030329609766357013
                },
                {
                    "keywords": "straight line falling",
                    "page_id": 928,
                    "page_title": "Axiom",
                    "searchrank": 4,
                    "median_graph_score": 0.5594457017244829,
                    "search_graph_ratio": 0.13986142543112073,
                    "levenshtein_score": 0.15384615384615385,
                    "mixed_score": 0.008253182503235177
                },
                {
                    "keywords": "straight line falling",
                    "page_id": 9417,
                    "page_title": "Euclidean_geometry",
                    "searchrank": 5,
                    "median_graph_score": 0.6011022594685048,
                    "search_graph_ratio": 0.12022045189370097,
                    "levenshtein_score": 0.3076923076923077,
                    "mixed_score": 0.0212500508378333
                },
                {
                    "keywords": "straight lines",
                    "page_id": 946975,
                    "page_title": "Line_(geometry)",
                    "searchrank": 1,
                    "median_graph_score": 0.5915461851792454,
                    "search_graph_ratio": 0.5915461851792454,
                    "levenshtein_score": 0.27586206896551724,
                    "mixed_score": 0.08440929451917713
                },
                {
                    "keywords": "straight lines makes",
                    "page_id": 20110824,
                    "page_title": "Infinity",
                    "searchrank": 2,
                    "median_graph_score": 0.5035806444244773,
                    "search_graph_ratio": 0.2517903222122386,
                    "levenshtein_score": 0.21428571428571427,
                    "mixed_score": 0.023243527253928767
                },
                {
                    "keywords": "straight lines makes",
                    "page_id": 946975,
                    "page_title": "Line_(geometry)",
                    "searchrank": 3,
                    "median_graph_score": 0.5915461851792454,
                    "search_graph_ratio": 0.19718206172641511,
                    "levenshtein_score": 0.4,
                    "mixed_score": 0.061131470999058996
                },
                {
                    "keywords": "straight lines makes",
                    "page_id": 33731493,
                    "page_title": "Parallel_postulate",
                    "searchrank": 5,
                    "median_graph_score": 0.5147619767637447,
                    "search_graph_ratio": 0.10295239535274894,
                    "levenshtein_score": 0.3684210526315789,
                    "mixed_score": 0.026635852549379505
                }
            ]
        }
