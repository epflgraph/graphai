from pydantic import ConfigDict, BaseModel, Field
from typing import List


class WikifyFromRawTextRequest(BaseModel):
    """
    Object containing the raw text to be wikified.
    """
    raw_text: str = Field(
        ...,
        title="Raw Text",
        description="Raw text to be wikified",
        examples=["To draw a straight line from any point to any point.\nTo produce a finite straight line continuously in a straight line.\nTo describe a circle with any center and radius.\nThat all right angles equal one another.\nThat, if a straight line falling on two straight lines makes the interior angles on the same side less than two right angles, the two straight lines, if produced indefinitely, meet on that side on which are the angles less than the two right angles."]
    )


class WikifyFromKeywordsRequest(BaseModel):
    """
    Object containing the keywords to be wikified.
    """
    keywords: list[str] = Field(
        ...,
        title="Keywords",
        description="List of keywords to be wikified",
        examples=[["straight line", "point to point", "describe a circle", "all right angles equal", "two straight lines", "interior angles", "less than two right angles"]],
    )


class WikifyResponseElem(BaseModel):
    """
    Object representing each of the wikify results. It consists of a set of keywords, a Wikipedia page and several
    scores which measure the degree of relevance of the result with respect to the text.
    """

    concept_id: int = Field(
        ...,
        title="Concept ID",
        description="ID of the concept (wikipage)"
    )

    concept_name: str = Field(
        ...,
        title="Concept name",
        description="Name of the concept (wikipage)"
    )

    search_score: float = Field(
        ...,
        title="Search score",
        description="Score that measures the relevance of the given concept for the given set of keywords"
    )

    levenshtein_score: float = Field(
        ...,
        title="Levenshtein score",
        description="Score that measures the Levenshtein similarity of the given concept name and the given set of keywords"
    )

    graph_score: float = Field(
        ...,
        title="Graph score",
        description="Score that measures the relevance of the given concept with respect to all the other concepts according to the graph"
    )

    ontology_local_score: float = Field(
        ...,
        title="Ontology local score",
        description="Score that measures the relevance of the given concept with respect to the other concepts found for the given set of keywords according to the ontology"
    )

    ontology_global_score: float = Field(
        ...,
        title="Ontology global score",
        description="Score that measures the relevance of the given concept with respect to all the other concepts according to the ontology"
    )

    keywords_score: float = Field(
        ...,
        title="Keywords score",
        description="Score that measures the relevance of the sets of keywords for which the given concept is found according to the ontology"
    )

    mixed_score: float = Field(
        ...,
        title="Mixed score",
        description="Score that aggregates the other scores as a weighted average"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "concept_id": 1196,
                    "concept_name": "Angle",
                    "search_score": 0.9821428571428572,
                    "levenshtein_score": 0.995049504950495,
                    "graph_score": 0.8233916857660931,
                    "ontology_local_score": 1,
                    "ontology_global_score": 1,
                    "keywords_score": 1,
                    "mixed_score": 0.978025165747755,
                },
                {
                    "concept_id": 12401488,
                    "concept_name": "Triangle center",
                    "search_score": 0.3392857142857143,
                    "levenshtein_score": 0.4680631523928115,
                    "graph_score": 0.31985952600915474,
                    "ontology_local_score": 0.6666666666666666,
                    "ontology_global_score": 0.6666666666666666,
                    "keywords_score": 0.15980113636363635,
                    "mixed_score": 0.3846595758927376,
                },
                {
                    "concept_id": 13295107,
                    "concept_name": "Transversal (geometry)",
                    "search_score": 0.7178571428571429,
                    "levenshtein_score": 0.2110192540065071,
                    "graph_score": 0.07350115135142847,
                    "ontology_local_score": 0.8333333333333333,
                    "ontology_global_score": 0.8333333333333333,
                    "keywords_score": 0.8636363636363636,
                    "mixed_score": 0.6499986742317898,
                },
                {
                    "concept_id": 152547,
                    "concept_name": "Bisection",
                    "search_score": 0.27142857142857146,
                    "levenshtein_score": 0.2470772552749216,
                    "graph_score": 0.4148654292351757,
                    "ontology_local_score": 0.6666666666666666,
                    "ontology_global_score": 0.6666666666666666,
                    "keywords_score": 0.6818181818181819,
                    "mixed_score": 0.5040459667125913,
                },
                {
                    "concept_id": 161243,
                    "concept_name": "Nine-point circle",
                    "search_score": 0.5428571428571429,
                    "levenshtein_score": 0.3974121105221985,
                    "graph_score": 0.26965758333464235,
                    "ontology_local_score": 0.6666666666666666,
                    "ontology_global_score": 0.6666666666666666,
                    "keywords_score": 0.5859375,
                    "mixed_score": 0.5375969201498892,
                },
                {
                    "concept_id": 165487,
                    "concept_name": "World line",
                    "search_score": 0.27142857142857146,
                    "levenshtein_score": 0.20854298868986648,
                    "graph_score": 0.01689901923028631,
                    "ontology_local_score": 0.6666666666666666,
                    "ontology_global_score": 0.6666666666666666,
                    "keywords_score": 0.29829545454545453,
                    "mixed_score": 0.3434123675425259,
                },
                {
                    "concept_id": 1780815,
                    "concept_name": "Radius",
                    "search_score": 0.6785714285714286,
                    "levenshtein_score": 0.7313486756137682,
                    "graph_score": 0.3788549219571284,
                    "ontology_local_score": 0.6666666666666666,
                    "ontology_global_score": 0.6666666666666666,
                    "keywords_score": 0.15980113636363635,
                    "mixed_score": 0.4979090868278213,
                },
                {
                    "concept_id": 1896705,
                    "concept_name": "Osculating circle",
                    "search_score": 0.06785714285714284,
                    "levenshtein_score": 0.3974121105221985,
                    "graph_score": 0.14000281682995766,
                    "ontology_local_score": 0.6666666666666666,
                    "ontology_global_score": 0.6666666666666666,
                    "keywords_score": 0.5859375,
                    "mixed_score": 0.42963144349942073,
                },
                {
                    "concept_id": 1898401,
                    "concept_name": "Arc length",
                    "search_score": 0.5428571428571429,
                    "levenshtein_score": 0.09530697673156802,
                    "graph_score": 0.3465412719755263,
                    "ontology_local_score": 0.6666666666666666,
                    "ontology_global_score": 0.6666666666666666,
                    "keywords_score": 0.41548295454545453,
                    "mixed_score": 0.4488331553090194,
                },
                {
                    "concept_id": 19298354,
                    "concept_name": "Exterior angle theorem",
                    "search_score": 0.6142857142857143,
                    "levenshtein_score": 0.6415265270857766,
                    "graph_score": 0.019658831306227295,
                    "ontology_local_score": 0.8333333333333333,
                    "ontology_global_score": 0.8333333333333333,
                    "keywords_score": 0.8636363636363636,
                    "mixed_score": 0.6884762474748745,
                },
                {
                    "concept_id": 1975821,
                    "concept_name": "Skew lines",
                    "search_score": 0.5428571428571429,
                    "levenshtein_score": 0.4842714203388464,
                    "graph_score": 0.1360588093140798,
                    "ontology_local_score": 0.6666666666666666,
                    "ontology_global_score": 0.6666666666666666,
                    "keywords_score": 0.41548295454545453,
                    "mixed_score": 0.4861295755839665,
                },
                {
                    "concept_id": 201359,
                    "concept_name": "Squaring the circle",
                    "search_score": 0.2035714285714286,
                    "levenshtein_score": 0.33646712232710085,
                    "graph_score": 0.1721140113269017,
                    "ontology_local_score": 0.6666666666666666,
                    "ontology_global_score": 0.6666666666666666,
                    "keywords_score": 0.5859375,
                    "mixed_score": 0.4508436718627077,
                },
                {
                    "concept_id": 22634860,
                    "concept_name": "Line segment",
                    "search_score": 0.475,
                    "levenshtein_score": 0.1325957938789834,
                    "graph_score": 0.5986913115507982,
                    "ontology_local_score": 0.6666666666666666,
                    "ontology_global_score": 0.6666666666666666,
                    "keywords_score": 0.41548295454545453,
                    "mixed_score": 0.46607005326723033,
                },
                {
                    "concept_id": 250265,
                    "concept_name": "Rhumb line",
                    "search_score": 0.75,
                    "levenshtein_score": 0.36085995785542146,
                    "graph_score": 0.06455550670823762,
                    "ontology_local_score": 0.8333333333333333,
                    "ontology_global_score": 0.8333333333333333,
                    "keywords_score": 0.49502840909090906,
                    "mixed_score": 0.567426400409743,
                },
                {
                    "concept_id": 30654,
                    "concept_name": "Triangle",
                    "search_score": 0.06785714285714284,
                    "levenshtein_score": 0.6138533146135234,
                    "graph_score": 0.5733705181747822,
                    "ontology_local_score": 0.6666666666666666,
                    "ontology_global_score": 0.6666666666666666,
                    "keywords_score": 0.6818181818181819,
                    "mixed_score": 0.5341985987930565,
                },
                {
                    "concept_id": 31482,
                    "concept_name": "Tangent",
                    "search_score": 0.06785714285714284,
                    "levenshtein_score": 0.1724878951919264,
                    "graph_score": 0.49466512442988103,
                    "ontology_local_score": 0.6666666666666666,
                    "ontology_global_score": 0.6666666666666666,
                    "keywords_score": 0.29829545454545453,
                    "mixed_score": 0.34506642832350864,
                },
                {
                    "concept_id": 3307757,
                    "concept_name": "Simson line",
                    "search_score": 0.27142857142857146,
                    "levenshtein_score": 0.4842714203388464,
                    "graph_score": 0.016711001857071323,
                    "ontology_local_score": 0.6666666666666666,
                    "ontology_global_score": 0.6666666666666666,
                    "keywords_score": 0.41548295454545453,
                    "mixed_score": 0.4199090805525514,
                },
                {
                    "concept_id": 462730,
                    "concept_name": "Inscribed angle",
                    "search_score": 0.46071428571428574,
                    "levenshtein_score": 0.6445516077439113,
                    "graph_score": 0.16070615599273574,
                    "ontology_local_score": 0.8333333333333333,
                    "ontology_global_score": 0.8333333333333333,
                    "keywords_score": 0.8636363636363636,
                    "mixed_score": 0.6723204563279598,
                },
                {
                    "concept_id": 48082,
                    "concept_name": "Great circle",
                    "search_score": 0.40714285714285714,
                    "levenshtein_score": 0.5850789404910146,
                    "graph_score": 0.5021767816453578,
                    "ontology_local_score": 0.6666666666666666,
                    "ontology_global_score": 0.6666666666666666,
                    "keywords_score": 0.5859375,
                    "mixed_score": 0.561856007333426,
                },
                {
                    "concept_id": 524003,
                    "concept_name": "Internal and external angles",
                    "search_score": 0.9464285714285714,
                    "levenshtein_score": 0.6012700464524801,
                    "graph_score": 0.07081068093054668,
                    "ontology_local_score": 1,
                    "ontology_global_score": 1,
                    "keywords_score": 1,
                    "mixed_score": 0.8365572893466409,
                },
                {
                    "concept_id": 5407025,
                    "concept_name": "Sum of angles of a triangle",
                    "search_score": 0.35714285714285715,
                    "levenshtein_score": 0.39303319759855243,
                    "graph_score": 0.03988456214366818,
                    "ontology_local_score": 0.8333333333333333,
                    "ontology_global_score": 0.8333333333333333,
                    "keywords_score": 0.8636363636363636,
                    "mixed_score": 0.6017962497069635,
                },
                {
                    "concept_id": 6220,
                    "concept_name": "Circle",
                    "search_score": 0.6107142857142858,
                    "levenshtein_score": 0.7313486756137682,
                    "graph_score": 0.5639569790200692,
                    "ontology_local_score": 0.6666666666666666,
                    "ontology_global_score": 0.6666666666666666,
                    "keywords_score": 0.5859375,
                    "mixed_score": 0.6306887730535959,
                },
                {
                    "concept_id": 664497,
                    "concept_name": "Parallel (geometry)",
                    "search_score": 0.2035714285714286,
                    "levenshtein_score": 0.06793369410635872,
                    "graph_score": 0.37648143621213553,
                    "ontology_local_score": 0.6666666666666666,
                    "ontology_global_score": 0.6666666666666666,
                    "keywords_score": 0.41548295454545453,
                    "mixed_score": 0.3798640364817561,
                },
                {
                    "concept_id": 76956,
                    "concept_name": "Right angle",
                    "search_score": 0.65,
                    "levenshtein_score": 0.5768412624599064,
                    "graph_score": 0.33344055889923974,
                    "ontology_local_score": 0.8333333333333333,
                    "ontology_global_score": 0.8333333333333333,
                    "keywords_score": 0.8181818181818181,
                    "mixed_score": 0.7036581240467887,
                },
                {
                    "concept_id": 89246,
                    "concept_name": "Curve",
                    "search_score": 0.6107142857142858,
                    "levenshtein_score": 0.009984282260938813,
                    "graph_score": 0.5770427269536036,
                    "ontology_local_score": 0.6666666666666666,
                    "ontology_global_score": 0.6666666666666666,
                    "keywords_score": 0.41548295454545453,
                    "mixed_score": 0.4726563252076613,
                },
                {
                    "concept_id": 91111,
                    "concept_name": "Angle trisection",
                    "search_score": 0.6821428571428572,
                    "levenshtein_score": 0.5472899496445176,
                    "graph_score": 0.41459377337261616,
                    "ontology_local_score": 0.8333333333333333,
                    "ontology_global_score": 0.8333333333333333,
                    "keywords_score": 0.8181818181818181,
                    "mixed_score": 0.7137693200003894,
                },
                {
                    "concept_id": 946975,
                    "concept_name": "Line (geometry)",
                    "search_score": 1,
                    "levenshtein_score": 0.2862599599983249,
                    "graph_score": 1,
                    "ontology_local_score": 1,
                    "ontology_global_score": 1,
                    "keywords_score": 0.6058238636363636,
                    "mixed_score": 0.7746861530906578,
                },
            ]
        }
    )


WikifyResponse = List[WikifyResponseElem]


class KeywordsRequest(BaseModel):
    """
    Object containing the raw text to extract keywords.
    """
    raw_text: str = Field(
        ...,
        title="Raw Text",
        description="Raw text to be wikified",
        examples=["To draw a straight line from any point to any point.\nTo produce a finite straight line continuously in a straight line.\nTo describe a circle with any center and radius.\nThat all right angles equal one another.\nThat, if a straight line falling on two straight lines makes the interior angles on the same side less than two right angles, the two straight lines, if produced indefinitely, meet on that side on which are the angles less than the two right angles."]
    )


KeywordsResponse = List[str]
