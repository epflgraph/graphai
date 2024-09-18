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

    embedding_local_score: float = Field(
        ...,
        title="Embedding local score",
        description="Score that measures the relevance of the given concept with respect to the other concepts found for the given set of keywords according to the embedding vectors"
    )

    embedding_global_score: float = Field(
        ...,
        title="Embedding global score",
        description="Score that measures the relevance of the given concept with respect to all the other concepts according to the embedding vectors"
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

    embedding_keywords_score: float = Field(
        ...,
        title="Embedding keywords score",
        description="Score that measures the relevance of the sets of keywords for which the given concept is found according to the embedding vectors"
    )

    graph_keywords_score: float = Field(
        ...,
        title="Graph keywords score",
        description="Score that measures the relevance of the sets of keywords for which the given concept is found according to the graph"
    )

    ontology_keywords_score: float = Field(
        ...,
        title="Ontology keywords score",
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
                    "embedding_local_score": 0.936190393844887,
                    "embedding_global_score": 0.9447807640350212,
                    "graph_score": 0.7920787642145825,
                    "ontology_local_score": 1,
                    "ontology_global_score": 1,
                    "embedding_keywords_score": 1,
                    "graph_keywords_score": 0.9806112066572416,
                    "ontology_keywords_score": 1,
                    "mixed_score": 0.974893873592604,
                },
                {
                    "concept_id": 12401488,
                    "concept_name": "Triangle center",
                    "search_score": 0.3392857142857143,
                    "levenshtein_score": 0.4680631523928115,
                    "embedding_local_score": 0.5498335395955196,
                    "embedding_global_score": 0.6047569988500392,
                    "graph_score": 0.30184919393927595,
                    "ontology_local_score": 0.6666666666666666,
                    "ontology_global_score": 0.6666666666666666,
                    "embedding_keywords_score": 0.18117340257708767,
                    "graph_keywords_score": 0.22692356368077668,
                    "ontology_keywords_score": 0.18009831669894236,
                    "mixed_score": 0.3889476967863415,
                },
                {
                    "concept_id": 13295107,
                    "concept_name": "Transversal (geometry)",
                    "search_score": 0.7178571428571429,
                    "levenshtein_score": 0.2110192540065071,
                    "embedding_local_score": 0.7899500675092881,
                    "embedding_global_score": 0.7783132229139607,
                    "graph_score": 0.079490218003065,
                    "ontology_local_score": 0.8333333333333333,
                    "ontology_global_score": 0.8333333333333333,
                    "embedding_keywords_score": 0.8541509524253893,
                    "graph_keywords_score": 0.8158838242881319,
                    "ontology_keywords_score": 0.8540145985401459,
                    "mixed_score": 0.6477110513680882,
                },
                {
                    "concept_id": 146689,
                    "concept_name": "Earth radius",
                    "search_score": 0.475,
                    "levenshtein_score": 0.5850789404910146,
                    "embedding_local_score": 0.6058537722364034,
                    "embedding_global_score": 0.6191743554865394,
                    "graph_score": 0.04560562055085982,
                    "ontology_local_score": 0.2222222222222222,
                    "ontology_global_score": 0.0006414535135995178,
                    "embedding_keywords_score": 0.3393292344705985,
                    "graph_keywords_score": 0.32094309692256273,
                    "ontology_keywords_score": 0.18025826114750698,
                    "mixed_score": 0.2747973601576835,
                },
                {
                    "concept_id": 152547,
                    "concept_name": "Bisection",
                    "search_score": 0.27142857142857146,
                    "levenshtein_score": 0.2470772552749216,
                    "embedding_local_score": 0.5916531383849596,
                    "embedding_global_score": 0.5829791653941659,
                    "graph_score": 0.3652836360342975,
                    "ontology_local_score": 0.6666666666666666,
                    "ontology_global_score": 0.6666666666666666,
                    "embedding_keywords_score": 0.6658021950254382,
                    "graph_keywords_score": 0.6072448552560294,
                    "ontology_keywords_score": 0.6649784001191716,
                    "mixed_score": 0.49403585288280044,
                },
                {
                    "concept_id": 161243,
                    "concept_name": "Nine-point circle",
                    "search_score": 0.5428571428571429,
                    "levenshtein_score": 0.3974121105221985,
                    "embedding_local_score": 0.6128662674658956,
                    "embedding_global_score": 0.5954070757052229,
                    "graph_score": 0.23567484697484573,
                    "ontology_local_score": 0.6666666666666666,
                    "ontology_global_score": 0.6666666666666666,
                    "embedding_keywords_score": 0.6252501137210749,
                    "graph_keywords_score": 0.6809346631686672,
                    "ontology_keywords_score": 0.6234172501117234,
                    "mixed_score": 0.5454425715474266,
                },
                {
                    "concept_id": 165487,
                    "concept_name": "World line",
                    "search_score": 0.27142857142857146,
                    "levenshtein_score": 0.20854298868986648,
                    "embedding_local_score": 0.5557247926738698,
                    "embedding_global_score": 0.5875702591903744,
                    "graph_score": 0.018301492693509606,
                    "ontology_local_score": 0.6666666666666666,
                    "ontology_global_score": 0.6666666666666666,
                    "embedding_keywords_score": 0.32915926499040604,
                    "graph_keywords_score": 0.35937866385263295,
                    "ontology_keywords_score": 0.3324892000595859,
                    "mixed_score": 0.35381073854308764,
                },
                {
                    "concept_id": 1780815,
                    "concept_name": "Radius",
                    "search_score": 0.6785714285714286,
                    "levenshtein_score": 0.7313486756137682,
                    "embedding_local_score": 0.6058537722364034,
                    "embedding_global_score": 0.6123892146782264,
                    "graph_score": 0.40341637127911656,
                    "ontology_local_score": 0.2222222222222222,
                    "ontology_global_score": 0.6666666666666666,
                    "embedding_keywords_score": 0.3393292344705985,
                    "graph_keywords_score": 0.32094309692256273,
                    "ontology_keywords_score": 0.18025826114750698,
                    "mixed_score": 0.43983570252851467,
                },
                {
                    "concept_id": 1896705,
                    "concept_name": "Osculating circle",
                    "search_score": 0.06785714285714284,
                    "levenshtein_score": 0.3974121105221985,
                    "embedding_local_score": 0.6095596040570643,
                    "embedding_global_score": 0.6001356168702854,
                    "graph_score": 0.15095693713821154,
                    "ontology_local_score": 0.6666666666666666,
                    "ontology_global_score": 0.6666666666666666,
                    "embedding_keywords_score": 0.6252501137210749,
                    "graph_keywords_score": 0.6809346631686672,
                    "ontology_keywords_score": 0.6234172501117234,
                    "mixed_score": 0.44197078056376315,
                },
                {
                    "concept_id": 1898401,
                    "concept_name": "Arc length",
                    "search_score": 0.475,
                    "levenshtein_score": 0.09530697673156802,
                    "embedding_local_score": 0.6214735523605726,
                    "embedding_global_score": 0.6206023872446591,
                    "graph_score": 0.37885045617052326,
                    "ontology_local_score": 0.6075750180441352,
                    "ontology_global_score": 0.6666666666666666,
                    "embedding_keywords_score": 0.45726985451624325,
                    "graph_keywords_score": 0.5967180832515523,
                    "ontology_keywords_score": 0.41496111506273836,
                    "mixed_score": 0.429472346018896,
                },
                {
                    "concept_id": 1975821,
                    "concept_name": "Skew lines",
                    "search_score": 0.5428571428571429,
                    "levenshtein_score": 0.4842714203388464,
                    "embedding_local_score": 0.5926923831317541,
                    "embedding_global_score": 0.6071866898797716,
                    "graph_score": 0.1467272460558672,
                    "ontology_local_score": 0.6666666666666666,
                    "ontology_global_score": 0.6666666666666666,
                    "embedding_keywords_score": 0.453132323677798,
                    "graph_keywords_score": 0.6342079203068262,
                    "ontology_keywords_score": 0.4571726500819306,
                    "mixed_score": 0.4997033279190881,
                },
                {
                    "concept_id": 201359,
                    "concept_name": "Squaring the circle",
                    "search_score": 0.2035714285714286,
                    "levenshtein_score": 0.33646712232710085,
                    "embedding_local_score": 0.6228894106876862,
                    "embedding_global_score": 0.6094673983855882,
                    "graph_score": 0.1853365490872417,
                    "ontology_local_score": 0.6666666666666666,
                    "ontology_global_score": 0.6666666666666666,
                    "embedding_keywords_score": 0.6252501137210749,
                    "graph_keywords_score": 0.6809346631686672,
                    "ontology_keywords_score": 0.6234172501117234,
                    "mixed_score": 0.4634098506722587,
                },
                {
                    "concept_id": 22634860,
                    "concept_name": "Line segment",
                    "search_score": 0.475,
                    "levenshtein_score": 0.1325957938789834,
                    "embedding_local_score": 0.6289329297461709,
                    "embedding_global_score": 0.6015345514255728,
                    "graph_score": 0.5940970321127149,
                    "ontology_local_score": 0.6666666666666666,
                    "ontology_global_score": 0.6666666666666666,
                    "embedding_keywords_score": 0.44397434138761116,
                    "graph_keywords_score": 0.6463107906576082,
                    "ontology_keywords_score": 0.4571726500819306,
                    "mixed_score": 0.47811753398436485,
                },
                {
                    "concept_id": 250265,
                    "concept_name": "Rhumb line",
                    "search_score": 0.7678571428571428,
                    "levenshtein_score": 0.36085995785542146,
                    "embedding_local_score": 0.7146040614704057,
                    "embedding_global_score": 0.7314235936190492,
                    "graph_score": 0.10852920758292811,
                    "ontology_local_score": 0.4848520041529702,
                    "ontology_global_score": 0.5297660279561411,
                    "embedding_keywords_score": 0.5403307788136165,
                    "graph_keywords_score": 0.7244902789723203,
                    "ontology_keywords_score": 0.48877312914546234,
                    "mixed_score": 0.49088968517023296,
                },
                {
                    "concept_id": 31482,
                    "concept_name": "Tangent",
                    "search_score": 0.06785714285714284,
                    "levenshtein_score": 0.1724878951919264,
                    "embedding_local_score": 0.5557247926738698,
                    "embedding_global_score": 0.59976908308104,
                    "graph_score": 0.4963883113854736,
                    "ontology_local_score": 0.6666666666666666,
                    "ontology_global_score": 0.6666666666666666,
                    "embedding_keywords_score": 0.32915926499040604,
                    "graph_keywords_score": 0.35937866385263295,
                    "ontology_keywords_score": 0.3324892000595859,
                    "mixed_score": 0.35549687067330726,
                },
                {
                    "concept_id": 3307757,
                    "concept_name": "Simson line",
                    "search_score": 0.27142857142857146,
                    "levenshtein_score": 0.4842714203388464,
                    "embedding_local_score": 0.6270069103429565,
                    "embedding_global_score": 0.578947689005502,
                    "graph_score": 0.013581124397822893,
                    "ontology_local_score": 0.6666666666666666,
                    "ontology_global_score": 0.6666666666666666,
                    "embedding_keywords_score": 0.44397434138761116,
                    "graph_keywords_score": 0.6463107906576082,
                    "ontology_keywords_score": 0.4571726500819306,
                    "mixed_score": 0.43210300146756936,
                },
                {
                    "concept_id": 462730,
                    "concept_name": "Inscribed angle",
                    "search_score": 0.44285714285714284,
                    "levenshtein_score": 0.6445516077439113,
                    "embedding_local_score": 0.7717801666974846,
                    "embedding_global_score": 0.768343013064605,
                    "graph_score": 0.12482171404042144,
                    "ontology_local_score": 0.8333333333333333,
                    "ontology_global_score": 0.8333333333333333,
                    "embedding_keywords_score": 0.8541509524253893,
                    "graph_keywords_score": 0.8158838242881319,
                    "ontology_keywords_score": 0.8540145985401459,
                    "mixed_score": 0.6622740540324344,
                },
                {
                    "concept_id": 48082,
                    "concept_name": "Great circle",
                    "search_score": 0.40714285714285714,
                    "levenshtein_score": 0.5850789404910146,
                    "embedding_local_score": 0.6194506424602977,
                    "embedding_global_score": 0.6151678032679938,
                    "graph_score": 0.5237975686048405,
                    "ontology_local_score": 0.6666666666666666,
                    "ontology_global_score": 0.6666666666666666,
                    "embedding_keywords_score": 0.6252501137210749,
                    "graph_keywords_score": 0.6809346631686672,
                    "ontology_keywords_score": 0.6234172501117234,
                    "mixed_score": 0.5752620110628913,
                },
                {
                    "concept_id": 524003,
                    "concept_name": "Internal and external angles",
                    "search_score": 0.9464285714285714,
                    "levenshtein_score": 0.6012700464524801,
                    "embedding_local_score": 0.9547191776223249,
                    "embedding_global_score": 0.9544700262181692,
                    "graph_score": 0.06348267860292894,
                    "ontology_local_score": 1,
                    "ontology_global_score": 1,
                    "embedding_keywords_score": 1,
                    "graph_keywords_score": 0.9806112066572416,
                    "ontology_keywords_score": 1,
                    "mixed_score": 0.8358244891138791,
                },
                {
                    "concept_id": 5407025,
                    "concept_name": "Sum of angles of a triangle",
                    "search_score": 0.3928571428571429,
                    "levenshtein_score": 0.39303319759855243,
                    "embedding_local_score": 0.7502904601610197,
                    "embedding_global_score": 0.7428783723631838,
                    "graph_score": 0.006235350315249621,
                    "ontology_local_score": 0.8333333333333333,
                    "ontology_global_score": 0.8333333333333333,
                    "embedding_keywords_score": 0.8541509524253893,
                    "graph_keywords_score": 0.8158838242881319,
                    "ontology_keywords_score": 0.8540145985401459,
                    "mixed_score": 0.6026876561381135,
                },
                {
                    "concept_id": 6220,
                    "concept_name": "Circle",
                    "search_score": 0.6107142857142858,
                    "levenshtein_score": 0.7313486756137682,
                    "embedding_local_score": 0.6247087590356082,
                    "embedding_global_score": 0.6044584589208712,
                    "graph_score": 0.5495989056289238,
                    "ontology_local_score": 0.6666666666666666,
                    "ontology_global_score": 0.6666666666666666,
                    "embedding_keywords_score": 0.6252501137210749,
                    "graph_keywords_score": 0.6809346631686672,
                    "ontology_keywords_score": 0.6234172501117234,
                    "mixed_score": 0.6404968907479984,
                },
                {
                    "concept_id": 664497,
                    "concept_name": "Parallel (geometry)",
                    "search_score": 0.2035714285714286,
                    "levenshtein_score": 0.06793369410635872,
                    "embedding_local_score": 0.6017217718876411,
                    "embedding_global_score": 0.6220133585096995,
                    "graph_score": 0.40095604092651027,
                    "ontology_local_score": 0.6666666666666666,
                    "ontology_global_score": 0.6666666666666666,
                    "embedding_keywords_score": 0.453132323677798,
                    "graph_keywords_score": 0.6342079203068262,
                    "ontology_keywords_score": 0.4571726500819306,
                    "mixed_score": 0.3948184056141364,
                },
                {
                    "concept_id": 76956,
                    "concept_name": "Right angle",
                    "search_score": 0.65,
                    "levenshtein_score": 0.5768412624599064,
                    "embedding_local_score": 0.7571720795352108,
                    "embedding_global_score": 0.7548885234150651,
                    "graph_score": 0.3076508249923096,
                    "ontology_local_score": 0.8333333333333333,
                    "ontology_global_score": 0.8333333333333333,
                    "embedding_keywords_score": 0.8246343320198054,
                    "graph_keywords_score": 0.8192570426095808,
                    "ontology_keywords_score": 0.8248175182481752,
                    "mixed_score": 0.7030698606760027,
                },
                {
                    "concept_id": 89246,
                    "concept_name": "Curve",
                    "search_score": 0.6107142857142858,
                    "levenshtein_score": 0.009984282260938813,
                    "embedding_local_score": 0.6061010985898527,
                    "embedding_global_score": 0.6058837846959582,
                    "graph_score": 0.6010512105924976,
                    "ontology_local_score": 0.6075750180441352,
                    "ontology_global_score": 0.6666666666666666,
                    "embedding_keywords_score": 0.45726985451624325,
                    "graph_keywords_score": 0.5967180832515523,
                    "ontology_keywords_score": 0.41496111506273836,
                    "mixed_score": 0.4660368744333562,
                },
                {
                    "concept_id": 91111,
                    "concept_name": "Angle trisection",
                    "search_score": 0.6821428571428572,
                    "levenshtein_score": 0.5472899496445176,
                    "embedding_local_score": 0.7114737175075206,
                    "embedding_global_score": 0.7222511927302219,
                    "graph_score": 0.42053296186769257,
                    "ontology_local_score": 0.8333333333333333,
                    "ontology_global_score": 0.8333333333333333,
                    "embedding_keywords_score": 0.8246343320198054,
                    "graph_keywords_score": 0.8192570426095808,
                    "ontology_keywords_score": 0.8248175182481752,
                    "mixed_score": 0.7163539488698042,
                },
                {
                    "concept_id": 946975,
                    "concept_name": "Line (geometry)",
                    "search_score": 1,
                    "levenshtein_score": 0.2862599599983249,
                    "embedding_local_score": 0.9162572314594017,
                    "embedding_global_score": 0.8476721575861583,
                    "graph_score": 1,
                    "ontology_local_score": 0.9778406317665507,
                    "ontology_global_score": 1,
                    "embedding_keywords_score": 0.653131668312657,
                    "graph_keywords_score": 0.9426016455778317,
                    "ontology_keywords_score": 0.6514226203690341,
                    "mixed_score": 0.7850418748754415,
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


class LectureExerciseRequest(BaseModel):
    """
    Object containing the input to generate with an LLM a lecture-aware exercise.
    """

    lecture_id: str = Field(
        ...,
        title="lecture_id",
        description="ID of the lecture for which to generate an exercise with an LLM.",
        examples=["0_92916guq"]
    )

    description: str = Field(
        ...,
        title="description",
        description="A description in plain language, that will be sent to the LLM, of what the exercise should be about.",
        examples=[r"An exercise to compute the volume of a sphere cap of angle $\alpha$ using spherical coordinates."]
    )

    include_solution: bool = Field(
        ...,
        title="include_solution",
        description="Whether to ask the LLM to return a solution along with the exercise.",
        examples=[True]
    )