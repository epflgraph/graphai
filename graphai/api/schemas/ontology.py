from pydantic import BaseModel, Field
from typing import List, Union, Dict, Literal, Tuple, Optional

from graphai.api.schemas.common import TaskStatusResponse


class TreeResponseElem(BaseModel):
    """
    Object representing the output of the /ontology/tree endpoint.
    """

    child_id: int = Field(
        ...,
        title="Child category ID",
        description="ID of the child category"
    )

    parent_id: int = Field(
        ...,
        title="Parent category ID",
        description="ID of the parent category"
    )


class TreeResponse(BaseModel):
    child_to_parent: Union[List[TreeResponseElem], None] = Field(
        title="Ontology tree results",
        description="Child-parent relationships in the ontology's predefined tree."
    )


class CategoryInfoRequest(BaseModel):
    category_id: str = Field(
        title="Category ID"
    )


class CategoryInfoResponse(BaseModel):
    category_id: str = Field(
        title="Category ID"
    )

    depth: int = Field(
        title="Depth"
    )

    id: str = Field(
        title="Reference concept ID"
    )

    name: str = Field(
        title="Reference concept name"
    )


class CategoryParentResponse(BaseModel):
    parent: Union[str, None] = Field(
        title="Parent category"
    )


class CategoryChildrenRequest(BaseModel):
    category_id: str = Field(
        title="Category ID"
    )

    tgt_type: Literal['concept', 'category', 'cluster'] = Field(
        title="Target type"
    )


class CategoryChildrenResponse(BaseModel):
    children: Union[List[str], None] = Field(
        title="Child categories"
    )


class TreeParentResponse(TaskStatusResponse):
    task_result: Union[List[TreeResponseElem], None] = Field(
        title="Ontology tree results",
        description="Child-parent relationships in the ontology's predefined tree as a list of dicts."
    )


class RecomputeClustersRequest(BaseModel):
    n_clusters: int = Field(
        title="# of clusters",
        description="Number of clusters to cluster the existing ontology concepts into.",
        default=2200
    )

    min_n: int = Field(
        title="Minimum non-outlier size",
        description="Minimum size that a cluster must have in order not to be considered an outlier. Outlier clusters "
                    "are reassigned to larger clusters. Defaults to 1, which would result in outlier reassignment "
                    "being skipped.",
        default=1
    )


class OneConceptResponseElement(BaseModel):
    name: str = Field(
        title="Concept name"
    )

    id: int = Field(
        title="Concept id"
    )


class RecomputeClustersTaskResponse(BaseModel):
    results: Union[Dict[int, List[OneConceptResponseElement]], None] = Field(
        title="Cluster recomputation results",
        description="A mapping of each cluster number to the list of the cluster's concepts (each of which "
                    "is a dictionary with the 'name' and 'id' of the concept)."
    )

    category_assignments: Union[Dict[int, str], None] = Field(
        title="Cluster to category assignments",
        description="A mapping of each cluster to its category"
    )

    impurity_count: int = Field(
        title="Impurity count",
        description="Number of concepts that, through the process of cluster recomputation, ended up assigned "
                    "to a category different than their original."
    )

    impurity_proportion: float = Field(
        title="Impurity proportion",
        description="Proportion of concepts that, through the process of cluster recomputation, ended up assigned "
                    "to a category different than their original. Equal to impurity_count divided by number of "
                    "ontology concepts."
    )

    successful: bool = Field(
        title="Success flag",
        description="Whether the computation was successful."
    )


class RecomputeClustersResponse(TaskStatusResponse):
    task_result: Union[RecomputeClustersTaskResponse, None] = Field(
        title="Cluster recomputation response",
        description="A dict containing the resulting recomputed clusters and a freshness flag."
    )


class GraphDistanceRequest(BaseModel):
    src: str = Field(
        title="Source node"
    )

    src_type: Literal['concept', 'category'] = Field(
        title="Type of source node",
        default="concept"
    )

    tgt: str = Field(
        title="Target node"
    )

    tgt_type: Literal['concept', 'category'] = Field(
        title="Type of target node",
        default="category"
    )

    avg: Literal['none', 'linear', 'log'] = Field(
        title="Averaging",
        default="linear"
    )

    coeffs: Union[None, Tuple[float, float]] = Field(
        title="Coefficients",
        default=(1.0, 1.0)
    )


class GraphDistanceResponse(BaseModel):
    sim: Union[None, float] = Field(
        title="Node similarity"
    )


class GraphNearestCategoryRequest(BaseModel):
    src: str = Field(
        title="Source concept"
    )

    avg: Literal['none', 'linear', 'log'] = Field(
        title="Averaging",
        default="linear"
    )

    coeffs: Union[None, Tuple[float, float]] = Field(
        title="Coefficients",
        description="The coefficients for the concepts and anchor pages of a category, respectively. ",
        default=(1.0, 1.0)
    )

    top_n: int = Field(
        title="Top n",
        default=1
    )

    top_down_search: bool = Field(
        title="Top-down search",
        description="Only valid for concept-category. "
                    "Whether to directly search in depth-4 categories or to start the search higher, at depth 3. "
                    "True by default, as this generally yield better results. Set to False in order to get "
                    "a ranking based on raw similarity scores between the given concept and depth-4 categories.",
        default=True
    )

    return_clusters: bool = Field(
        title="Return clusters",
        description="If set, the results will include, for each of the most similar categories, "
                    "the top 3 most similar clusters to the given concept.",
        default=False
    )


class NearestClusterElement(BaseModel):
    cluster_id: str = Field(
        title="Cluster ID"
    )

    score: float = Field(
        title="Score"
    )

    rank: int = Field(
        title="Rank"
    )


class NearestCategoryElement(BaseModel):
    category_id: str = Field(
        title="Category ID"
    )

    score: float = Field(
        title="Score"
    )

    rank: int = Field(
        title="Rank"
    )

    clusters: Optional[NearestClusterElement] = Field(
        title="Clusters"
    )


class GraphNearestCategoryResponse(BaseModel):
    scores: Union[None, List[NearestCategoryElement]] = Field(
        title="Closest matches"
    )

    parent_category: Union[None, str] = Field(
        title="Parent category",
        description="If the `top_down_search` flag was set, this field will contain the id of the closest "
                    "depth-3 category. In that case, the top few categories (as many as this depth-3 category "
                    "has children) will be children of this category. If the flag is not set, this value is null."
    )


class GraphNearestConceptRequest(BaseModel):
    src: str = Field(
        title="Source concept"
    )

    top_n: int = Field(
        title="Top n",
        default=1
    )


class GraphNearestConceptResponse(BaseModel):
    closest: Union[None, List[str]] = Field(
        title="Closest matches"
    )

    scores: Union[None, List[float]] = Field(
        title="Scores"
    )
