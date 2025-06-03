from pydantic import BaseModel, Field
from typing import List, Union, Dict, Literal, Tuple, Optional

from graphai.api.common.schemas import TaskStatusResponse


class TreeResponseElem(BaseModel):
    """
    Object representing the output of the /ontology/tree endpoint.
    """

    child_id: str = Field(
        ...,
        title="Child category ID",
        description="ID of the child category"
    )

    parent_id: str = Field(
        ...,
        title="Parent category ID",
        description="ID of the parent category"
    )


class TreeResponse(BaseModel):
    child_to_parent: Union[List[TreeResponseElem], None] = Field(
        None, title="Ontology tree results",
        description="Child-parent relationships in the ontology's predefined tree."
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


class ConceptDetails(BaseModel):
    id: Union[str, None] = Field(
        title="Concept ID"
    )

    name: Union[str, None] = Field(
        title="Concept name"
    )


class CategoryDetailsResponse(BaseModel):
    info: Union[CategoryInfoResponse, None] = Field(
        title="Category info"
    )

    parent_category: Union[str, None] = Field(
        title="Parent category"
    )

    child_categories: Union[List[str], None] = Field(
        title="Child categories"
    )

    clusters: Union[List[str], None] = Field(
        title="Clusters",
        description="Depth 4 only, clusters attached to this category"
    )

    concepts: Union[List[ConceptDetails], None] = Field(
        title="Concepts",
        description="Depth 4 only, concepts attached to this category"
    )


class ClusterDetailsResponse(BaseModel):
    parent: Union[str, None] = Field(
        title="Parent category"
    )
    concepts: Union[List[ConceptDetails], None] = Field(
        title="Concepts",
        description="Concepts under this cluster"
    )


class ConceptDetailsSingleResponse(ConceptDetails):
    parent_category: Union[str, None] = Field(
        title="Parent category"
    )
    parent_cluster: Union[str, None] = Field(
        title="Parent cluster"
    )
    branch: Union[List[str], None] = Field(
        title="Full category branch"
    )


class OpenalexCategoryNearestTopicsResponseElem(BaseModel):
    category_id: str = Field(title="Category ID")
    category_name: str = Field(title="Category name")
    topic_id: str = Field(title="Topic ID")
    topic_name: str = Field(title="Topic name")
    embedding_score: float = Field(title="Embedding score")
    wikipedia_score: float = Field(title="Wikipedia score")
    score: float = Field(title="Score")


OpenalexCategoryNearestTopicsResponse = List[OpenalexCategoryNearestTopicsResponseElem]

OpenalexTopicNearestCategoriesResponseElem = OpenalexCategoryNearestTopicsResponseElem
OpenalexTopicNearestCategoriesResponse = List[OpenalexTopicNearestCategoriesResponseElem]


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
        None, title="Cluster recomputation results",
        description="A mapping of each cluster number to the list of the cluster's concepts (each of which "
                    "is a dictionary with the 'name' and 'id' of the concept)."
    )

    category_assignments: Union[Dict[int, str], None] = Field(
        None, title="Cluster to category assignments",
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


class BreakUpClusterRequest(BaseModel):
    cluster_id: str = Field(
        title="Cluster ID",
    )

    n_clusters: Union[int, List[int]] = Field(
        title="# of clusters",
        description="Number of clusters to break the given cluster into. Can be one integer or a list of integers.",
        default=2
    )


class BreakUpClustersClusterNumberResponse(BaseModel):
    clusters: Union[Dict[int, List[OneConceptResponseElement]], None] = Field(
        None, title="Cluster recomputation results",
        description="A mapping of each cluster number to the list of the cluster's concepts (each of which "
                    "is a dictionary with the 'name' and 'id' of the concept)."
    )

    n_clusters: int = Field(
        title="Number of clusters",
    )


class BreakUpClustersResponse(BaseModel):
    results: Union[List[BreakUpClustersClusterNumberResponse], None] = Field(
        None, title="Cluster break-up results",
    )


class GraphDistanceRequest(BaseModel):
    src: str = Field(
        title="Source node"
    )

    src_type: Literal['concept', 'cluster', 'category'] = Field(
        title="Type of source node",
        default="concept"
    )

    tgt: str = Field(
        title="Target node"
    )

    tgt_type: Literal['concept', 'cluster', 'category'] = Field(
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
        None, title="Node similarity"
    )


class GraphNearestCategoryRequest(BaseModel):
    avg: Literal['none', 'linear', 'log', 'adaptive'] = Field(
        title="Averaging",
        default="log"
    )

    coeffs: Union[None, Tuple[float, float]] = Field(
        title="Coefficients",
        description="The coefficients for the anchor pages and concepts of a category, respectively. ",
        default=(1.0, 10.0)
    )

    top_n: int = Field(
        title="Top n",
        default=1
    )

    top_down_search: bool = Field(
        title="Top-down search",
        description="Whether to directly search in depth-4 categories or to start the search higher, at depth 3. "
                    "True by default, as this generally yield better results. Set to False in order to get "
                    "a ranking based on raw similarity scores between the given entity and depth-4 categories.",
        default=False
    )

    use_embeddings: bool = Field(
        title="Use embeddings",
        description="Use embeddings. Currently, embeddings are only used "
                    "as a fallback if nothing is found through the graph.",
        default=False
    )


class GraphConceptNearestCategoryRequest(GraphNearestCategoryRequest):
    src: str = Field(
        title="Source concept"
    )

    return_clusters: Union[int, None] = Field(
        title="Return clusters",
        description="If not null, determines the k for which the top k closest clusters in "
                    "each of the top n categories are returned. If null, clusters are not returned.",
        default=3
    )


class GraphClusterNearestCategoryRequest(GraphNearestCategoryRequest):
    src: Union[List[str], str] = Field(
        title="Source cluster",
        description="The cluster to find the closest category for. The cluster can be an existing or a 'custom' one. "
                    "If this parameter is a single string, the string represents the ID of an existing cluster. "
                    "On the other hand, if a list of strings is provided, each element of the list is "
                    "considered to be the ID of a concept, and the cluster is a 'custom' cluster."
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


class NearestCategoryElementWithClusters(NearestCategoryElement):
    clusters: Optional[List[NearestClusterElement]] = Field(
        None, title="Clusters"
    )


class EmbeddingEnabledNearestEntityResponse(BaseModel):
    embeddings_used: bool = Field(
        title="Embeddings used",
        description="A flag that indicates whether embeddings were used for calculating the returned result."
    )


class GraphConceptNearestCategoryResponse(EmbeddingEnabledNearestEntityResponse):
    scores: Union[None, List[NearestCategoryElementWithClusters]] = Field(
        None, title="Closest matches"
    )

    parent_category: Union[None, str] = Field(
        None, title="Parent category",
        description="If the `top_down_search` flag was set, this field will contain the id of the closest "
                    "depth-3 category. In that case, the top few categories (as many as this depth-3 category "
                    "has children) will be children of this category. If the flag is not set, this value is null."
    )

    valid: bool = Field(
        title="Valid results",
        description="If this flag is unset while the results are not null, "
                    "it means that the top result has a score of 0, "
                    "meaning that effectively, all the results returned are random."
    )

    existing_label: Union[str, None] = Field(
        None, title="Existing category",
        description="If the requested concept already exists as part of the ontology, this value will reflect "
                    "its existing parent category."
    )


class GraphClusterNearestCategoryResponse(EmbeddingEnabledNearestEntityResponse):
    scores: Union[None, List[NearestCategoryElement]] = Field(
        None, title="Closest matches"
    )

    parent_category: Union[None, str] = Field(
        None, title="Parent category",
        description="If the `top_down_search` flag was set, this field will contain the id of the closest "
                    "depth-3 category. In that case, the top few categories (as many as this depth-3 category "
                    "has children) will be children of this category. If the flag is not set, this value is null."
    )

    existing_label: Union[str, None] = Field(
        None, title="Existing category",
        description="If the requested cluster already exists as part of the ontology, this value will reflect "
                    "its existing parent category."
    )


class GraphNearestConceptRequest(BaseModel):
    src: str = Field(
        title="Source concept"
    )

    top_n: int = Field(
        title="Top n",
        default=1
    )

    use_embeddings: bool = Field(
        title="Use embeddings",
        description="Use embeddings. Currently, embeddings are only used "
                    "as a fallback if nothing is found through the graph.",
        default=False
    )


class GraphNearestConceptResponse(EmbeddingEnabledNearestEntityResponse):
    closest: Union[None, List[str]] = Field(
        None, title="Closest matches"
    )

    scores: Union[None, List[float]] = Field(
        None, title="Scores"
    )
