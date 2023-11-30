from pydantic import BaseModel, Field
from typing import List, Union, Dict

from graphai.api.schemas.common import TaskStatusResponse


class TreeResponseElem(BaseModel):
    """
    Object representing the output of the /ontology/tree endpoint.
    """

    ChildCategoryID: int = Field(
        ...,
        title="Child category ID",
        description="ID of the child category"
    )

    ParentCategoryID: int = Field(
        ...,
        title="Parent category ID",
        description="ID of the parent category"
    )


class TreeResponse(TaskStatusResponse):
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

    successful: bool = Field(
        title="Success flag",
        description="Whether the computation was successful."
    )


class RecomputeClustersResponse(TaskStatusResponse):
    task_result: Union[RecomputeClustersTaskResponse, None] = Field(
        title="Cluster recomputation response",
        description="A dict containing the resulting recomputed clusters and a freshness flag."
    )
