from typing import Union, List, Annotated

from fastapi import APIRouter, Security, Query

from graphai.api.auth.router import get_current_active_user
import graphai.api.ontology.schemas as schemas
from graphai.api.common.schemas import TaskIDResponse
from graphai.api.common.utils import format_api_results

import graphai.celery.ontology.jobs as jobs
from graphai.celery.common.celery_config import get_task_info

# Initialise ontology router
router = APIRouter(
    prefix='/ontology',
    tags=['ontology'],
    responses={404: {'description': 'Not found'}},
    dependencies=[Security(get_current_active_user, scopes=['ontology'])]
)


@router.get('/tree', response_model=schemas.TreeResponse)
async def tree():
    return jobs.tree_job()


@router.get('/tree/category/{category_id}', response_model=Union[schemas.CategoryDetailsResponse, None])
async def cat_info(category_id):
    return jobs.category_info_job(category_id)


@router.get('/tree/cluster/{cluster_id}', response_model=schemas.ClusterDetailsResponse)
async def cluster_info(cluster_id):
    return jobs.cluster_info_job(cluster_id)


@router.get('/tree/concept/',
            response_model=Union[schemas.ConceptDetailsSingleResponse, List[schemas.ConceptDetailsSingleResponse]])
async def concept_info(concept_id: Annotated[List[str], Query()]):
    return jobs.concept_info_job(concept_id)


@router.get('/openalex/category/{category_id}/nearest_topics', response_model=schemas.OpenalexCategoryNearestTopicsResponse)
async def openalex_category_nearest_topics(category_id):
    return jobs.openalex_category_nearest_topics_job(category_id)


@router.get('/openalex/topic/{topic_id}/nearest_categories', response_model=schemas.OpenalexTopicNearestCategoriesResponse)
async def openalex_topic_nearest_categories(topic_id):
    return jobs.openalex_topic_nearest_categories_job(topic_id)


@router.post('/recompute_clusters', response_model=TaskIDResponse)
async def recompute_clusters(data: schemas.RecomputeClustersRequest):
    n_clusters = data.n_clusters
    min_n = data.min_n
    task_id = jobs.recompute_clusters_job(n_clusters, min_n)
    return {'task_id': task_id}


@router.get('/recompute_clusters/status/{task_id}', response_model=schemas.RecomputeClustersResponse)
async def recompute_clusters_status(task_id):
    full_results = get_task_info(task_id)
    task_results = full_results['results']
    if task_results is not None:
        if 'results' in task_results:
            task_results = {
                'results': task_results['results'],
                'category_assignments': task_results['category_assignments'],
                'impurity_count': task_results['impurity_count'],
                'impurity_proportion': task_results['impurity_proportion'],
                'successful': task_results['results'] is not None
            }
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)


@router.post('/graph_distance', response_model=schemas.GraphDistanceResponse)
async def compute_graph_distance(data: schemas.GraphDistanceRequest):
    src = data.src
    tgt = data.tgt
    src_type = data.src_type
    tgt_type = data.tgt_type
    avg = data.avg
    coeffs = data.coeffs
    return jobs.graph_distance_job(src, tgt, src_type, tgt_type, avg, coeffs)


@router.post('/nearest_neighbor/concept/category', response_model=schemas.GraphConceptNearestCategoryResponse)
async def compute_graph_concept_nearest_category(data: schemas.GraphConceptNearestCategoryRequest):
    src = data.src
    avg = data.avg
    coeffs = data.coeffs
    top_n = data.top_n
    use_depth_3 = data.top_down_search
    return_clusters = data.return_clusters
    use_embeddings = data.use_embeddings
    return jobs.concept_nearest_category_job(src, avg, coeffs, top_n, use_depth_3, return_clusters, use_embeddings)


@router.post('/nearest_neighbor/cluster/category', response_model=schemas.GraphClusterNearestCategoryResponse)
async def compute_graph_cluster_nearest_category(data: schemas.GraphClusterNearestCategoryRequest):
    src = data.src
    avg = data.avg
    coeffs = data.coeffs
    top_n = data.top_n
    use_depth_3 = data.top_down_search
    use_embeddings = data.use_embeddings
    return jobs.cluster_nearest_category_job(src, avg, coeffs, top_n, use_depth_3, use_embeddings)


@router.post('/nearest_neighbor/concept/concept', response_model=schemas.GraphNearestConceptResponse)
async def compute_graph_nearest_concept(data: schemas.GraphNearestConceptRequest):
    src = data.src
    top_n = data.top_n
    use_embeddings = data.use_embeddings
    return jobs.concept_nearest_concept_job(src, top_n, use_embeddings)


@router.post('/break_up_cluster', response_model=schemas.BreakUpClustersResponse)
async def break_up_cluster(data: schemas.BreakUpClusterRequest):
    cluster_id = data.cluster_id
    n_clusters = data.n_clusters
    return jobs.breakup_cluster_job(cluster_id, n_clusters)
