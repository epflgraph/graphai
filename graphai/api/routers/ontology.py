from fastapi import APIRouter, Security
from celery import chain
from typing import Union

from graphai.api.schemas.ontology import (
    TreeResponse,
    CategoryInfoRequest,
    CategoryInfoResponse,
    CategoryParentResponse,
    CategoryChildrenRequest,
    TreeChildrenResponse,
    ClusterInfoRequest,
    RecomputeClustersRequest,
    RecomputeClustersResponse,
    GraphDistanceRequest,
    GraphDistanceResponse,
    GraphConceptNearestCategoryRequest,
    GraphConceptNearestCategoryResponse,
    GraphClusterNearestCategoryRequest,
    GraphClusterNearestCategoryResponse,
    GraphNearestConceptRequest,
    GraphNearestConceptResponse,
    BreakUpClusterRequest,
    BreakUpClustersResponse,
)
from graphai.api.schemas.common import TaskIDResponse

from graphai.api.celery_tasks.common import format_api_results
from graphai.api.celery_tasks.ontology import (
    get_cluster_parent_task,
    get_cluster_children_task,
    recompute_clusters_task,
    break_up_cluster_task,
    get_concept_category_similarity_task,
    get_concept_cluster_similarity_task,
    get_cluster_cluster_similarity_task,
    get_concept_category_closest_task,
    get_cluster_category_closest_task,
    get_concept_concept_similarity_task,
    get_concept_concept_closest_task,
    get_category_category_similarity_task,
    get_cluster_category_similarity_task
)

from graphai.api.celery_jobs.ontology import (
    tree_job,
    category_info_job,
    category_parent_job,
    category_children_job,
    cluster_parent_job,
    cluster_children_job,
    recompute_clusters_job
)
from graphai.api.routers.auth import get_current_active_user

from graphai.core.interfaces.celery_config import get_task_info

# Initialise ontology router
router = APIRouter(
    prefix='/ontology',
    tags=['ontology'],
    responses={404: {'description': 'Not found'}},
    dependencies=[Security(get_current_active_user, scopes=['ontology'])]
)


@router.get('/tree', response_model=TreeResponse)
async def tree():
    return tree_job()


@router.post('/tree/category/info', response_model=Union[CategoryInfoResponse, None])
async def cat_info(data: CategoryInfoRequest):
    category_id = data.category_id
    return category_info_job(category_id)


@router.post('/tree/category/parent', response_model=CategoryParentResponse)
async def cat_parent(data: CategoryInfoRequest):
    category_id = data.category_id
    return category_parent_job(category_id)


@router.post('/tree/category/children', response_model=TreeChildrenResponse)
async def cat_children(data: CategoryChildrenRequest):
    category_id = data.category_id
    dest_type = data.tgt_type
    return category_children_job(category_id, dest_type)


@router.post('/tree/cluster/parent', response_model=CategoryParentResponse)
async def cluster_parent(data: ClusterInfoRequest):
    cluster_id = data.cluster_id
    return cluster_parent_job(cluster_id)


@router.post('/tree/cluster/children', response_model=TreeChildrenResponse)
async def cluster_children(data: ClusterInfoRequest):
    cluster_id = data.cluster_id
    return cluster_children_job(cluster_id)


@router.post('/recompute_clusters', response_model=TaskIDResponse)
async def recompute_clusters(data: RecomputeClustersRequest):
    n_clusters = data.n_clusters
    min_n = data.min_n
    task_id = recompute_clusters_job(n_clusters, min_n)
    return {'task_id': task_id}


@router.get('/recompute_clusters/status/{task_id}', response_model=RecomputeClustersResponse)
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


@router.post('/graph_distance', response_model=GraphDistanceResponse)
async def compute_graph_distance(data: GraphDistanceRequest):
    src = data.src
    tgt = data.tgt
    src_type = data.src_type
    tgt_type = data.tgt_type
    avg = data.avg
    coeffs = data.coeffs
    assert coeffs is None or len(coeffs) == 2
    if src_type == 'concept' and tgt_type == 'category':
        task = get_concept_category_similarity_task.s(src, tgt, avg, coeffs)
    elif src_type == 'category' and tgt_type == 'concept':
        task = get_concept_category_similarity_task.s(tgt, src, avg, coeffs)
    elif src_type == 'concept' and tgt_type == 'cluster':
        task = get_concept_cluster_similarity_task.s(src, tgt, avg)
    elif src_type == 'cluster' and tgt_type == 'concept':
        task = get_concept_cluster_similarity_task.s(tgt, src, avg)
    elif src_type == 'cluster' and tgt_type == 'cluster':
        task = get_cluster_cluster_similarity_task.s(src, tgt, avg)
    elif src_type == 'cluster' and tgt_type == 'category':
        task = get_cluster_category_similarity_task.s(src, tgt, avg, coeffs)
    elif src_type == 'category' and tgt_type == 'cluster':
        task = get_cluster_category_similarity_task.s(tgt, src, avg, coeffs)
    elif src_type == 'category' and tgt_type == 'category':
        task = get_category_category_similarity_task.s(src, tgt, avg, coeffs)
    else:
        task = get_concept_concept_similarity_task.s(src, tgt)
    res = task.apply_async(priority=6).get(timeout=30)
    return res


@router.post('/nearest_neighbor/concept/category', response_model=GraphConceptNearestCategoryResponse)
async def compute_graph_concept_nearest_category(data: GraphConceptNearestCategoryRequest):
    src = data.src
    avg = data.avg
    coeffs = data.coeffs
    top_n = data.top_n
    use_depth_3 = data.top_down_search
    return_clusters = data.return_clusters
    assert coeffs is None or len(coeffs) == 2
    task = get_concept_category_closest_task.s(src, avg, coeffs, top_n, use_depth_3, return_clusters)
    res = task.apply_async(priority=6).get(timeout=30)
    return res


@router.post('/nearest_neighbor/cluster/category', response_model=GraphClusterNearestCategoryResponse)
async def compute_graph_cluster_nearest_category(data: GraphClusterNearestCategoryRequest):
    src = data.src
    avg = data.avg
    coeffs = data.coeffs
    top_n = data.top_n
    use_depth_3 = data.top_down_search
    assert coeffs is None or len(coeffs) == 2
    task = get_cluster_category_closest_task.s(src, avg, coeffs, top_n, use_depth_3)
    res = task.apply_async(priority=6).get(timeout=30)
    return res


@router.post('/nearest_neighbor/concept/concept', response_model=GraphNearestConceptResponse)
async def compute_graph_nearest_concept(data: GraphNearestConceptRequest):
    src = data.src
    top_n = data.top_n
    task = get_concept_concept_closest_task.s(src, top_n)
    res = task.apply_async(priority=6).get(timeout=30)
    return res


@router.post('/break_up_cluster', response_model=BreakUpClustersResponse)
async def break_up_cluster(data: BreakUpClusterRequest):
    cluster_id = data.cluster_id
    n_clusters = data.n_clusters
    task_list = [break_up_cluster_task.s(cluster_id, n_clusters)]
    task = chain(task_list)
    res = task.apply_async(priority=6).get(timeout=30)
    return res
