from fastapi import APIRouter
from celery import chain

from graphai.api.schemas.ontology import (
    TreeResponse,
    CategoryInfoRequest,
    CategoryInfoResponse,
    CategoryParentResponse,
    CategoryChildrenRequest,
    CategoryChildrenResponse,
    RecomputeClustersRequest,
    RecomputeClustersResponse,
    GraphDistanceRequest,
    GraphDistanceResponse,
    GraphNearestCategoryRequest,
    GraphNearestCategoryResponse,
    GraphNearestConceptRequest,
    GraphNearestConceptResponse,
)
from graphai.api.schemas.common import TaskIDResponse

from graphai.api.common.log import log
from graphai.api.celery_tasks.common import format_api_results
from graphai.api.celery_tasks.ontology import (
    get_ontology_tree_task,
    get_category_info_task,
    get_category_parent_task,
    get_category_children_task,
    get_category_concepts_task,
    get_category_clusters_task,
    recompute_clusters_task,
    get_concept_category_similarity_task,
    get_concept_category_closest_task,
    get_concept_concept_similarity_task,
    get_concept_concept_closest_task,
    get_category_category_similarity_task
)
from graphai.core.interfaces.celery_config import get_task_info

# Initialise ontology router
router = APIRouter(
    prefix='/ontology',
    tags=['ontology'],
    responses={404: {'description': 'Not found'}}
)


@router.get('/tree', response_model=TreeResponse)
async def tree():
    log('Returning the ontology tree')
    results = get_ontology_tree_task.s().apply_async(priority=6).get(timeout=10)
    return results


@router.get('/tree/info/{category_id}', response_model=CategoryInfoResponse)
async def parent(data: CategoryInfoRequest):
    category_id = data.category_id
    results = get_category_info_task.s(category_id).apply_async(priority=6).get(timeout=10)
    return results


@router.get('/tree/parent/{category_id}', response_model=CategoryParentResponse)
async def parent(data: CategoryInfoRequest):
    category_id = data.category_id
    results = get_category_parent_task.s(category_id).apply_async(priority=6).get(timeout=10)
    return results


@router.get('/tree/children/{category_id}', response_model=CategoryChildrenResponse)
async def children(data: CategoryChildrenRequest):
    category_id = data.category_id
    dest_type = data.tgt_type
    if dest_type == 'category':
        task = get_category_children_task.s(category_id)
    elif dest_type == 'concept':
        task = get_category_concepts_task.s(category_id)
    else:
        task = get_category_clusters_task.s(category_id)
    results = task.apply_async(priority=6).get(timeout=10)
    return results


@router.post('/recompute_clusters', response_model=TaskIDResponse)
async def recompute_clusters(data: RecomputeClustersRequest):
    n_clusters = data.n_clusters
    min_n = data.min_n
    task_list = [recompute_clusters_task.s(n_clusters, min_n)]
    task = chain(task_list)
    task = task.apply_async(priority=6)
    return {'task_id': task.id}


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
    elif src_type == 'category' and tgt_type == 'category':
        task = get_category_category_similarity_task.s(src, tgt, avg, coeffs)
    else:
        task = get_concept_concept_similarity_task.s(src, tgt)
    res = task.apply_async(priority=6).get(timeout=30)
    return res


@router.post('/nearest_neighbor/category', response_model=GraphNearestCategoryResponse)
async def compute_graph_nearest_category(data: GraphNearestCategoryRequest):
    src = data.src
    avg = data.avg
    coeffs = data.coeffs
    top_n = data.top_n
    use_depth_3 = data.top_down_search
    assert coeffs is None or len(coeffs) == 2
    task = get_concept_category_closest_task.s(src, avg, coeffs, top_n, use_depth_3)
    res = task.apply_async(priority=6).get(timeout=30)
    return res


@router.post('/nearest_neighbor/concept', response_model=GraphNearestConceptResponse)
async def compute_graph_nearest_concept(data: GraphNearestConceptRequest):
    src = data.src
    top_n = data.top_n
    task = get_concept_concept_closest_task.s(src, top_n)
    res = task.apply_async(priority=6).get(timeout=30)
    return res
