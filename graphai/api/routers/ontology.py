from fastapi import APIRouter
from celery import chain

from graphai.api.schemas.ontology import (
    TreeResponse,
    RecomputeClustersRequest,
    RecomputeClustersResponse,
    GraphDistanceRequest,
    GraphDistanceResponse,
    GraphNearestNeighborRequest,
    GraphNearestNeighborResponse
)
from graphai.api.schemas.common import TaskIDResponse

from graphai.api.common.log import log
from graphai.api.celery_tasks.common import format_api_results
from graphai.api.celery_tasks.ontology import (
    get_ontology_tree_task,
    get_category_parent_task,
    get_category_children_task,
    recompute_clusters_task,
    get_concept_category_similarity_task,
    get_concept_category_closest_task,
    get_concept_concept_similarity_task,
    get_concept_concept_closest_task
)
from graphai.core.interfaces.celery_config import get_task_info

# Initialise ontology router
router = APIRouter(
    prefix='/ontology',
    tags=['ontology'],
    responses={404: {'description': 'Not found'}}
)


def ontology_tree_response_handler(id_and_results):
    full_results = get_task_info(id_and_results['id'], task_results=id_and_results['results'])
    task_results = id_and_results['results']
    if task_results is not None:
        if 'child_to_parent' in task_results:
            task_results = task_results['child_to_parent']
        else:
            task_results = None
    return format_api_results(full_results['id'], full_results['name'], full_results['status'], task_results)


@router.get('/tree', response_model=TreeResponse)
async def tree():
    log('Returning the ontology tree')
    task = (get_ontology_tree_task.s()).apply_async(priority=6)
    task_id = task.id
    try:
        results = task.get(timeout=10)
    except TimeoutError as e:
        print(e)
        results = None
    id_and_results = {'id': task_id, 'results': results}
    return ontology_tree_response_handler(id_and_results)


@router.get('/tree/parent/{category_id}', response_model=TreeResponse)
async def parent(category_id):
    log('Returning the parent of category %s' % category_id)
    task = (get_category_parent_task.s(int(category_id))).apply_async(priority=6)
    try:
        results = task.get(timeout=10)
    except TimeoutError as e:
        print(e)
        results = None
    id_and_results = {'id': task.id, 'results': results}
    return ontology_tree_response_handler(id_and_results)


@router.get('/tree/children/{category_id}', response_model=TreeResponse)
async def children(category_id):
    log('Returning the children of category %s' % category_id)
    task = (get_category_children_task.s(int(category_id))).apply_async(priority=6)
    task_id = task.id
    try:
        results = task.get(timeout=10)
    except TimeoutError as e:
        print(e)
        results = None
    id_and_results = {'id': task_id, 'results': results}
    return ontology_tree_response_handler(id_and_results)


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
    dest = data.dest
    src_type = data.src_type
    dest_type = data.dest_type
    avg = data.avg
    coeffs = data.coeffs
    assert coeffs is None or len(coeffs) == 2
    assert src_type != 'category' or dest_type != 'category'
    if src_type == 'concept' and dest_type == 'category':
        task = get_concept_category_similarity_task.s(src, dest, avg, coeffs)
    elif src_type == 'category' and dest_type == 'concept':
        task = get_concept_category_similarity_task.s(dest, src, avg, coeffs)
    else:
        task = get_concept_concept_similarity_task.s(src, dest)
    res = task.apply_async(priority=6).get(timeout=30)
    return res


@router.post('/graph_nearest_neighbor', response_model=GraphNearestNeighborResponse)
async def compute_graph_nearest_neighbor(data: GraphNearestNeighborRequest):
    src = data.src
    src_type = data.src_type
    dest_type = data.dest_type
    avg = data.avg
    coeffs = data.coeffs
    top_n = data.top_n
    assert coeffs is None or len(coeffs) == 2
    assert src_type != 'category'
    if src_type == 'concept' and dest_type == 'category':
        task = get_concept_category_closest_task.s(src, avg, coeffs, top_n)
    else:
        task = get_concept_concept_closest_task.s(src, top_n)
    res = task.apply_async(priority=6).get(timeout=30)
    return res
