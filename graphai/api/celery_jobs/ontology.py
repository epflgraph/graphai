from celery import chain

from graphai.api.celery_tasks.ontology import (
    get_ontology_tree_task,
    get_category_info_task,
    get_category_parent_task,
    get_category_children_task,
    get_category_concepts_task,
    get_category_clusters_task,
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


def tree_job():
    print('Returning the ontology tree')
    results = get_ontology_tree_task.s().apply_async(priority=6).get(timeout=10)
    return results


def category_info_job(category_id):
    results = get_category_info_task.s(category_id).apply_async(priority=6).get(timeout=10)
    return results


def category_parent_job(category_id):
    results = get_category_parent_task.s(category_id).apply_async(priority=6).get(timeout=10)
    return results


def category_children_job(category_id, dest_type):
    if dest_type == 'category':
        task = get_category_children_task.s(category_id)
    elif dest_type == 'concept':
        task = get_category_concepts_task.s(category_id)
    else:
        task = get_category_clusters_task.s(category_id)
    results = task.apply_async(priority=6).get(timeout=10)
    results['child_type'] = dest_type
    return results


def cluster_parent_job(cluster_id):
    results = get_cluster_parent_task.s(cluster_id).apply_async(priority=6).get(timeout=10)
    return results


def cluster_children_job(cluster_id):
    results = get_cluster_children_task.s(cluster_id).apply_async(priority=6).get(timeout=10)
    results['child_type'] = 'concept'
    return results


def recompute_clusters_job(n_clusters, min_n):
    task = recompute_clusters_task.s(n_clusters, min_n)
    task = task.apply_async(priority=6)
    return task.id
