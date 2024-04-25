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


def graph_distance_job(src, tgt, src_type, tgt_type, avg, coeffs):
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


def concept_nearest_category_job(src, avg, coeffs, top_n, use_depth_3, return_clusters):
    assert coeffs is None or len(coeffs) == 2
    task = get_concept_category_closest_task.s(src, avg, coeffs, top_n, use_depth_3, return_clusters)
    res = task.apply_async(priority=6).get(timeout=30)
    return res


def cluster_nearest_category_job(src, avg, coeffs, top_n, use_depth_3):
    assert coeffs is None or len(coeffs) == 2
    task = get_cluster_category_closest_task.s(src, avg, coeffs, top_n, use_depth_3)
    res = task.apply_async(priority=6).get(timeout=30)
    return res


def concept_nearest_concept_job(src, top_n):
    task = get_concept_concept_closest_task.s(src, top_n)
    res = task.apply_async(priority=6).get(timeout=30)
    return res


def breakup_cluster_job(cluster_id, n_clusters):
    task = break_up_cluster_task.s(cluster_id, n_clusters)
    res = task.apply_async(priority=6).get(timeout=30)
    return res
