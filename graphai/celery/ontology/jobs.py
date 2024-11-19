import graphai.celery.ontology.tasks as tasks


def tree_job():
    print('Returning the ontology tree')
    results = tasks.get_ontology_tree_task.s().apply_async(priority=6).get(timeout=10)
    return results


def category_info_job(category_id):
    results = tasks.get_category_info_task.s(category_id).apply_async(priority=6).get(timeout=10)
    return results


def cluster_info_job(cluster_id):
    results = tasks.get_cluster_info_task.s(cluster_id).apply_async(priority=6).get(timeout=10)
    return results


def concept_info_job(concept_ids):
    results = tasks.get_concept_info_task.s(concept_ids).apply_async(priority=6).get(timeout=10)
    return results


def openalex_category_nearest_topics_job(category_id):
    job = tasks.get_openalex_category_nearest_topics_task.s(category_id)
    return job.apply_async(priority=6).get(timeout=10)


def openalex_topic_nearest_categories_job(topic_id):
    job = tasks.get_openalex_topic_nearest_categories_task.s(topic_id)
    return job.apply_async(priority=6).get(timeout=10)


def recompute_clusters_job(n_clusters, min_n):
    task = tasks.recompute_clusters_task.s(n_clusters, min_n)
    task = task.apply_async(priority=6)
    return task.id


def graph_distance_job(src, tgt, src_type, tgt_type, avg, coeffs):
    assert coeffs is None or len(coeffs) == 2
    if src_type == 'concept' and tgt_type == 'category':
        task = tasks.get_concept_category_similarity_task.s(src, tgt, avg, coeffs)
    elif src_type == 'category' and tgt_type == 'concept':
        task = tasks.get_concept_category_similarity_task.s(tgt, src, avg, coeffs)
    elif src_type == 'concept' and tgt_type == 'cluster':
        task = tasks.get_concept_cluster_similarity_task.s(src, tgt, avg)
    elif src_type == 'cluster' and tgt_type == 'concept':
        task = tasks.get_concept_cluster_similarity_task.s(tgt, src, avg)
    elif src_type == 'cluster' and tgt_type == 'cluster':
        task = tasks.get_cluster_cluster_similarity_task.s(src, tgt, avg)
    elif src_type == 'cluster' and tgt_type == 'category':
        task = tasks.get_cluster_category_similarity_task.s(src, tgt, avg, coeffs)
    elif src_type == 'category' and tgt_type == 'cluster':
        task = tasks.get_cluster_category_similarity_task.s(tgt, src, avg, coeffs)
    elif src_type == 'category' and tgt_type == 'category':
        task = tasks.get_category_category_similarity_task.s(src, tgt, avg, coeffs)
    else:
        task = tasks.get_concept_concept_similarity_task.s(src, tgt)
    res = task.apply_async(priority=6).get(timeout=30)
    return res


def concept_nearest_category_job(src, avg, coeffs, top_n, use_depth_3, return_clusters, use_embeddings=False):
    assert coeffs is None or len(coeffs) == 2
    task = tasks.get_concept_category_closest_task.s(src, avg, coeffs, top_n, use_depth_3, return_clusters,
                                                     use_embeddings)
    res = task.apply_async(priority=6).get(timeout=30)
    return res


def cluster_nearest_category_job(src, avg, coeffs, top_n, use_depth_3):
    assert coeffs is None or len(coeffs) == 2
    task = tasks.get_cluster_category_closest_task.s(src, avg, coeffs, top_n, use_depth_3)
    res = task.apply_async(priority=6).get(timeout=30)
    return res


def concept_nearest_concept_job(src, top_n):
    task = tasks.get_concept_concept_closest_task.s(src, top_n)
    res = task.apply_async(priority=6).get(timeout=30)
    return res


def breakup_cluster_job(cluster_id, n_clusters):
    task = tasks.break_up_cluster_task.s(cluster_id, n_clusters)
    res = task.apply_async(priority=6).get(timeout=30)
    return res
