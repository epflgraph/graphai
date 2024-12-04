from graphai.core.ontology.clustering import (
    compute_all_graphs_from_scratch,
    combine_and_embed_laplacian,
    cluster_and_reassign_outliers,
    convert_cluster_labels_to_dict,
    assign_to_categories_using_existing
)
from db_cache_manager.db import DB
from graphai.core.common.config import config
import pandas as pd


def recompute_clusters(ontology_data_obj, n_clusters, min_n):
    concept_concept = ontology_data_obj.get_concept_concept_graphscore_table()
    concept_names = ontology_data_obj.get_ontology_concept_names_table()
    category_concept = ontology_data_obj.get_category_concept_table()
    try:
        graphs_dict, base_graph_dict, row_index_dicts, concept_index_to_name, concept_index_to_id = (
            compute_all_graphs_from_scratch(
                {'graphscore': concept_concept, 'existing': category_concept},
                concept_names)
        )
        _, embedding = combine_and_embed_laplacian(list(graphs_dict.values()))
        cluster_labels = cluster_and_reassign_outliers(embedding, n_clusters, min_n)
        result_dict = convert_cluster_labels_to_dict(cluster_labels, concept_index_to_id, concept_index_to_name)
        category_assignments, impurity_count, impurity_proportion = \
            assign_to_categories_using_existing(cluster_labels, base_graph_dict['existing'],
                                                row_index_dicts['existing'])
    except Exception as e:
        print(e)
        result_dict = None
        category_assignments = None
        impurity_count = 0
        impurity_proportion = 0
    return {'results': result_dict, 'category_assignments': category_assignments,
            'impurity_count': impurity_count, 'impurity_proportion': impurity_proportion}


def get_concept_category_closest(ontology_data_obj, concept_id, avg, coeffs, top_n, use_depth_3, return_clusters,
                                 use_embeddings=False):
    embeddings_used = False

    closest, scores, d3_cat, best_clusters = (
        ontology_data_obj.get_concept_closest_category(concept_id,
                                                       avg,
                                                       coeffs,
                                                       top_n,
                                                       use_depth_3=use_depth_3,
                                                       return_clusters=return_clusters)
    )

    if closest is None and use_embeddings:
        # Fall back to embeddings if their use is enabled AND if no closest match is found through the graph
        closest, scores, d3_cat, best_clusters = ontology_data_obj.get_concept_category_closest_embedding(
            concept_id,
            avg,
            coeffs,
            top_n,
            return_clusters=return_clusters
        )
        embeddings_used = True
    if closest is None:
        return {
            'scores': None,
            'parent_category': None,
            'valid': False,
            'existing_label': None,
            'embeddings_used': embeddings_used
        }
    result_list = list()
    for i in range(len(closest)):
        current_cat = {
            'category_id': closest[i],
            'score': scores[i],
            'rank': i + 1,
            'clusters': None
        }
        if best_clusters is not None:
            if best_clusters[i] is not None:
                current_cat['clusters'] = [
                    {
                        'cluster_id': best_clusters[i][0][j],
                        'score': best_clusters[i][1][j],
                        'rank': j + 1
                    }
                    for j in range(len(best_clusters[i][0]))
                ]
        result_list.append(current_cat)
    existing_label = ontology_data_obj.get_concept_parent_category(concept_id)
    return {
        'scores': result_list,
        'parent_category': d3_cat,
        'valid': scores[0] > 0,
        'existing_label': existing_label,
        'embeddings_used': embeddings_used
    }


def get_cluster_category_closest(ontology_data_obj, cluster_id, avg, coeffs, top_n, use_depth_3, use_embeddings=False):
    if isinstance(cluster_id, list):
        # If it's a list, it's assumed to be a list of concepts (i.e. a "custom" cluster)
        closest, scores, d3_cat = (
            ontology_data_obj.get_custom_cluster_closest_category(cluster_id, avg, coeffs, top_n,
                                                                  use_depth_3=use_depth_3)
        )
        if closest is None and use_embeddings:
            closest, scores, d3_cat = (
                ontology_data_obj.get_custom_cluster_closest_category_embedding(cluster_id, avg, coeffs, top_n)
            )
    else:
        # Otherwise, it's a single string, and represents an existing cluster
        closest, scores, d3_cat = (
            ontology_data_obj.get_cluster_closest_category(cluster_id, avg, coeffs, top_n,
                                                           use_depth_3=use_depth_3)
        )
        if closest is None and use_embeddings:
            closest, scores, d3_cat = (
                ontology_data_obj.get_cluster_closest_category_embedding(cluster_id, avg, coeffs, top_n)
            )
    if closest is None:
        return {
            'scores': None,
            'parent_category': None,
            'existing_label': None
        }
    result_list = [
        {
            'category_id': closest[i],
            'score': scores[i],
            'rank': i + 1,
        }
        for i in range(len(closest))
    ]
    if isinstance(cluster_id, str):
        existing_label = ontology_data_obj.get_cluster_parent(cluster_id)
    else:
        existing_label = None
    return {
        'scores': result_list,
        'parent_category': d3_cat,
        'existing_label': existing_label
    }


def get_concept_concept_closest(ontology_data_obj, concept_id, top_n, use_embeddings=False):
    embeddings_used = False
    closest, scores = ontology_data_obj.get_concept_closest_concept(concept_id, top_n)
    if closest is None and use_embeddings:
        closest, scores = ontology_data_obj.get_concept_closest_concept_embedding(concept_id, top_n)
        embeddings_used = True
    return {
        'closest': closest,
        'scores': scores,
        'embeddings_used': embeddings_used
    }


def break_up_cluster(ontology_data_obj, cluster_id, n_clusters):
    concepts_to_use = ontology_data_obj.get_cluster_concepts(cluster_id)
    n_total_concepts = len(concepts_to_use)
    if n_total_concepts == 0:
        return {'results': None}
    concept_concept = ontology_data_obj.get_concept_concept_graphscore_table(concepts_to_keep=concepts_to_use)
    concept_names = ontology_data_obj.get_ontology_concept_names_table(concepts_to_keep=concepts_to_use)
    try:
        graphs_dict, base_graph_dict, row_index_dicts, concept_index_to_name, concept_index_to_id = (
            compute_all_graphs_from_scratch(
                {'graphscore': concept_concept},
                concept_names)
        )
        _, embedding = combine_and_embed_laplacian(list(graphs_dict.values()),
                                                   n_dims=min([1000, max([1, int(n_total_concepts / 2)])]))
        if isinstance(n_clusters, int):
            n_clusters = [n_clusters]
        all_results = list()
        for current_n_clusters in n_clusters:
            if current_n_clusters <= n_total_concepts:
                cluster_labels = cluster_and_reassign_outliers(embedding, current_n_clusters, min_n=1)
                result_dict = convert_cluster_labels_to_dict(cluster_labels, concept_index_to_id,
                                                             concept_index_to_name)
                all_results.append({'n_clusters': current_n_clusters, 'clusters': result_dict})
            else:
                all_results.append({'n_clusters': current_n_clusters, 'clusters': None})
    except Exception as e:
        print(e)
        all_results = None
    return {'results': all_results}


def get_category_info(ontology_data_obj, cat_id):
    info = ontology_data_obj.get_ontology_category_info(cat_id)
    parent = ontology_data_obj.get_category_parent(cat_id)
    child_categories = ontology_data_obj.get_category_children(cat_id)
    clusters = ontology_data_obj.get_category_cluster_list(cat_id)
    concepts = ontology_data_obj.get_category_concept_list(cat_id)
    if concepts is not None:
        concepts = ontology_data_obj.get_concept_names_list(concepts)
    return {
        'info': info,
        'parent_category': parent,
        'child_categories': child_categories,
        'clusters': clusters,
        'concepts': concepts
    }


def get_cluster_info(ontology_data_obj, cluster_id):
    parent = ontology_data_obj.get_cluster_parent(cluster_id)
    concepts = ontology_data_obj.get_cluster_concept_list(cluster_id)
    if concepts is not None:
        concepts = ontology_data_obj.get_concept_names_list(concepts)
    return {
        'parent': parent,
        'concepts': concepts
    }


def get_concept_info(ontology_data_obj, concept_ids):
    results = list()
    for current_concept in concept_ids:
        parent_category = ontology_data_obj.get_concept_parent_category(current_concept)
        branch = ontology_data_obj.get_category_branch(parent_category)
        parent_cluster = ontology_data_obj.get_concept_parent_cluster(current_concept)
        name = ontology_data_obj.get_concept_name(current_concept)
        results.append({
            'id': current_concept,
            'name': name,
            'parent_category': parent_category,
            'branch': branch,
            'parent_cluster': parent_cluster
        })
    if len(concept_ids) == 1:
        return results[0]
    return results
