from celery import shared_task
import numpy as np
from graphai.api.common.ontology import ontology_data
from graphai.core.common.ontology_utils.clustering import (
    compute_all_graphs_from_scratch, assign_to_categories_using_existing,
    combine_and_embed_laplacian, cluster_and_reassign_outliers, convert_cluster_labels_to_dict
)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.tree', ignore_result=False, ontology_obj=ontology_data)
def get_ontology_tree_task(self):
    return {'child_to_parent': self.ontology_obj.get_category_to_category()}


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.category_info', ignore_result=False, ontology_obj=ontology_data)
def get_category_info_task(self, cat_id):
    return self.ontology_obj.get_ontology_category_info(cat_id)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.category_parent', ignore_result=False, ontology_obj=ontology_data)
def get_category_parent_task(self, child_id):
    return {'parent': self.ontology_obj.get_category_parent(child_id)}


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.category_children', ignore_result=False, ontology_obj=ontology_data)
def get_category_children_task(self, parent_id):
    return {'children': self.ontology_obj.get_category_children(parent_id)}


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.category_concepts', ignore_result=False, ontology_obj=ontology_data)
def get_category_concepts_task(self, parent_id):
    return {'children': self.ontology_obj.get_category_concept_list(parent_id)}


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.category_clusters', ignore_result=False, ontology_obj=ontology_data)
def get_category_clusters_task(self, parent_id):
    return {'children': self.ontology_obj.get_category_cluster_list(parent_id)}


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.recompute_clusters', ignore_result=False, ontology_data_obj=ontology_data)
def recompute_clusters_task(self, n_clusters, min_n=None):
    concept_concept = self.ontology_data_obj.get_concept_concept_graphscore_table()
    concept_names = self.ontology_data_obj.get_ontology_concept_names_table()
    category_concept = self.ontology_data_obj.get_category_concept_table()
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


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.concept_category_similarity_graph_task',
             ignore_result=False, ontology_data_obj=ontology_data)
def get_concept_category_similarity_task(self, concept_id, category_id, avg='linear', coeffs=(1, 1)):
    sim = self.ontology_data_obj.get_concept_category_similarity(concept_id, category_id, avg, coeffs)
    return {
        'sim': sim
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.concept_cluster_similarity_graph_task',
             ignore_result=False, ontology_data_obj=ontology_data)
def get_concept_cluster_similarity_task(self, concept_id, cluster_id, avg='linear'):
    sim = self.ontology_data_obj.get_concept_cluster_similarity(concept_id, cluster_id, avg)
    return {
        'sim': sim
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.cluster_cluster_similarity_graph_task',
             ignore_result=False, ontology_data_obj=ontology_data)
def get_cluster_cluster_similarity_task(self, cluster_1_id, cluster_2_id, avg='linear'):
    sim = self.ontology_data_obj.get_cluster_cluster_similarity(cluster_1_id, cluster_2_id, avg)
    return {
        'sim': sim
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.category_category_similarity_graph_task',
             ignore_result=False, ontology_data_obj=ontology_data)
def get_category_category_similarity_task(self, category_1_id, category_2_id, avg='linear', coeffs=(1, 1)):
    sim = self.ontology_data_obj.get_category_category_similarity(category_1_id, category_2_id, avg, coeffs)
    return {
        'sim': sim
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.concept_concept_similarity_graph_task',
             ignore_result=False, ontology_data_obj=ontology_data)
def get_concept_concept_similarity_task(self, concept_1_id, concept_2_id):
    sim = self.ontology_data_obj.get_concept_concept_similarity(concept_1_id, concept_2_id)
    return {
        'sim': sim
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.concept_closest_category_graph_task',
             ignore_result=False, ontology_data_obj=ontology_data)
def get_concept_category_closest_task(self, concept_id, avg='linear', coeffs=(1, 1), top_n=1,
                                      use_depth_3=False, return_clusters=False):
    closest, scores, d3_cat, best_clusters = (
        self.ontology_data_obj.get_concept_closest_category(concept_id, avg, coeffs, top_n,
                                                            use_depth_3=use_depth_3,
                                                            return_clusters=return_clusters)
    )
    result_dict = list()
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
        result_dict.append(current_cat)
    return {
        'scores': result_dict,
        'parent_category': d3_cat,
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.concept_closest_concept_graph_task',
             ignore_result=False, ontology_data_obj=ontology_data)
def get_concept_concept_closest_task(self, concept_id, top_n=1):
    closest, scores = self.ontology_data_obj.get_concept_closest_concept(concept_id, top_n)
    return {
        'closest': closest,
        'scores': scores
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.break_up_cluster', ignore_result=False, ontology_data_obj=ontology_data)
def break_up_cluster_task(self, cluster_id, n_clusters=2):
    concepts_to_use = self.ontology_data_obj.get_cluster_concepts(cluster_id)
    if len(concepts_to_use) == 0:
        return {'results': None}
    concept_concept = self.ontology_data_obj.get_concept_concept_graphscore_table(concepts_to_keep=concepts_to_use)
    concept_names = self.ontology_data_obj.get_ontology_concept_names_table(concepts_to_keep=concepts_to_use)
    try:
        graphs_dict, base_graph_dict, row_index_dicts, concept_index_to_name, concept_index_to_id = (
            compute_all_graphs_from_scratch(
                {'graphscore': concept_concept},
                concept_names)
        )
        _, embedding = combine_and_embed_laplacian(list(graphs_dict.values()),
                                                   n_dims=min([1000, max([1, int(len(concepts_to_use)/2)])]))
        if isinstance(n_clusters, int):
            n_clusters = [n_clusters]
        all_results = list()
        for current_n_clusters in n_clusters:
            cluster_labels = cluster_and_reassign_outliers(embedding, current_n_clusters, min_n=1)
            result_dict = convert_cluster_labels_to_dict(cluster_labels, concept_index_to_id, concept_index_to_name)
            all_results.append({'n_clusters': current_n_clusters, 'clusters': result_dict})
    except Exception as e:
        print(e)
        all_results = None
    return {'results': all_results}
