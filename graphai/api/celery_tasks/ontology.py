from celery import shared_task
import numpy as np
from graphai.api.common.ontology import ontology, ontology_data
from graphai.core.common.ontology_utils.clustering import (
    compute_all_graphs_from_scratch, assign_to_categories_using_existing,
    combine_and_embed_laplacian, cluster_and_reassign_outliers
)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.tree', ignore_result=False, ontology_obj=ontology)
def get_ontology_tree_task(self):
    return {'child_to_parent': self.ontology_obj.get_predefined_tree()}


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.parent', ignore_result=False, ontology_obj=ontology)
def get_category_parent_task(self, child_id):
    return {'child_to_parent': self.ontology_obj.get_category_parent(child_id)}


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.children', ignore_result=False, ontology_obj=ontology)
def get_category_children_task(self, parent_id):
    return {'child_to_parent': self.ontology_obj.get_category_children(parent_id)}


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.recompute_clusters', ignore_result=False, ontology_data_obj=ontology_data)
def recompute_clusters_task(self, n_clusters, min_n=None):
    concept_concept = self.ontology_data_obj.get_concept_concept_graphscore()
    concept_names = self.ontology_data_obj.get_ontology_concept_names()
    category_concept = self.ontology_data_obj.get_category_concept()
    try:
        graphs_dict, base_graph_dict, row_index_dicts, concept_index_to_name, concept_index_to_id = (
            compute_all_graphs_from_scratch(
                {'graphscore': concept_concept, 'existing': category_concept},
                concept_names)
        )
        _, embedding = combine_and_embed_laplacian(list(graphs_dict.values()))
        cluster_labels = cluster_and_reassign_outliers(embedding, n_clusters, min_n)
        unique_cluster_labels = sorted(list(set(cluster_labels.tolist())))
        result_dict = {
            label: [{'name': concept_index_to_name[i], 'id': concept_index_to_id[i]}
                    for i in np.where(cluster_labels == label)[0]]
            for label in unique_cluster_labels
        }
        category_assignments, impurity_count, impurity_proportion = \
            assign_to_categories_using_existing(cluster_labels, base_graph_dict['existing'],
                                                row_index_dicts['existing'])
    except Exception as e:
        print(e)
        result_dict = None
        category_assignments = None
        impurity_count = 0
        impurity_proportion = 0
    # TODO add evaluation to ensure that concepts aren't moved around between categories, or just leave it be?
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
def get_concept_category_closest_task(self, concept_id, avg='linear', coeffs=(1, 1), top_n=1):
    closest, scores = self.ontology_data_obj.get_concept_closest_category(concept_id, avg, coeffs, top_n)
    return {
        'closest': closest,
        'scores': scores
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
