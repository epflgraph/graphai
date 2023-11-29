from celery import shared_task
import numpy as np
from graphai.api.common.ontology import ontology
from db_cache_manager.db import DB
from graphai.core.common.ontology_utils.clustering import (db_results_to_pandas_df, compute_all_graphs_from_scratch,
                                                           combine_and_embed_laplacian, cluster_and_reassign_outliers)
from graphai.core.interfaces.config_loader import load_db_config


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
             name='ontology_6.recompute_clusters', ignore_result=False, ontology_obj=ontology)
def recompute_clusters_task(self, n_clusters, min_n=None):
    # TODO move the data loading step into a separate object
    db_manager = DB(load_db_config())
    concept_names = db_results_to_pandas_df(db_manager.execute_query(
        "SELECT id, name FROM graph_ontology.Nodes_N_Concept WHERE is_ontology_concept=1"),
        ['id', 'name']
    )
    concept_concept = db_results_to_pandas_df(db_manager.execute_query(
        "SELECT from_id, to_id, score FROM graph_new.Edges_N_Concept_N_Concept_T_Undirected"),
        ['from_id', 'to_id', 'score']
    )
    category_concept = db_results_to_pandas_df(db_manager.execute_query(
        "SELECT from_id, to_id FROM graph_ontology.Edges_N_Category_N_Concept_T_ParentToChild"),
        ['from_id', 'to_id']
    )
    graphs_dict, concept_index_to_name, concept_index_to_id = (
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
    # TODO add evaluation to ensure that concepts aren't moved around between categories, or just leave it be?
    return result_dict
