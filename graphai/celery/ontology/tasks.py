from celery import shared_task
from graphai.core.ontology.clustering import (
    compute_all_graphs_from_scratch,
    assign_to_categories_using_existing,
    combine_and_embed_laplacian,
    cluster_and_reassign_outliers,
    convert_cluster_labels_to_dict
)
from graphai.core.interfaces.config import config
from graphai.core.common.common_utils import strtobool
from graphai.core.ontology.data import OntologyData

ontology_data = OntologyData()


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='text_6.init_ontology', ignore_result=False,
             ontology_data_obj=ontology_data)
def ontology_init_task(self):
    # This task initialises the video celery worker by loading into memory the transcription and NLP models
    print('Start init_ontology task')

    if strtobool(config['preload'].get('ontology', 'no')):
        print('Loading ontology data...')
        self.ontology_data_obj.load_data()
    else:
        print('Skipping preloading for ontology endpoints.')

    return True


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
             name='ontology_6.cluster_parent', ignore_result=False, ontology_obj=ontology_data)
def get_cluster_parent_task(self, child_id):
    return {'parent': self.ontology_obj.get_cluster_parent(child_id)}


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.cluster_children', ignore_result=False, ontology_obj=ontology_data)
def get_cluster_children_task(self, parent_id):
    return {'children': self.ontology_obj.get_cluster_children(parent_id)}


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
def get_concept_category_similarity_task(self, concept_id, category_id, avg='linear', coeffs=(1, 4)):
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
             name='ontology_6.cluster_category_similarity_graph_task',
             ignore_result=False, ontology_data_obj=ontology_data)
def get_cluster_category_similarity_task(self, cluster_id, category_id, avg='linear', coeffs=(1, 4)):
    sim = self.ontology_data_obj.get_cluster_category_similarity(cluster_id, category_id, avg, coeffs)
    return {
        'sim': sim
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.category_category_similarity_graph_task',
             ignore_result=False, ontology_data_obj=ontology_data)
def get_category_category_similarity_task(self, category_1_id, category_2_id, avg='linear', coeffs=(1, 4)):
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
def get_concept_category_closest_task(self, concept_id, avg='log', coeffs=(1, 10), top_n=1,
                                      use_depth_3=False, return_clusters=None):
    closest, scores, d3_cat, best_clusters = (
        self.ontology_data_obj.get_concept_closest_category(concept_id, avg, coeffs, top_n,
                                                            use_depth_3=use_depth_3,
                                                            return_clusters=return_clusters)
    )
    if closest is None:
        return {
            'scores': None,
            'parent_category': None,
            'valid': False,
            'existing_label': None
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
    existing_label = self.ontology_data_obj.get_concept_parent_category(concept_id)
    return {
        'scores': result_list,
        'parent_category': d3_cat,
        'valid': scores[0] > 0,
        'existing_label': existing_label
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.cluster_closest_category_graph_task',
             ignore_result=False, ontology_data_obj=ontology_data)
def get_cluster_category_closest_task(self, cluster_id, avg='log', coeffs=(1, 10), top_n=1,
                                      use_depth_3=False):
    if isinstance(cluster_id, list):
        # If it's a list, it's assumed to be a list of concepts (i.e. a "custom" cluster)
        closest, scores, d3_cat = (
            self.ontology_data_obj.get_custom_cluster_closest_category(cluster_id, avg, coeffs, top_n,
                                                                       use_depth_3=use_depth_3)
        )
    else:
        # Otherwise, it's a single string, and represents an existing cluster
        closest, scores, d3_cat = (
            self.ontology_data_obj.get_cluster_closest_category(cluster_id, avg, coeffs, top_n,
                                                                use_depth_3=use_depth_3)
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
        existing_label = self.ontology_data_obj.get_cluster_parent(cluster_id)
    else:
        existing_label = None
    return {
        'scores': result_list,
        'parent_category': d3_cat,
        'existing_label': existing_label
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
    n_total_concepts = len(concepts_to_use)
    if n_total_concepts == 0:
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
