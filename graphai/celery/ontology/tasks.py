from celery import shared_task

from graphai.core.common.config import config
from graphai.core.common.common_utils import strtobool
from graphai.core.ontology import (
    OntologyData,
    recompute_clusters,
    get_concept_category_closest,
    get_cluster_category_closest,
    get_concept_concept_closest,
    break_up_cluster,
    get_openalex_nearest,
)
from graphai.core.ontology.ontology import get_category_info, get_cluster_info, get_concept_info

ontology_data = OntologyData()


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
             name='ontology_6.init_ontology', ignore_result=False,
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
    return get_category_info(self.ontology_obj, cat_id)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.cluster_info', ignore_result=False, ontology_obj=ontology_data)
def get_cluster_info_task(self, cluster_id):
    return get_cluster_info(self.ontology_obj, cluster_id)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.concept_info', ignore_result=False, ontology_obj=ontology_data)
def get_concept_info_task(self, concept_ids):
    return get_concept_info(self.ontology_obj, concept_ids)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.openalex_category_nearest_topics', ignore_result=False)
def get_openalex_category_nearest_topics_task(self, category_id):
    return get_openalex_nearest(category_id=category_id)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.openalex_topic_nearest_categories', ignore_result=False)
def get_openalex_topic_nearest_categories_task(self, topic_id):
    return get_openalex_nearest(topic_id=topic_id)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.recompute_clusters', ignore_result=False, ontology_data_obj=ontology_data)
def recompute_clusters_task(self, n_clusters, min_n=None):
    return recompute_clusters(self.ontology_data_obj, n_clusters, min_n)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.concept_category_similarity_graph_task',
             ignore_result=False, ontology_data_obj=ontology_data)
def get_concept_category_similarity_task(self, concept_id, category_id, avg='linear', coeffs=(1, 4)):
    return {
        'sim': self.ontology_data_obj.get_concept_category_similarity(concept_id, category_id, avg, coeffs)
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.concept_cluster_similarity_graph_task',
             ignore_result=False, ontology_data_obj=ontology_data)
def get_concept_cluster_similarity_task(self, concept_id, cluster_id, avg='linear'):
    return {
        'sim': self.ontology_data_obj.get_concept_cluster_similarity(concept_id, cluster_id, avg)
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.cluster_cluster_similarity_graph_task',
             ignore_result=False, ontology_data_obj=ontology_data)
def get_cluster_cluster_similarity_task(self, cluster_1_id, cluster_2_id, avg='linear'):
    return {
        'sim': self.ontology_data_obj.get_cluster_cluster_similarity(cluster_1_id, cluster_2_id, avg)
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.cluster_category_similarity_graph_task',
             ignore_result=False, ontology_data_obj=ontology_data)
def get_cluster_category_similarity_task(self, cluster_id, category_id, avg='linear', coeffs=(1, 4)):
    return {
        'sim': self.ontology_data_obj.get_cluster_category_similarity(cluster_id, category_id, avg, coeffs)
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.category_category_similarity_graph_task',
             ignore_result=False, ontology_data_obj=ontology_data)
def get_category_category_similarity_task(self, category_1_id, category_2_id, avg='linear', coeffs=(1, 4)):
    return {
        'sim': self.ontology_data_obj.get_category_category_similarity(category_1_id, category_2_id, avg, coeffs)
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.concept_concept_similarity_graph_task',
             ignore_result=False, ontology_data_obj=ontology_data)
def get_concept_concept_similarity_task(self, concept_1_id, concept_2_id):
    return {
        'sim': self.ontology_data_obj.get_concept_concept_similarity(concept_1_id, concept_2_id)
    }


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.concept_closest_category_graph_task',
             ignore_result=False, ontology_data_obj=ontology_data)
def get_concept_category_closest_task(self, concept_id, avg='log', coeffs=(1, 10), top_n=1,
                                      use_depth_3=False, return_clusters=None):
    return get_concept_category_closest(self.ontology_data_obj, concept_id, avg, coeffs,
                                        top_n, use_depth_3, return_clusters)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.cluster_closest_category_graph_task',
             ignore_result=False, ontology_data_obj=ontology_data)
def get_cluster_category_closest_task(self, cluster_id, avg='log', coeffs=(1, 10), top_n=1,
                                      use_depth_3=False):
    return get_cluster_category_closest(self.ontology_data_obj, cluster_id, avg, coeffs, top_n, use_depth_3)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.concept_closest_concept_graph_task',
             ignore_result=False, ontology_data_obj=ontology_data)
def get_concept_concept_closest_task(self, concept_id, top_n=1):
    return get_concept_concept_closest(self.ontology_data_obj, concept_id, top_n)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='ontology_6.break_up_cluster', ignore_result=False, ontology_data_obj=ontology_data)
def break_up_cluster_task(self, cluster_id, n_clusters=2):
    return break_up_cluster(self.ontology_data_obj, cluster_id, n_clusters)
