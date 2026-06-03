import pandas as pd

from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded

from elasticsearch_interface.es import ESConceptDetection

from graphai.core.common.config import config
from graphai.core.common.common_utils import strtobool

from graphai.core.text import (
    ConceptsGraph,
    extract_keywords,
    wikisearch,
    compute_scores,
    draw_ontology,
    draw_graph,
    generate_exercise,
)
from graphai.core.text.wikisearch import search_elasticsearch_http


################################################################
# Objects shared across tasks                                  #
################################################################

# Object that holds all graph and ontology data in memory
graph = ConceptsGraph()


# Elasticsearch interface
es = ESConceptDetection(
    config['elasticsearch'],
    index=config['elasticsearch'].get('concept_detection_index', 'concepts_detection'),
)


################################################################
# Tasks                                                        #
################################################################


@shared_task(bind=True, name='text.init', graph=graph)
def text_init_task(self):
    """
    Celery task that spawns and populates graph and ontology objects so that they are held in memory ready for requests to arrive.
    """

    # This task initialises the text celery worker by loading into memory the graph and ontology tables
    print('Start text_init task')

    if strtobool(config['preload']['text']):
        print('Loading concepts graph and ontology tables...')
        self.graph.load_from_db()
    else:
        print('Skipping preloading for text endpoints')

    print('Concepts graph and ontology tables loaded')

    return True


@shared_task(bind=True, name='text.extract_keywords')
def extract_keywords_task(self, raw_text, **kwargs):
    return extract_keywords(raw_text, **kwargs)


@shared_task(bind=True, name='text.wikisearch', es=es, soft_time_limit=30, time_limit=45)
def wikisearch_task(self, keywords_list, **kwargs):
    es_timeout = config['elasticsearch'].get('request_timeout', 10)
    es_timeout_retries = config['elasticsearch'].get('request_timeout_retries', 2)
    wikipedia_timeout = config['elasticsearch'].get('wikipedia_timeout', 6)

    try:
        return wikisearch(
            keywords_list,
            es=self.es,
            es_timeout=es_timeout,
            es_timeout_retries=es_timeout_retries,
            wikipedia_timeout=wikipedia_timeout,
            **kwargs,
        )
    except SoftTimeLimitExceeded:
        print('[WARNING] text.wikisearch exceeded soft time limit; returning empty result for this shard.')
        return pd.DataFrame()


@shared_task(bind=True, name='text.wiki_search', es=es, soft_time_limit=30, time_limit=45)
def wiki_search_task(self, search_term, limit=10):
    es_timeout = config['elasticsearch'].get('request_timeout', 10)
    es_timeout_retries = config['elasticsearch'].get('request_timeout_retries', 2)

    try:
        return search_elasticsearch_http(
            search_term,
            config['elasticsearch'],
            index=config['elasticsearch'].get('concept_detection_index', 'concepts_detection'),
            limit=limit,
            timeout=es_timeout,
            timeout_retries=es_timeout_retries,
        )
    except SoftTimeLimitExceeded:
        print('[WARNING] text.wiki_search exceeded soft time limit; returning empty result.')
        return []


@shared_task(bind=True, name='text.compute_scores', graph=graph)
def compute_scores_task(self, results, **kwargs):
    return compute_scores(pd.concat(results, ignore_index=True), graph=self.graph, **kwargs)


@shared_task(bind=True, name='text.draw_ontology', graph=graph)
def draw_ontology_task(self, results, **kwargs):
    return draw_ontology(results, graph=self.graph, **kwargs)


@shared_task(bind=True, name='text.draw_graph', graph=graph)
def draw_graph_task(self, results, **kwargs):
    return draw_graph(results, graph=self.graph, **kwargs)


@shared_task(bind=True, name='text.generate_exercise_task')
def generate_exercise_task(self, *args, **kwargs):
    return generate_exercise(*args, **kwargs)
