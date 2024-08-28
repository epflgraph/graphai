import pandas as pd

from celery import shared_task

from elasticsearch_interface.es import ES

from graphai.core.common.config import config
from graphai.core.common.common_utils import strtobool

from graphai.core.text import (
    ConceptsGraph,
    extract_keywords,
    wikisearch,
    compute_scores,
    draw_ontology,
    draw_graph,
)


################################################################
# Objects shared across tasks                                  #
################################################################

# Object that holds all graph and ontology data in memory
graph = ConceptsGraph()

# Elasticsearch interface
es = ES(config['elasticsearch'], index=config['elasticsearch'].get('concept_detection_index', 'concepts_detection'))


################################################################
# Tasks                                                        #
################################################################


@shared_task(bind=True, name='text_10.init', graph=graph)
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


@shared_task(bind=True, name='text_10.extract_keywords')
def extract_keywords_task(self, raw_text, **kwargs):
    return extract_keywords(raw_text, **kwargs)


@shared_task(bind=True, name='text_10.wikisearch', es=es)
def wikisearch_task(self, keywords_list, **kwargs):
    return wikisearch(keywords_list, es=self.es, **kwargs)


@shared_task(bind=True, name='text_10.compute_scores', graph=graph)
def compute_scores_task(self, results, **kwargs):
    return compute_scores(pd.concat(results, ignore_index=True), graph=self.graph, **kwargs)


@shared_task(bind=True, name='text_10.draw_ontology', graph=graph)
def draw_ontology_task(self, results, **kwargs):
    return draw_ontology(results, graph=self.graph, **kwargs)


@shared_task(bind=True, name='text_10.draw_graph', graph=graph)
def draw_graph_task(self, results, **kwargs):
    return draw_graph(results, graph=self.graph, **kwargs)
