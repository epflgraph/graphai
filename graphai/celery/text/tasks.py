import pandas as pd

from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded

from elasticsearch_interface.es import ESConceptDetection

from graphai.core.common.config import config
from graphai.core.common.common_utils import strtobool
from graphai.core.common.logging import get_logger

logger = get_logger('graphai.celery.text')


class DataFrameResult:
    """Lightweight wrapper around a DataFrame's records for Celery serialization.

    Celery logs task return values via ``repr()``.  A raw pandas DataFrame produces
    multi-line dumps such as ``[10 rows x 13 columns]`` and individual row lines.
    This wrapper stores the records and renders as a single short line, while still
    being pickle-serializable and cheap to unwrap back into a DataFrame.
    """

    def __init__(self, records, columns=None):
        self.records = records
        self.columns = columns

    def to_dataframe(self):
        if not self.records:
            return pd.DataFrame(columns=self.columns)
        return pd.DataFrame(self.records)

    def __repr__(self):
        n = len(self.records)
        m = len(self.columns) if self.columns else 0
        return f'<DataFrameResult: {n} rows x {m} columns>'

    def __len__(self):
        return len(self.records)

from graphai.core.text import (
    ConceptsGraph,
    extract_keywords,
    wikisearch,
    compute_scores,
    draw_ontology,
    draw_graph,
    generate_exercise,
)
from graphai.core.text.wikisearch import search_elasticsearch_http, validate_elasticsearch_index_http


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
    logger.info('🚀 Start text_init task')

    es_config = config['elasticsearch']
    es_index = es_config.get('concept_detection_index', 'concepts_detection')
    es_timeout = es_config.get('request_timeout', 10)
    logger.info(
        '🔍 Validating Elasticsearch concept detection index',
        host=es_config['host'],
        port=es_config['port'],
        index=es_index,
    )
    es_doc_count = validate_elasticsearch_index_http(es_config, es_index, timeout=es_timeout)
    logger.info(
        '✅ Elasticsearch concept detection index is reachable',
        index=es_index,
        document_count=es_doc_count,
    )

    if strtobool(config['preload']['text']):
        logger.info('⏳ Loading concepts graph and ontology tables')
        self.graph.load_from_db()
    else:
        logger.info('⏭️ Skipping preloading for text endpoints')

    logger.info('✅ Concepts graph and ontology tables loaded')

    return True


@shared_task(bind=True, name='text.extract_keywords')
def extract_keywords_task(self, raw_text, **kwargs):
    return extract_keywords(raw_text, **kwargs)


@shared_task(bind=True, name='text.wikisearch', es=es, soft_time_limit=300000, time_limit=300000)
def wikisearch_task(self, keywords_list, **kwargs):
    es_timeout = config['elasticsearch'].get('request_timeout', 300000)
    es_timeout_retries = config['elasticsearch'].get('request_timeout_retries', 300000)
    wikipedia_timeout = config['elasticsearch'].get('wikipedia_timeout', 300000)

    try:
        df = wikisearch(
            keywords_list,
            es=self.es,
            es_timeout=es_timeout,
            es_timeout_retries=es_timeout_retries,
            wikipedia_timeout=wikipedia_timeout,
            **kwargs,
        )
        return DataFrameResult(df.to_dict(orient='records'), columns=list(df.columns))
    except SoftTimeLimitExceeded:
        logger.warning(
            '⚠️ text.wikisearch exceeded soft time limit; returning empty result for this shard',
            keywords_count=len(keywords_list),
        )
        return DataFrameResult([], columns=['keywords', 'concept_id', 'concept_name', 'searchrank', 'search_score'])


@shared_task(bind=True, name='text.wiki_search', es=es, soft_time_limit=300000, time_limit=300000)
def wiki_search_task(self, search_term, limit=10):
    es_timeout = config['elasticsearch'].get('request_timeout', 300000)
    es_timeout_retries = config['elasticsearch'].get('request_timeout_retries', 300000)

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
        logger.warning(
            '⚠️ text.wiki_search exceeded soft time limit; returning empty result',
            search_term=search_term,
        )
        return []


@shared_task(bind=True, name='text.compute_scores', graph=graph)
def compute_scores_task(self, results, **kwargs):
    # Unwrap DataFrameResult wrappers (or plain DataFrames for backward compat).
    dataframes = [
        r.to_dataframe() if isinstance(r, DataFrameResult) else r
        for r in results
    ]
    df = compute_scores(pd.concat(dataframes, ignore_index=True), graph=self.graph, **kwargs)
    return DataFrameResult(df.to_dict(orient='records'), columns=list(df.columns))


@shared_task(bind=True, name='text.draw_ontology', graph=graph)
def draw_ontology_task(self, results, **kwargs):
    return draw_ontology(results, graph=self.graph, **kwargs)


@shared_task(bind=True, name='text.draw_graph', graph=graph)
def draw_graph_task(self, results, **kwargs):
    return draw_graph(results, graph=self.graph, **kwargs)


@shared_task(bind=True, name='text.generate_exercise_task')
def generate_exercise_task(self, *args, **kwargs):
    return generate_exercise(*args, **kwargs)
