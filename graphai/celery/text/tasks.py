import time

import pandas as pd
import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars

from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded

from elasticsearch_interface.es import ESConceptDetection

from graphai.core.common.config import config
from graphai.core.common.common_utils import strtobool
from graphai.core.common.logging import get_logger

from typing import Optional

logger = get_logger('graphai.celery.text')


def _bind_request_id(request_id: Optional[str]) -> None:
    """Clear stale contextvars and bind the API request id for this task."""
    clear_contextvars()
    if request_id:
        bind_contextvars(request_id=request_id)


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
def text_init_task(self, request_id: Optional[str] = None):
    """
    Celery task that spawns and populates graph and ontology objects so that they are held in memory ready for requests to arrive.
    """

    _bind_request_id(request_id)
    start = time.perf_counter()
    logger.info('🚀 Start text_init task')

    es_config = config['elasticsearch']
    es_index = es_config.get('concept_detection_index', 'concepts_detection')
    es_timeout = es_config.get('request_timeout', 10)
    logger.debug(
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
        logger.debug('⏳ Loading concepts graph and ontology tables')
        self.graph.load_from_db()
    else:
        logger.debug('⏭️ Skipping preloading for text endpoints')

    logger.info(
        '✅ Concepts graph and ontology tables loaded',
        duration_ms=int((time.perf_counter() - start) * 1000),
    )

    return True


@shared_task(bind=True, name='text.extract_keywords')
def extract_keywords_task(self, raw_text, request_id: Optional[str] = None, **kwargs):
    _bind_request_id(request_id)
    start = time.perf_counter()
    logger.debug(
        '🔑 Extracting keywords',
        task_id=self.request.id,
        raw_text_length=len(raw_text) if raw_text else 0,
    )
    keywords = extract_keywords(raw_text, **kwargs)
    logger.info(
        '✅ Keywords extracted',
        task_id=self.request.id,
        num_keywords=len(keywords),
        duration_ms=int((time.perf_counter() - start) * 1000),
    )
    return keywords


@shared_task(bind=True, name='text.wikisearch', es=es, soft_time_limit=300000, time_limit=300000)
def wikisearch_task(self, keywords_list, request_id: Optional[str] = None, **kwargs):
    _bind_request_id(request_id)
    start = time.perf_counter()
    fraction = kwargs.get('fraction')
    method = kwargs.get('method', 'es-base')
    logger.debug(
        '🔎 Searching concepts for keyword shard',
        task_id=self.request.id,
        method=method,
        fraction=fraction,
        keywords_count=len(keywords_list) if keywords_list else 0,
    )

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
        result = DataFrameResult(df.to_dict(orient='records'), columns=list(df.columns))
        logger.info(
            '✅ Concept shard search completed',
            task_id=self.request.id,
            method=method,
            fraction=fraction,
            num_results=len(result),
            duration_ms=int((time.perf_counter() - start) * 1000),
        )
        return result
    except SoftTimeLimitExceeded:
        logger.warning(
            '⚠️ text.wikisearch exceeded soft time limit; returning empty result for this shard',
            task_id=self.request.id,
            fraction=fraction,
            keywords_count=len(keywords_list) if keywords_list else 0,
        )
        return DataFrameResult([], columns=['keywords', 'concept_id', 'concept_name', 'searchrank', 'search_score'])


@shared_task(bind=True, name='text.wiki_search', es=es, soft_time_limit=300000, time_limit=300000)
def wiki_search_task(self, search_term, limit=10, request_id: Optional[str] = None):
    _bind_request_id(request_id)
    start = time.perf_counter()
    logger.debug(
        '🔍 Searching Elasticsearch for term',
        task_id=self.request.id,
        search_term=search_term,
        limit=limit,
    )

    es_timeout = config['elasticsearch'].get('request_timeout', 300000)
    es_timeout_retries = config['elasticsearch'].get('request_timeout_retries', 300000)

    try:
        result = search_elasticsearch_http(
            search_term,
            config['elasticsearch'],
            index=config['elasticsearch'].get('concept_detection_index', 'concepts_detection'),
            limit=limit,
            timeout=es_timeout,
            timeout_retries=es_timeout_retries,
        )
        logger.info(
            '✅ Elasticsearch term search completed',
            task_id=self.request.id,
            search_term=search_term,
            num_results=len(result),
            duration_ms=int((time.perf_counter() - start) * 1000),
        )
        return result
    except SoftTimeLimitExceeded:
        logger.warning(
            '⚠️ text.wiki_search exceeded soft time limit; returning empty result',
            task_id=self.request.id,
            search_term=search_term,
        )
        return []


@shared_task(bind=True, name='text.compute_scores', graph=graph)
def compute_scores_task(self, results, request_id: Optional[str] = None, **kwargs):
    _bind_request_id(request_id)
    start = time.perf_counter()
    logger.debug(
        '🧮 Computing concept scores',
        task_id=self.request.id,
        input_shards=len(results),
        restrict_to_ontology=kwargs.get('restrict_to_ontology'),
        score_smoothing=kwargs.get('score_smoothing'),
        aggregation_coef=kwargs.get('aggregation_coef'),
        filtering_threshold=kwargs.get('filtering_threshold'),
        refresh_scores=kwargs.get('refresh_scores'),
    )

    # Unwrap DataFrameResult wrappers (or plain DataFrames for backward compat).
    dataframes = [
        r.to_dataframe() if isinstance(r, DataFrameResult) else r
        for r in results
    ]
    combined = pd.concat(dataframes, ignore_index=True)
    logger.debug(
        '📊 Aggregated shard results',
        task_id=self.request.id,
        input_rows=len(combined),
    )

    df = compute_scores(combined, graph=self.graph, **kwargs)
    result = DataFrameResult(df.to_dict(orient='records'), columns=list(df.columns))
    logger.info(
        '✅ Concept scores computed',
        task_id=self.request.id,
        num_results=len(result),
        duration_ms=int((time.perf_counter() - start) * 1000),
    )
    return result


@shared_task(bind=True, name='text.draw_ontology', graph=graph)
def draw_ontology_task(self, results, request_id: Optional[str] = None, **kwargs):
    _bind_request_id(request_id)
    start = time.perf_counter()
    level = kwargs.get('level', 2)
    logger.debug(
        '🎨 Drawing ontology SVG',
        task_id=self.request.id,
        num_results=len(results),
        level=level,
    )
    svg = draw_ontology(results, graph=self.graph, **kwargs)
    logger.info(
        '✅ Ontology SVG drawn',
        task_id=self.request.id,
        svg_size_bytes=len(svg) if isinstance(svg, (str, bytes)) else None,
        duration_ms=int((time.perf_counter() - start) * 1000),
    )
    return svg


@shared_task(bind=True, name='text.draw_graph', graph=graph)
def draw_graph_task(self, results, request_id: Optional[str] = None, **kwargs):
    _bind_request_id(request_id)
    start = time.perf_counter()
    logger.debug(
        '🕸️ Drawing graph SVG',
        task_id=self.request.id,
        num_results=len(results),
        concept_score_threshold=kwargs.get('concept_score_threshold'),
        edge_threshold=kwargs.get('edge_threshold'),
        min_component_size=kwargs.get('min_component_size'),
    )
    svg = draw_graph(results, graph=self.graph, **kwargs)
    logger.info(
        '✅ Graph SVG drawn',
        task_id=self.request.id,
        svg_size_bytes=len(svg) if isinstance(svg, (str, bytes)) else None,
        duration_ms=int((time.perf_counter() - start) * 1000),
    )
    return svg


@shared_task(bind=True, name='text.generate_exercise_task')
def generate_exercise_task(self, *args, request_id: Optional[str] = None, **kwargs):
    _bind_request_id(request_id)
    start = time.perf_counter()
    logger.debug('🎓 Generating exercise', task_id=self.request.id)
    exercise = generate_exercise(*args, **kwargs)
    logger.info(
        '✅ Exercise generated',
        task_id=self.request.id,
        duration_ms=int((time.perf_counter() - start) * 1000),
    )
    return exercise
