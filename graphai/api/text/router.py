import time
import uuid
from typing import Optional, Union

from fastapi import APIRouter, HTTPException, Query, Security, Request
from fastapi.responses import FileResponse

import pandas as pd

from graphai.api.auth.router import get_current_active_user
from graphai.core.common.logging import get_logger
from graphai.core.text.wikisearch import ElasticsearchSearchError
import graphai.api.text.schemas as schemas
import graphai.celery.text.jobs as jobs


def _request_id(request: Request) -> str:
    """Return an existing X-Request-ID header or generate a short correlation id."""
    return request.headers.get('x-request-id') or uuid.uuid4().hex[:8]


def _duration_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)

pd.set_option('display.max_rows', 400)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

logger = get_logger('graphai.api.text')

# Initialise text router
router = APIRouter(
    prefix='/text',
    tags=['text'],
    responses={404: {'description': 'Not found'}},
    dependencies=[Security(get_current_active_user, scopes=['text'])]
)


@router.post('/keywords', response_model=schemas.KeywordsResponse)
async def keywords(
    request: Request,
    data: schemas.KeywordsRequest,
    use_nltk: Optional[bool] = False,
):
    """
    Processes raw text (e.g. from an abstract of a publication, a course description or a lecture slide) and returns a
    list of keywords from the text.
    """

    start = time.perf_counter()
    log = logger.bind(
        endpoint='/text/keywords',
        method=request.method,
        request_id=_request_id(request),
        use_nltk=use_nltk,
    )

    if not data.raw_text:
        log.warning('⚠️ Empty raw text received; returning empty result')
        return []

    log.debug('📝 Extracting keywords', raw_text_length=len(data.raw_text))
    result = jobs.keywords(data.raw_text, use_nltk, request_id=_request_id(request))
    log.info(
        '✅ Keywords extracted',
        num_keywords=len(result),
        duration_ms=_duration_ms(start),
    )
    return result


@router.post('/wiki_search', response_model=schemas.WikiSearchResponse)
async def wiki_search(
    request: Request,
    data: schemas.WikiSearchRequest,
    limit: Optional[int] = Query(default=10, ge=1, le=100),
):
    """
    Searches the Wikipedia concept-detection Elasticsearch index and returns the ordered matches.
    """

    start = time.perf_counter()
    search_term = data.search_term.strip()
    log = logger.bind(
        endpoint='/text/wiki_search',
        method=request.method,
        request_id=_request_id(request),
        search_term=search_term,
        limit=limit,
    )

    if not search_term:
        log.warning('⚠️ Empty search term received; returning empty result')
        return []

    log.debug('🔍 Searching Wikipedia concept index')
    try:
        result = jobs.wiki_search(search_term, limit, request_id=_request_id(request))
        log.info(
            '✅ Wiki search completed',
            num_results=len(result),
            duration_ms=_duration_ms(start),
        )
        return result
    except ElasticsearchSearchError as exc:
        log.error(
            '❌ Wiki search failed',
            status_code=exc.api_status_code,
            upstream_status=exc.upstream_status,
            error=str(exc),
            duration_ms=_duration_ms(start),
        )
        raise HTTPException(status_code=exc.api_status_code, detail=str(exc)) from exc


@router.post('/wikify', response_model=schemas.WikifyResponse)
async def wikify(
    request: Request,
    data: Union[schemas.WikifyFromRawTextRequest, schemas.WikifyFromKeywordsRequest],
    method: Optional[str] = 'es-base',
    restrict_to_ontology: Optional[bool] = False,
    score_smoothing: Optional[bool] = True,
    aggregation_coef: Optional[float] = 0.5,
    filtering_threshold: Optional[float] = 0.15,
    refresh_scores: Optional[bool] = True,
):
    """
    Processes raw text (e.g. from an abstract of a publication, a course description or a lecture slide) and returns a
    list of concepts (Wikipedia pages) that are relevant to the text, each with a set of scores in [0, 1]
    quantifying their relevance. This is done as follows:

    1. Keyword extraction: Automatic extraction of keywords from the text. Omitted if a list of strings is provided as input
        under "keywords" instead of "raw_text".
    2. Wikisearch: For each set of keywords, a set of at most 10 concepts (Wikipedia pages) is retrieved. This can be
        done through requests to the Wikipedia API or through elasticsearch requests.
    3. Scores: For each pair of keywords and concept, several scores are derived, taking into account the concepts graph,
        the ontology and embedding vectors, among others.
    4. Aggregation and filter: For each concept, their scores are aggregated and filtered according to some rules,
        to keep only the most relevant results.

    Several arguments can be passed to have a more precise control:
    * method (str): Method to retrieve the concepts (Wikipedia pages). It can be either 'wikipedia-api', to use the
    Wikipedia API, or one of {'es-base', 'es-score'}, to use elasticsearch. Default: 'es-base'.
    * restrict_to_ontology (bool): Whether to filter concepts that are not in the ontology. Default: False.
    * score_smoothing (bool): Whether to apply a transformation to some scores to distribute them more evenly in [0, 1]. Default: True.
    * aggregation_coef (float): A number in [0, 1] that controls how the scores of the aggregated pages are computed.
    A value of 0 takes the sum of scores over Keywords, then normalises in [0, 1]. A value of 1 takes the max of scores over Keywords.
    Any value in between linearly interpolates those two approaches. Default: 0.5.
    * filtering_threshold (float): A number in [0, 1] that is used as a threshold for all the scores to decide whether the page is good enough
    from that score's perspective. Default: 0.15.
    * refresh_scores (bool): Whether to recompute scores after filtering. Default: True.
    """

    start = time.perf_counter()
    log = logger.bind(
        endpoint=str(request.url.path),
        method=request.method,
        request_id=_request_id(request),
        wiki_method=method,
        restrict_to_ontology=restrict_to_ontology,
        score_smoothing=score_smoothing,
        aggregation_coef=aggregation_coef,
        filtering_threshold=filtering_threshold,
        refresh_scores=refresh_scores,
    )
    log.info('🚀 Wikify endpoint invoked')

    if isinstance(data, schemas.WikifyFromRawTextRequest):
        input_type = 'raw_text'
        raw_text_len = len(data.raw_text)
        log = log.bind(input_type=input_type, raw_text_length=raw_text_len)

        # Return if no input
        if not data.raw_text:
            log.warning('⚠️ Empty raw text received; returning empty result')
            return []

        log.debug('📝 Starting wikify from raw text')
        try:
            results = jobs.wikify_text(
                data.raw_text,
                method,
                restrict_to_ontology,
                score_smoothing,
                aggregation_coef,
                filtering_threshold,
                refresh_scores,
                request_id=_request_id(request),
            )
            log.info(
                '✅ Wikify from raw text completed',
                num_results=len(results),
                duration_ms=_duration_ms(start),
            )
            return results
        except ElasticsearchSearchError as exc:
            log.error(
                '❌ Elasticsearch search failed while wikifying raw text',
                status_code=exc.api_status_code,
                upstream_status=exc.upstream_status,
                error=str(exc),
                duration_ms=_duration_ms(start),
            )
            raise HTTPException(status_code=exc.api_status_code, detail=str(exc)) from exc

    if isinstance(data, schemas.WikifyFromKeywordsRequest):
        input_type = 'keywords'
        keywords_count = len(data.keywords)
        log = log.bind(input_type=input_type, keywords_count=keywords_count)

        # Return if no input
        if not data.keywords:
            log.warning('⚠️ Empty keyword list received; returning empty result')
            return []

        # Remove duplicate keywords
        keyword_list = list(set(data.keywords))
        deduped_count = len(keyword_list)
        log.debug(
            '🔑 Starting wikify from keywords',
            unique_keywords_count=deduped_count,
            duplicate_keywords_removed=keywords_count - deduped_count,
        )

        try:
            results = jobs.wikify_keywords(
                keyword_list,
                method,
                restrict_to_ontology,
                score_smoothing,
                aggregation_coef,
                filtering_threshold,
                refresh_scores,
                request_id=_request_id(request),
            )
            log.info(
                '✅ Wikify from keywords completed',
                num_results=len(results),
                duration_ms=_duration_ms(start),
            )
            return results
        except ElasticsearchSearchError as exc:
            log.error(
                '❌ Elasticsearch search failed while wikifying keywords',
                status_code=exc.api_status_code,
                upstream_status=exc.upstream_status,
                error=str(exc),
                duration_ms=_duration_ms(start),
            )
            raise HTTPException(status_code=exc.api_status_code, detail=str(exc)) from exc

    log.warning('⚠️ Unrecognized wikify input type; returning empty result')
    return []


@router.post('/wikify_ontology_svg')
async def wikify_ontology_svg(
    request: Request,
    results: schemas.WikifyResponse,
    level: Optional[int] = 2,
):
    """
    Returns a svg file representing the ontology subgraph induced by the provided set of results.
    """

    start = time.perf_counter()
    log = logger.bind(
        endpoint='/text/wikify_ontology_svg',
        method=request.method,
        request_id=_request_id(request),
        num_results=len(results),
        level=level,
    )

    # Convert WikifyResponseElems into dictionaries
    results = [vars(result) for result in results]

    # Switch to default level if not properly defined
    if level not in [1, 2, 3, 4, 5]:
        level = 2
        log.debug('⚙️ Invalid level provided; defaulting to 2')

    log.debug('🎨 Generating ontology SVG')
    jobs.wikify_ontology_svg(results, level, request_id=_request_id(request))
    log.info('✅ Ontology SVG generated', duration_ms=_duration_ms(start))

    # Return svg file
    return FileResponse('/tmp/file.svg')


@router.post('/wikify_graph_svg')
async def wikify_graph_svg(
    request: Request,
    results: schemas.WikifyResponse,
    concept_score_threshold: Optional[float] = 0.3,
    edge_threshold: Optional[float] = 0.3,
    min_component_size: Optional[int] = 3,
):
    """
    Returns a svg file representing the graph subgraph induced by the provided set of results.
    """

    start = time.perf_counter()
    log = logger.bind(
        endpoint='/text/wikify_graph_svg',
        method=request.method,
        request_id=_request_id(request),
        num_results=len(results),
        concept_score_threshold=concept_score_threshold,
        edge_threshold=edge_threshold,
        min_component_size=min_component_size,
    )

    # Convert WikifyResponseElems into dictionaries
    results = [vars(result) for result in results]

    log.debug('🕸️ Generating graph SVG')
    jobs.wikify_graph_svg(results, concept_score_threshold, edge_threshold, min_component_size, request_id=_request_id(request))
    log.info('✅ Graph SVG generated', duration_ms=_duration_ms(start))

    # Return svg file
    return FileResponse('/tmp/file.svg')


@router.post('/generate_exercise')
async def generate_exercise(request: Request, data: schemas.GenerateExerciseRequest):
    """
    Makes a request to the Chatbot API to generate an exercise.
    """

    start = time.perf_counter()
    log = logger.bind(
        endpoint='/text/generate_exercise',
        method=request.method,
        request_id=_request_id(request),
    )
    log.debug('🎓 Generating exercise')
    result = jobs.generate_exercise(data, request_id=_request_id(request))
    log.info('✅ Exercise generated', duration_ms=_duration_ms(start))
    return result
