from fastapi import APIRouter, Security
from fastapi.responses import FileResponse

from celery import chain, group
from typing import Optional, Union

import pandas as pd

from graphai.api.schemas.text import (
    KeywordsRequest,
    KeywordsResponse,
    WikifyFromRawTextRequest,
    WikifyFromKeywordsRequest,
    WikifyResponse,
)
from graphai.api.celery_tasks.text import (
    extract_keywords_task,
    wikisearch_task,
    compute_scores_task,
    draw_ontology_task,
    draw_graph_task,
    text_test_task,
)
from graphai.api.routers.auth import get_current_active_user

pd.set_option('display.max_rows', 400)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Initialise text router
router = APIRouter(
    prefix='/text',
    tags=['text'],
    responses={404: {'description': 'Not found'}},
    dependencies=[Security(get_current_active_user, scopes=['text'])]
)


@router.post('/keywords', response_model=KeywordsResponse)
async def keywords(data: KeywordsRequest, use_nltk: Optional[bool] = False):
    """
    Processes raw text (e.g. from an abstract of a publication, a course description or a lecture slide) and returns a
    list of keywords from the text.
    """

    # Get input parameters
    raw_text = data.raw_text

    # Return if no input
    if not raw_text:
        return []

    # Set up job
    job = extract_keywords_task.s(raw_text, use_nltk=use_nltk)

    # Schedule job
    results = job.apply_async(priority=10)

    # Wait for results
    results = results.get(timeout=10)

    return results


@router.post('/wikify', response_model=WikifyResponse)
async def wikify(
    data: Union[WikifyFromRawTextRequest, WikifyFromKeywordsRequest],
    method: Optional[str] = 'es-base',
    restrict_to_ontology: Optional[bool] = False,
    graph_score_smoothing: Optional[bool] = True,
    ontology_score_smoothing: Optional[bool] = True,
    keywords_score_smoothing: Optional[bool] = True,
    aggregation_coef: Optional[float] = 0.5,
    filtering_threshold: Optional[float] = 0.1,
    filtering_min_votes: Optional[int] = 5,
    refresh_scores: Optional[bool] = True,
):
    """
    Processes raw text (e.g. from an abstract of a publication, a course description or a lecture slide) and returns a
    list of concepts (Wikipedia pages) that are relevant to the text, each with a set of scores in [0, 1]
    quantifying their relevance. This is done as follows:

    1. Keyword extraction: Automatic extraction of keywords from the text. Omitted if keyword_list is provided as
        input instead of raw_text.
    2. Wikisearch: For each set of keywords, a set of at most 10 concepts (Wikipedia pages) is retrieved. This can be
        done through requests to the Wikipedia API or through elasticsearch requests.
    3. Scores: For each pair of keywords and concept, several scores are derived, taking into account the concepts graph and
        the ontology, among others.
    4. Aggregation and filter: For each concept, their scores are aggregated and filtered according to some rules,
        to keep only the most relevant results.
    """

    # Get input parameters
    if isinstance(data, WikifyFromRawTextRequest):
        raw_text = data.raw_text
    elif isinstance(data, WikifyFromKeywordsRequest):
        raw_text = data.keywords
    else:
        return []

    # Return if no input
    if not raw_text:
        return []

    n = 16

    job = chain(
        extract_keywords_task.s(raw_text),
        group(wikisearch_task.s(fraction=(i / n, (i + 1) / n), method=method) for i in range(n)),
        compute_scores_task.s(
            restrict_to_ontology=restrict_to_ontology,
            graph_score_smoothing=graph_score_smoothing,
            ontology_score_smoothing=ontology_score_smoothing,
            keywords_score_smoothing=keywords_score_smoothing,
            aggregation_coef=aggregation_coef,
            filtering_threshold=filtering_threshold,
            filtering_min_votes=filtering_min_votes,
            refresh_scores=refresh_scores,
        )
    )

    # Schedule job
    results = job.apply_async(priority=10)

    # Wait for results
    results = results.get(timeout=300)

    # FIXME delete this and go ahead with the new lowercased keys
    # Replace column names not to break compatibility
    columns_map = {
        'keywords': 'Keywords',
        'concept_id': 'PageID',
        'concept_name': 'PageTitle',
        'searchrank': 'Searchrank',
        'search_score': 'SearchScore',
        'levenshtein_score': 'LevenshteinScore',
        'graph_score': 'GraphScore',
        'ontology_local_score': 'OntologyLocalScore',
        'ontology_global_score': 'OntologyGlobalScore',
        'keywords_score': 'KeywordsScore',
        'mixed_score': 'MixedScore',
    }
    results = results.rename(columns=columns_map)

    return results.to_dict(orient='records')


@router.post('/wikify_ontology_svg')
async def wikify_ontology_svg(
    results: WikifyResponse,
    level: Optional[int] = 2
):
    """
    Returns a svg file representing the ontology subgraph induced by the provided set of results.
    """

    # Convert WikifyResponseElems into dictionaries
    results = [vars(result) for result in results]

    # Switch to default level if not properly defined
    if level not in [1, 2, 3, 4, 5]:
        level = 2

    # Set up job
    job = draw_ontology_task.s(results, level)

    # Schedule job and block
    job.apply_async(priority=10).get(timeout=10)

    # Return file
    return FileResponse('/tmp/file.svg')


@router.post('/wikify_graph_svg')
async def wikify_graph_svg(
    results: WikifyResponse,
    concept_score_threshold: Optional[float] = 0.3,
    edge_threshold: Optional[float] = 0.3,
    min_component_size: Optional[int] = 3
):
    """
    Returns a svg file representing the graph subgraph induced by the provided set of results.
    """

    # Convert WikifyResponseElems into dictionaries
    results = [vars(result) for result in results]

    # Set up job
    job = draw_graph_task.s(results, concept_score_threshold, edge_threshold, min_component_size)

    # Schedule job and block
    job.apply_async(priority=10).get(timeout=10)

    # Return file
    return FileResponse('/tmp/file.svg')


@router.post('/priority_test')
async def priority_test_text():
    print('the dummy that matters is being launched')
    task = group(text_test_task.s() for _ in range(8)).apply_async(priority=10)
    return {'id': task.id}
