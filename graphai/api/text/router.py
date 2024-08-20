from fastapi import APIRouter, Security
from fastapi.responses import FileResponse

from typing import Optional, Union

import pandas as pd

from graphai.api.routers.auth import get_current_active_user
from graphai.api.text.schemas import (
    KeywordsRequest,
    KeywordsResponse,
    WikifyFromRawTextRequest,
    WikifyFromKeywordsRequest,
    WikifyResponse,
)
import graphai.celery.text.jobs as jobs


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

    # Return if no input
    if not data.raw_text:
        return []

    return jobs.keywords(data.raw_text, use_nltk)


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
    filtering_min_agreement_fraction: Optional[float] = 0.8,
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
    * graph_score_smoothing (bool): Whether to apply a transformation to the graph scores that bumps scores to avoid
    a pronounced negative exponential shape. Default: True.
    * ontology_score_smoothing (bool): Whether to apply a transformation to the ontology scores that pushes scores away from 0.5. Default: True.
    * keywords_score_smoothing (bool): Whether to apply a transformation to the keywords scores that bumps scores to avoid
    a pronounced negative exponential shape. Default: True.
    * aggregation_coef (float): A number in [0, 1] that controls how the scores of the aggregated pages are computed.
    A value of 0 takes the sum of scores over Keywords, then normalises in [0, 1]. A value of 1 takes the max of scores over Keywords.
    Any value in between linearly interpolates those two approaches. Default: 0.5.
    * filtering_threshold (float): A number in [0, 1] that is used as a threshold for all the scores to decide whether the page is good enough
    from that score's perspective. Default: 0.1.
    * filtering_min_agreement_fraction (float): A number between 0 and 1. A page will be kept iff it is good enough for at least this fraction of scores. Default: 0.8.
    * refresh_scores (bool): Whether to recompute scores after filtering. Default: True.
    """

    if isinstance(data, WikifyFromRawTextRequest):
        # Return if no input
        if not data.raw_text:
            return []

        return jobs.wikify_text(
            data.raw_text,
            method,
            restrict_to_ontology,
            graph_score_smoothing,
            ontology_score_smoothing,
            keywords_score_smoothing,
            aggregation_coef,
            filtering_threshold,
            filtering_min_agreement_fraction,
            refresh_scores,
        )

    if isinstance(data, WikifyFromKeywordsRequest):
        # Return if no input
        if not data.keywords:
            return []

        return jobs.wikify_keywords(
            data.keywords,
            method,
            restrict_to_ontology,
            graph_score_smoothing,
            ontology_score_smoothing,
            keywords_score_smoothing,
            aggregation_coef,
            filtering_threshold,
            filtering_min_agreement_fraction,
            refresh_scores,
        )

    return []


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

    # Run job that will create svg file in tmp location
    jobs.wikify_ontology_svg(results, level)

    # Return svg file
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

    # Run job that will create svg file in tmp location
    jobs.wikify_graph_svg(results, concept_score_threshold, edge_threshold, min_component_size)

    # Return svg file
    return FileResponse('/tmp/file.svg')
