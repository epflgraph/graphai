from typing import Optional

import pandas as pd

from fastapi import APIRouter

from celery import chain, group

from graphai.api.common.celery_tools import get_n_celery_workers

from graphai.api.schemas.text import *

from graphai.api.celery_tasks.text import extract_keywords_task, wikisearch_task, wikisearch_callback_task, compute_scores_task, aggregate_and_filter_task


pd.set_option('display.max_rows', 400)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Initialise text router
router = APIRouter(
    prefix='/text',
    tags=['text'],
    responses={404: {'description': 'Not found'}}
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
async def wikify(data: WikifyRequest, method: Optional[str] = None):
    """
    Processes raw text (e.g. from an abstract of a publication, a course description or a lecture slide) and returns a
    list of concepts (Wikipedia pages) that are relevant to the text, each with a set of scores in [0, 1]
    quantifying their relevance. This is done as follows:

    1. Keyword extraction: Automatic extraction of keywords from the text. Omitted if keyword_list is provided as
        input instead of raw_text.
    2. Wikisearch: For each set of keywords, a set of 10 concepts (Wikipedia pages) is retrieved. This can be done
        through requests to the Wikipedia API or through elasticsearch requests.
    3. Scores: For each pair of keywords and concept, several scores are derived, taking into account the ontology
        of concepts and the concepts graph, among others.
    4. Aggregation and filter: For each concept, their scores are aggregated and filtered according to some rules,
        to keep only the most relevant results.
    """

    # Get input parameters
    raw_text = data.raw_text

    # Return if no input
    if not raw_text:
        return []

    # Set up composite job
    try:
        n = get_n_celery_workers()
    except Exception as e:
        n = 10

    job = chain(
        extract_keywords_task.s(raw_text),
        group(wikisearch_task.s(fraction=(i / n, (i + 1) / n), method=method) for i in range(n)),
        wikisearch_callback_task.s(),
        compute_scores_task.s(),
        aggregate_and_filter_task.s()
    )

    # Schedule job
    results = job.apply_async(priority=10)

    # Wait for results
    results = results.get(timeout=10)

    return results.to_dict(orient='records')
