import pandas as pd

import Levenshtein

from fastapi import APIRouter

from typing import Optional

from graphai.api.schemas.text import *

from graphai.api.common.log import log
from graphai.api.common.graph import graph
from graphai.api.common.ontology import ontology

from graphai.core.text.keywords import get_keywords

from graphai.api.celery_tasks.text import wikisearch_master

from graphai.core.utils.time.stopwatch import Stopwatch

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
    return get_keywords(data.raw_text, use_nltk)


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

    # Initialize stopwatch to track time
    sw = Stopwatch()

    # Get input parameters
    raw_text = data.raw_text

    # Return if no input
    if not raw_text:
        return []

    log(f'Received raw text "{raw_text[:32]}..."', sw.delta())

    # Extract keywords from text
    keywords = get_keywords(raw_text)
    log(f'Extracted list of {len(keywords)} keywords', sw.delta())

    # Perform wikisearch to get concepts for all sets of keywords
    results = wikisearch_master(keywords, method)
    log(f"""Finished {method if method else 'es-base'} wikisearch with {len(results)} results""", sw.delta())

    ############################################################

    # Compute levenshtein score
    # S-shaped function on [0, 1] that pulls values away from 1/2, exaggerating differences
    def f(x):
        return 1 / (1 + ((1 - x) / x) ** 2)

    results['LevenshteinScore'] = results.apply(lambda row: Levenshtein.ratio(row['Keywords'], row['PageTitle'].replace('_', ' ').lower()), axis=1)
    results['LevenshteinScore'] = f(results['LevenshteinScore'])

    ############################################################

    # Compute ontology score
    results = ontology.add_ontology_scores(results)

    # Compute graph score
    results = graph.add_graph_score(results)

    ############################################################

    # Compute keywords score aggregating ontology global score over Keywords as an indicator for low-quality keywords
    results = pd.merge(
        results,
        results.groupby(by=['Keywords']).aggregate(KeywordsScore=('OntologyGlobalScore', 'sum')).reset_index(),
        how='left',
        on=['Keywords']
    )
    results['KeywordsScore'] = results['KeywordsScore'] / results['KeywordsScore'].max()

    log(f"""Computed scores for {len(results)} results""", sw.delta())

    ############################################################

    # Aggregate over pages
    results = results.groupby(by=['PageID', 'PageTitle']).aggregate(
        SearchScore=('SearchScore', 'sum'),
        LevenshteinScore=('LevenshteinScore', 'sum'),
        OntologyLocalScore=('OntologyLocalScore', 'sum'),
        OntologyGlobalScore=('OntologyGlobalScore', 'sum'),
        GraphScore=('GraphScore', 'sum'),
        KeywordsScore=('KeywordsScore', 'sum')
    ).reset_index()
    score_columns = ['SearchScore', 'LevenshteinScore', 'OntologyLocalScore', 'OntologyGlobalScore', 'GraphScore', 'KeywordsScore']

    # Normalise scores to [0, 1]
    for column in score_columns:
        results[column] = results[column] / results[column].max()

    log(f"""Aggregated results, got {len(results)} concepts""", sw.delta())

    ############################################################

    # Filter results with low scores through a majority vote among all scores
    # To be kept, we require a concept to have at least 5 out of 6 scores to be significant (>= epsilon)
    epsilon = 0.1
    votes = pd.DataFrame()
    for column in score_columns:
        votes[column] = (results[column] >= epsilon).astype(int)
    votes = votes.sum(axis=1)
    results = results[votes >= 5]
    results = results.sort_values(by='PageTitle')

    log(f"""Filtered concepts with low scores, kept {len(results)} concepts""", sw.delta())

    ############################################################

    # Compute mixed score as a convex combination of the different scores,
    # with prescribed coefficients found after running some analyses on manually tagged data.
    coefficients = pd.DataFrame({
        'SearchScore': [0.2],
        'LevenshteinScore': [0.15],
        'OntologyLocalScore': [0.15],
        'OntologyGlobalScore': [0.1],
        'GraphScore': [0.1],
        'KeywordsScore': [0.3]
    })
    results['MixedScore'] = results[score_columns] @ coefficients.transpose()

    log(f'Finished all tasks', sw.total(), total=True)

    ############################################################

    return results.to_dict(orient='records')
