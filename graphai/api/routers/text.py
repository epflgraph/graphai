import pandas as pd

import Levenshtein

from fastapi import APIRouter

from typing import Optional

from graphai.api.schemas.text import *

from graphai.api.common.log import log
from graphai.api.common.graph import graph
from graphai.api.common.ontology import ontology

from graphai.core.text.keywords import get_keywords
from graphai.core.text.wikisearch import wikisearch

from graphai.core.utils.text.markdown import strip
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
    results = wikisearch(keywords, method)
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


# @router.post('/legacy_wikify', response_model=WikifyResponse)
async def legacy_wikify(data: WikifyRequest, method: Optional[str] = None):
    """
    Wikifies some text.

    Wikifying a text is the composition of the following steps:

    * Keyword extraction: Automatic extraction of keywords from the text. Omitted if keyword_list is provided as input instead of raw_text.
    * Wikisearch: For each set of keywords, we call the Wikipedia API to search the 10 most related Wikipedia pages.
    * Graph scores: For each such page and each anchor page specified in the parameters, we search the graph and
        compute a score depending on how well-connected both pages are.
    * Postprocessing: For each set of keywords and Wikipedia page, the graph scores are aggregated over all anchor pages
        and several other scores are computed.
    """

    # Get input parameters
    raw_text = data.raw_text
    keyword_list = data.keyword_list
    anchor_page_ids = data.anchor_page_ids

    # Return if no input
    if not raw_text and not keyword_list:
        return []

    # Initialize stopwatch to track time
    sw = Stopwatch()

    # Extract keywords from text
    if raw_text:
        log(f'Received raw text "{raw_text[:32]}..."', sw.delta())
        keyword_list = get_keywords(raw_text)
        log(f'Extracted list of {len(keyword_list)} keywords', sw.delta())
    else:
        log(f'Received list of {len(keyword_list)} keywords: [{keyword_list[0]}, ...]', sw.delta())

    # Perform wikisearch and extract source_page_ids
    wikisearch_results = wikisearch(keyword_list, method)
    log(f"""Finished {method if method else 'es-base'} wikisearch with {len(wikisearch_results)} results""", sw.delta())

    # Extract source_page_ids and anchor_page_ids if needed
    source_page_ids = extract_page_ids(wikisearch_results)
    if not anchor_page_ids:
        anchor_page_ids = extract_anchor_page_ids(wikisearch_results)

    # Filter None values from source and anchor page ids, due to pages not found in the page title ids mapping
    source_page_ids = list(filter(None, source_page_ids))
    anchor_page_ids = list(filter(None, anchor_page_ids))
    n_source_page_ids = len(source_page_ids)
    n_anchor_page_ids = len(anchor_page_ids)
    log(f'Finished {f"{method} " if method else ""}wikisearch with {n_source_page_ids} source pages', sw.delta())

    # Compute graph scores
    graph_results = graph.compute_scores(source_page_ids, anchor_page_ids)
    log(f'Computed graph scores for {n_source_page_ids * n_anchor_page_ids} pairs', sw.delta())

    # Post-process results and derive the different scores
    results = compute_scores(wikisearch_results, graph_results)
    n_results = len(results)
    log(f'Post-processed results, got {n_results}', sw.delta())

    # Display total elapsed time
    log(f'Finished all tasks', sw.total(), total=True)

    return results


@router.post('/markdown_strip', response_model=StripResponse)
async def markdown_strip(data: StripRequest):
    return {'stripped_code': strip(data.markdown_code)['text']}
