import logging

from fastapi import FastAPI

import pandas as pd
import numpy as np

from typing import Optional

from api.schemas.wikify import *
from api.schemas.strip import *

from concept_detection.keyword_extraction import get_keywords
import concept_detection.wikisearch as ws
from graph.scores import ConceptsGraph, Ontology
from concept_detection.scores import compute_scores

from utils.text.markdown import strip

from utils.time.stopwatch import Stopwatch
from utils.time.date import now

import Levenshtein

# Initialise FastAPI
app = FastAPI(
    title="EPFL Graph AI API",
    description="This API offers several tools related with AI in the context of the EPFL Graph project, "
                "such as automatized concept detection from a given text.",
    version="0.1.0"
)

# Get uvicorn logger so we can write on it
logger = logging.getLogger('uvicorn.error')

# Create a ConceptsGraph instance to hold concepts graph in memory
logger.info(f'Fetching concepts graph from database...')
graph = ConceptsGraph()

# Create an Ontology instance to hold clusters graph in memory
logger.info(f'Fetching ontology from database...')
ontology = Ontology()


def build_log_msg(msg, seconds, total=False, length=64):
    padding_length = length - len(msg)
    if padding_length > 0:
        padding = '.' * padding_length
    else:
        padding = ''

    if total:
        time_msg = f'Elapsed total time: {seconds}s.'
    else:
        time_msg = f'Elapsed time: {seconds}s.'

    return f'[{now()}] {msg}{padding} {time_msg}'


@app.post('/keywords')
async def keywords(data: KeywordsRequest, use_nltk: Optional[bool] = False):
    return get_keyword_list(data.raw_text, use_nltk)


@app.post('/wikify', response_model=WikifyResponse)
async def wikify(data: WikifyRequest, method: Optional[str] = None, version: Optional[str] = None):
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

    if version == 'new':
        return new_wikify(data, method)

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
        logger.info(build_log_msg(f'Received raw text "{raw_text[:32]}..."', sw.delta()))
        keyword_list = get_keyword_list(raw_text)
        logger.info(build_log_msg(f'Extracted list of {len(keyword_list)} keywords', sw.delta()))
    else:
        logger.info(build_log_msg(f'Received list of {len(keyword_list)} keywords: [{keyword_list[0]}, ...]', sw.delta()))

    # Perform wikisearch and extract source_page_ids
    wikisearch_results = ws.wikisearch(keyword_list, method)
    logger.info(build_log_msg(f'Finished Wikipedia API wikisearch with {len(wikisearch_results)} source pages', sw.delta()))

    wikisearch_results_2 = ws.wikisearch(keyword_list, 'es-base')
    logger.info(build_log_msg(f'Finished elasticsearch wikisearch with {len(wikisearch_results_2)} source pages', sw.delta()))

    if version == 'new':
        return new_wikify(wikisearch_results) + new_wikify(wikisearch_results_2)
        # return new_wikify(wikisearch_results_2)

    # Extract source_page_ids and anchor_page_ids if needed
    source_page_ids = ws.extract_page_ids(wikisearch_results)
    if not anchor_page_ids:
        anchor_page_ids = ws.extract_anchor_page_ids(wikisearch_results)

    # Filter None values from source and anchor page ids, due to pages not found in the page title ids mapping
    source_page_ids = list(filter(None, source_page_ids))
    anchor_page_ids = list(filter(None, anchor_page_ids))
    n_source_page_ids = len(source_page_ids)
    n_anchor_page_ids = len(anchor_page_ids)
    logger.info(build_log_msg(f'Finished {f"{method} " if method else ""}wikisearch with {n_source_page_ids} source pages', sw.delta()))

    # Compute graph scores
    graph_results = graph.compute_scores(source_page_ids, anchor_page_ids)
    logger.info(build_log_msg(f'Computed graph scores for {n_source_page_ids * n_anchor_page_ids} pairs', sw.delta()))

    # Post-process results and derive the different scores
    results = compute_scores(wikisearch_results, graph_results, logger)
    n_results = len(results)
    logger.info(build_log_msg(f'Post-processed results, got {n_results}', sw.delta()))

    # Display total elapsed time
    logger.info(build_log_msg(f'Finished all tasks', sw.total(), total=True))

    return results


# Function f: [1, N] -> [0, 1] satisfying the following:
#   f(1) = 1
#   f(N) = 0
#   f decreasing
#   f concave
def f(x, N):
    return np.cos((np.pi / 2) * (x - 1) / (N - 1))


# S-shaped function on [0, 1] that pulls values away from 1/2, exaggerating differences
def h(x):
    return 1 / (1 + ((1 - x) / x)**2)


def new_wikify(data, method):
    # Initialize stopwatch to track time
    sw = Stopwatch()

    # Get input parameters
    raw_text = data.raw_text

    # Return if no input
    if not raw_text:
        return []

    logger.info(build_log_msg(f'Received raw text "{raw_text[:32]}..."', sw.delta()))

    # Extract keywords from text
    keywords = get_keywords(raw_text)
    logger.info(build_log_msg(f'Extracted list of {len(keywords)} keywords', sw.delta()))

    # Perform wikisearch to get concepts for all sets of keywords
    results = ws.wikisearch(keywords, 'es-base')
    logger.info(build_log_msg(f'Finished elasticsearch wikisearch with {len(results)} source pages', sw.delta()))

    results_2 = ws.wikisearch(keywords, 'wikipedia-api')
    logger.info(build_log_msg(f'Finished Wikipedia API wikisearch with {len(results_2)} source pages', sw.delta()))

    ############################################################

    # Compute levenshtein score
    results['LevenshteinScore'] = results.apply(lambda row: Levenshtein.ratio(row['Keywords'], row['PageTitle'].replace('_', ' ').lower()), axis=1)
    results['LevenshteinScore'] = h(results['LevenshteinScore'])

    ############################################################

    # Compute ontology score
    results = ontology.add_ontology_score(results)

    # Compute graph score
    results = graph.add_graph_score(results)

    # Compute keywords score aggregating ontology score to detect low-quality keywords
    results = pd.merge(
        results,
        results.groupby(by=['Keywords']).aggregate(KeywordsScore=('OntologyScore', 'sum')).reset_index(),
        how='left',
        on=['Keywords']
    )
    results['KeywordsScore'] = results['KeywordsScore'] / results['KeywordsScore'].max()

    # Aggregate over pages
    results = results.groupby(by=['PageID', 'PageTitle']).aggregate(
        SearchScore=('SearchScore', 'sum'),
        LevenshteinScore=('LevenshteinScore', 'sum'),
        OntologyLocalScore=('OntologyLocalScore', 'sum'),
        OntologyGlobalScore=('OntologyGlobalScore', 'sum'),
        Ontology2LocalScore=('Ontology2LocalScore', 'sum'),
        Ontology2GlobalScore=('Ontology2GlobalScore', 'sum'),
        OntologyScore=('OntologyScore', 'sum'),
        GraphScore=('GraphScore', 'sum'),
        KeywordsScore=('KeywordsScore', 'sum'),
    ).reset_index()

    results['SearchScore'] = results['SearchScore'] / results['SearchScore'].max()
    results['LevenshteinScore'] = results['LevenshteinScore'] / results['LevenshteinScore'].max()
    results['OntologyLocalScore'] = results['OntologyLocalScore'] / results['OntologyLocalScore'].max()
    results['OntologyGlobalScore'] = results['OntologyGlobalScore'] / results['OntologyGlobalScore'].max()
    results['Ontology2LocalScore'] = results['Ontology2LocalScore'] / results['Ontology2LocalScore'].max()
    results['Ontology2GlobalScore'] = results['Ontology2GlobalScore'] / results['Ontology2GlobalScore'].max()
    results['OntologyScore'] = results['OntologyScore'] / results['OntologyScore'].max()
    results['GraphScore'] = results['GraphScore'] / results['GraphScore'].max()
    results['KeywordsScore'] = results['KeywordsScore'] / results['KeywordsScore'].max()

    # results = results[(results['OntologyScore'] >= 0.1) | (results['GraphScore'] >= 0.1) | (results['KeywordsScore'] >= 0.1)]

    print(results)

    # aaa = results['OntologyScore'] < 0.1
    # bbb = results['GraphScore'] < 0.1
    # ccc = results['KeywordsScore'] < 0.1

    # print(results[aaa & bbb & ccc])
    # print(results[aaa & bbb])
    # print(results[aaa & ccc])
    # print(results[bbb & ccc])

    # return []
    #
    # stable = False
    # while not stable:
    #     print(results[results['OntologyScore'] < 0.05])
    #     print(results[results['GraphScore'] < 0.05])
    #     print(results[results['KeywordsScore'] < 0.05])
    #     # print(results[(results['OntologyScore'] < 0.05) & (results['GraphScore'] < 0.05) & (results['KeywordsScore'] < 0.05)])
    #
    #     n1 = len(results)
    #     results = results[(results['OntologyScore'] >= 0.05) | (results['GraphScore'] >= 0.05) | (results['KeywordsScore'] < 0.05)]
    #     n2 = len(results)
    #     stable = (n1 == n2)
    #
    #     results = results[['Keywords', 'PageID', 'PageTitle', 'Searchrank', 'SearchScore']]
    #
    #     # Compute ontology score
    #     results = ontology.add_ontology_score(results)
    #
    #     # Compute graph score
    #     results = graph.add_graph_score(results)
    #
    #     # Compute keywords score aggregating ontology score to detect low-quality keywords
    #     results = pd.merge(
    #         results,
    #         results.groupby(by=['Keywords']).aggregate(KeywordsScore=('OntologyGlobalScore', 'sum')).reset_index(),
    #         how='left',
    #         on=['Keywords']
    #     )
    #     results['KeywordsScore'] = results['KeywordsScore'] / results['KeywordsScore'].max()
    #
    # return []

    results = results.sort_values(by='PageTitle')

    # # Compute mixed score
    # results = results.fillna(0)
    # # results['MixedScore'] = results[['SearchScore', 'OntologyScore', 'GraphScore', 'LevenshteinScore']].mean(axis=1)
    # results['MixedScore'] = 0.3 * results['SearchScore'] + 0.5 * results['OntologyScore'] + 0.1 * results['GraphScore'] + 0.1 * results['LevenshteinScore']
    # results = results.sort_values(by='MixedScore', ascending=False)

    # Return results
    # output_columns = ['Keywords', 'PageID', 'PageTitle', 'Searchrank', 'SearchScore', 'OntologyScore', 'GraphScore', 'LevenshteinScore', 'MixedScore']
    # results = results[output_columns]
    return results.to_dict(orient='records')


def scores(results):
    # Add search count column
    results = pd.merge(
        results,
        results.groupby(by='Keywords').aggregate(SearchCount=('PageID', 'count')).reset_index(),
        how='left',
        on='Keywords'
    )

    # Compute search score (+ fix rounding error)
    results['SearchScore'] = f(results['Searchrank'], results['SearchCount'])
    results.loc[results['SearchScore'] < 0.001, 'SearchScore'] = 0
    results = results.drop(columns=['SearchCount'])

    # Compute ontology score
    results = ontology.add_ontology_score(results)

    # Compute graph score
    results = graph.add_graph_score(results)

    # Compute levenshtein score
    results['LevenshteinScore'] = results.apply(
        lambda row: Levenshtein.ratio(row['Keywords'], row['PageTitle'].lower()), axis=1)
    results['LevenshteinScore'] = h(results['LevenshteinScore'])

    # Compute mixed score
    results = results.fillna(0)
    # results['MixedScore'] = results[['SearchScore', 'OntologyScore', 'GraphScore', 'LevenshteinScore']].mean(axis=1)
    results['MixedScore'] = 0.3 * results['SearchScore'] + 0.5 * results['OntologyScore'] + 0.1 * results[
        'GraphScore'] + 0.1 * results['LevenshteinScore']
    results = results.sort_values(by='MixedScore', ascending=False)

    # Filter low scores
    # results = results[results['MixedScore'] >= 0.2]

    # print(results.head(20))
    # print(results.info())
    # print(results.describe())

    # Return results
    output_columns = ['Keywords', 'PageID', 'PageTitle', 'Searchrank', 'SearchScore', 'OntologyScore', 'GraphScore',
                      'LevenshteinScore', 'MixedScore']
    results = results[output_columns]
    return results.to_dict(orient='records')


@app.post('/markdown_strip')
async def markdown_strip(data: StripRequest):
    return {'stripped_code': strip(data.markdown_code)['text']}
