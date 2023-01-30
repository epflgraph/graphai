import logging

from fastapi import FastAPI

from typing import Optional

from api.schemas.wikify import *
from api.schemas.strip import *

from concept_detection.keyword_extraction import get_keyword_list
import concept_detection.wikisearch as ws
from graph.scores import ConceptsGraph, Ontology
from concept_detection.scores import compute_scores

from utils.text.markdown import strip

from utils.time.stopwatch import Stopwatch
from utils.time.date import now

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
cg = ConceptsGraph()

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

    if version == 'new':
        return new_wikify(wikisearch_results)

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
    graph_results = cg.compute_scores(source_page_ids, anchor_page_ids)
    logger.info(build_log_msg(f'Computed graph scores for {n_source_page_ids * n_anchor_page_ids} pairs', sw.delta()))

    # Post-process results and derive the different scores
    results = compute_scores(wikisearch_results, graph_results, logger)
    n_results = len(results)
    logger.info(build_log_msg(f'Post-processed results, got {n_results}', sw.delta()))

    # Display total elapsed time
    logger.info(build_log_msg(f'Finished all tasks', sw.total(), total=True))

    return results


def new_wikify(results):
    import pandas as pd
    import numpy as np

    # Convert into a pandas DataFrame
    table = []
    for result in results:
        for page in result.pages:
            table.append([result.keywords, page.page_id, page.page_title, page.searchrank, page.score])
    results = pd.DataFrame(table, columns=['Keywords', 'PageID', 'PageTitle', 'Searchrank', 'SearchScore'])

    # Compute search score
    results = pd.merge(
        results,
        results.groupby(by='Keywords').aggregate(Count=('PageID', 'count')).reset_index(),
        how='left',
        on='Keywords'
    )

    # Function f: [1, N] -> [0, 1] satisfying the following:
    #   f(1) = 1
    #   f(N) = 0
    #   f decreasing
    #   f concave
    results['SearchScore'] = np.cos((np.pi / 2) * (results['Searchrank'] - 1) / (results['Count'] - 1))
    results = results.fillna(0)
    results = results[['Keywords', 'PageID', 'PageTitle', 'Searchrank', 'SearchScore']]

    # Add cluster column
    results = ontology.add_cluster(results)

    # Compute ontology score
    results = pd.merge(
        results,
        results.groupby(by=['Keywords', 'ClusterID']).aggregate(Count=('PageID', 'count')).reset_index(),
        how='left',
        on=['Keywords', 'ClusterID']
    )

    # Function f: [1, N] -> [0, 1] satisfying the following:
    #   f(1) = 0
    #   f(N) = 1
    #   f increasing
    #   f concave
    results['OntologyScore'] = np.sin((np.pi / 2) * (results['Count'] - 1) / (results['Count'] - 1))
    results = results.fillna(0)
    results = results[['Keywords', 'PageID', 'PageTitle', 'Searchrank', 'ClusterID', 'SearchScore', 'OntologyScore']]

    # Compute graph score
    print(results)

    return []


@app.post('/markdown_strip')
async def markdown_strip(data: StripRequest):
    return {'stripped_code': strip(data.markdown_code)['text']}
