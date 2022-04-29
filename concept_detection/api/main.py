import time
import logging
import json

from fastapi import FastAPI

from typing import List, Optional

from definitions import DATA_DIR
from concept_detection.api.schemas.wikify import *

from concept_detection.keywords.extraction import get_keyword_list, get_keyword_list_nltk
import concept_detection.search.wikisearch as ws
import concept_detection.search.elasticwikisearch as ews
from concept_detection.graph.scores import graph_scores
from concept_detection.scores.postprocessing import compute_scores

from concept_detection.api.schemas.strip import *
from concept_detection.text.stripper import strip

# Initialise FastAPI
app = FastAPI(
    title="EPFL Graph - Concept detection API",
    description="This API offers several tools related with automatized concept detection, e.g. from a given text.",
    version="0.1.0"
)

# Load page id titles mapping
print('Loading page_id_titles mapping...')
with open(f'{DATA_DIR}/page_id_titles.json') as f:
    page_id_titles = json.load(f)
print('Loaded')

# Load page title ids mapping
print('Loading page_title_ids mapping...')
with open(f'{DATA_DIR}/page_title_ids.json') as f:
    page_title_ids = json.load(f)
print('Loaded')

# Get uvicorn logger so we can write on it
logger = logging.getLogger('uvicorn.error')


@app.post('/keywords')
async def keywords(data: KeywordsRequest, method: Optional[str] = None):
    return get_keyword_list(data.raw_text, method)


@app.post('/wikify', response_model=List[WikifyResult])
async def wikify(data: WikifyRequest, method: Optional[str] = None):
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
    # Measure execution time
    global_start_time = time.time()

    # Get input parameters
    raw_text = data.raw_text
    keyword_list = data.keyword_list
    anchor_page_ids = data.anchor_page_ids
    if not method:
        method = 'base'

    # Extract keywords from text
    if raw_text:
        start_time = time.time()
        keyword_list = get_keyword_list(raw_text)
        logger.info(f'Extracted list of {len(keyword_list)} keywords........................... Elapsed time: {time.time() - start_time}s.')

    # Perform wikisearch and extract source_page_ids
    start_time = time.time()
    if method == 'base':
        wikisearch_results = ws.wikisearch(keyword_list, page_title_ids)
    elif method == 'es-base':
        wikisearch_results = ews.wikisearch(keyword_list, es_scores=False)
    elif method == 'es-score':
        wikisearch_results = ews.wikisearch(keyword_list, es_scores=True)
    else:
        # Should never get here
        wikisearch_results = []

    # Extract source_page_ids and anchor_page_ids if needed
    source_page_ids = ws.extract_page_ids(wikisearch_results)
    if not anchor_page_ids:
        anchor_page_ids = ws.extract_anchor_page_ids(wikisearch_results)

    # Filter None values from source and anchor page ids, due to pages not found in the page title ids mapping
    source_page_ids = list(filter(None, source_page_ids))
    anchor_page_ids = list(filter(None, anchor_page_ids))
    logger.info(f'Finished {method} wikisearch with {len(source_page_ids)} source pages............... Elapsed time: {time.time() - start_time}s.')

    # Compute graph scores
    start_time = time.time()
    graph_results = graph_scores(source_page_ids, anchor_page_ids)
    logger.info(f'Computed graph scores for {len(source_page_ids)*len(anchor_page_ids)} pairs..................... Elapsed time: {time.time() - start_time}s.')

    # Post-process results and derive the different scores
    start_time = time.time()
    scores = compute_scores(wikisearch_results, graph_results, page_id_titles, logger)
    logger.info(f'Post-processed results, got {len(scores)}.......................... Elapsed time: {time.time() - start_time}s.')

    # Display total elapsed time
    logger.info(f'Finished all tasks...................................... Elapsed total time: {time.time() - global_start_time}s.')

    return scores


@app.post('/markup_strip')
async def markup_strip(data: StripData):
    return {'stripped_code': strip(data.markup_code)}
