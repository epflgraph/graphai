import time
import logging
import json

from fastapi import FastAPI

from typing import List, Optional

from definitions import DATA_DIR
from api.schemas.wikify import *

from concept_detection.keywords.extraction import get_keyword_list
import concept_detection.search.wikisearch as ws
import concept_detection.search.elasticwikisearch as ews
from graph.scores import compute_graph_scores
from concept_detection.scores.postprocessing import compute_scores

from api.schemas.strip import *
from concept_detection.text.stripper import strip

# Initialise FastAPI
app = FastAPI(
    title="EPFL Graph AI API",
    description="This API offers several tools related with AI in the context of the EPFL Graph project, "
                "such as automatized concept detection from a given text.",
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

    return f'{msg}{padding} {time_msg}'


@app.post('/keywords')
async def keywords(data: KeywordsRequest, use_nltk: Optional[bool] = False):
    return get_keyword_list(data.raw_text, use_nltk)


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
        method = 'wikipedia-api'

    # Return if no input
    if not raw_text and not keyword_list:
        return []

    # Extract keywords from text
    if raw_text:
        start_time = time.time()
        keyword_list = get_keyword_list(raw_text)
        logger.info(build_log_msg(f'Extracted list of {len(keyword_list)} keywords', time.time() - start_time))

    # Perform wikisearch and extract source_page_ids
    start_time = time.time()
    if method == 'es-base':
        wikisearch_results = ews.wikisearch(keyword_list, es_scores=False)
    elif method == 'es-score':
        wikisearch_results = ews.wikisearch(keyword_list, es_scores=True)
    else:
        wikisearch_results = ws.wikisearch(keyword_list)

    # Extract source_page_ids and anchor_page_ids if needed
    source_page_ids = ws.extract_page_ids(wikisearch_results)
    if not anchor_page_ids:
        anchor_page_ids = ws.extract_anchor_page_ids(wikisearch_results)

    # Filter None values from source and anchor page ids, due to pages not found in the page title ids mapping
    source_page_ids = list(filter(None, source_page_ids))
    anchor_page_ids = list(filter(None, anchor_page_ids))
    n_source_page_ids = len(source_page_ids)
    n_anchor_page_ids = len(anchor_page_ids)
    logger.info(build_log_msg(f'Finished {method} wikisearch with {n_source_page_ids} source pages', time.time() - start_time))

    # Compute graph scores
    start_time = time.time()
    graph_results = compute_graph_scores(source_page_ids, anchor_page_ids)
    logger.info(build_log_msg(f'Computed graph scores for {n_source_page_ids * n_anchor_page_ids} pairs', time.time() - start_time))

    # Post-process results and derive the different scores
    start_time = time.time()
    results = compute_scores(wikisearch_results, graph_results, page_id_titles, logger)
    n_results = len(results)
    logger.info(build_log_msg(f'Post-processed results, got {n_results}', time.time() - start_time))

    # Display total elapsed time
    logger.info(build_log_msg(f'Finished all tasks', time.time() - global_start_time, total=True))

    return results


@app.post('/markup_strip')
async def markup_strip(data: StripData):
    return {'stripped_code': strip(data.markup_code)['text']}
