import numpy as np
import json
import logging

from fastapi import FastAPI

from typing import List

from definitions import DATA_DIR
from concept_detection.graph.schemas import ScoresData, ScoresResult

# Initialise FastAPI
app = FastAPI()

# Get uvicorn logger so we can write on it
logger = logging.getLogger("uvicorn.error")

# Load successors adjacency list
logger.info('Loading successors adjacency list...')
with open(f'{DATA_DIR}/successors.json') as f:
    successors = json.load(f)
successors = {int(k): v for k, v in successors.items()}
logger.info('Loaded')

# Load predecessors adjacency list
logger.info('Loading predecessors adjacency list...')
with open(f'{DATA_DIR}/predecessors.json') as f:
    predecessors = json.load(f)
predecessors = {int(k): v for k, v in predecessors.items()}
logger.info('Loaded')


def graph_scores(source_page_ids, target_page_ids):
    pairs = [(s, t) for s in source_page_ids for t in target_page_ids]

    results = []
    for s, t in pairs:

        # If both pages are the same, the score is maximum
        if s == t:
            results.append({
                'source_page_id': s,
                'target_page_id': t,
                'score': 1
            })
            continue

        # If the graph does not have both nodes, the score is zero
        if (s not in successors.keys()) or (t not in successors.keys()):
            results.append({
                'source_page_id': s,
                'target_page_id': t,
                'score': 0
            })
            continue

        # Compute the number of outgoing paths of length <= 2
        s_out = set(successors[s]) - {s}
        t_in = set(predecessors[t]) - {t}
        n_out_paths = len(s_out & t_in) + (1 if t in s_out else 0)

        # If there are no outgoing paths, the score is zero
        if n_out_paths == 0:
            results.append({
                'source_page_id': s,
                'target_page_id': t,
                'score': 0
            })
            continue

        # Compute the number of ingoing paths of length <= 2
        s_in = set(predecessors[s]) - {s}
        t_out = set(successors[t]) - {t}
        n_in_paths = len(s_in & t_out) + (1 if s in t_out else 0)

        # Compute score
        rebound = (1 + min(n_out_paths, n_in_paths)) / (1 + max(n_out_paths, n_in_paths))
        score = 1 - 1 / (1 + np.log(1 + rebound * np.log(n_out_paths)))

        # Append result
        results.append({
            'source_page_id': s,
            'target_page_id': t,
            'score': score
        })

    return results


@app.post('/scores', response_model=List[ScoresResult])
async def scores(scores_data: ScoresData):
    """
    Computes the graph scores for all pairs of source and target Wikipedia pages.
    """

    # Get input parameters
    source_page_ids = scores_data.source_page_ids
    target_page_ids = scores_data.target_page_ids

    return graph_scores(source_page_ids, target_page_ids)
