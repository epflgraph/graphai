import numpy as np

from definitions import DATA_DIR
from utils.text.io import read_json

# Load successors adjacency list
print('Loading successors adjacency list...', end=' ')
successors = read_json(f'{DATA_DIR}/successors.json')
successors = {int(k): v for k, v in successors.items()}
print('Done')

# Load predecessors adjacency list
print('Loading predecessors adjacency list...', end=' ')
predecessors = read_json(f'{DATA_DIR}/predecessors.json')
predecessors = {int(k): v for k, v in predecessors.items()}
print('Done')


def compute_score(n_out_paths, n_in_paths):
    """
    Function to compute the graph score of an pair based on the number of in and outward paths.
    """
    rebound = (1 + min(n_out_paths, n_in_paths)) / (1 + max(n_out_paths, n_in_paths))
    return 1 - 1 / (1 + np.log(1 + rebound * np.log(n_out_paths)))


def compute_graph_scores(source_page_ids, target_page_ids):
    """
    Computes the graph scores for all possible source-target pairs from two lists of page ids.
    The graph score of a pair (s, t) is computed as:

    :math:`\\displaystyle score(s, t) = 1 - \\frac{1}{1 + \\ln(1 + rebound(s, t) * \\ln(out(s, t)))}`, with

    :math:`\\displaystyle rebound(s, t) = \\frac{1 + \\min\\{in(s, t), out(s, t)\\}}{1 + \max\\{in(s, t), out(s, t)\\}}`,

    :math:`in(s, t) =` number of paths from t to s,

    :math:`out(s, t) =` number of paths from s to t.

    Args:
        source_page_ids (list[int]): List of source page ids.
        target_page_ids (list[int]): List of target page ids.

    Returns:
        list[dict[str]]: A list with all possible source-target pairs and their graph score. Each element of the list
        has keys 'source_page_id' (int), 'target_page_id' (int) and 'score' (float).

    Examples:
        >>> compute_graph_scores([6220, 1196], [18973446, 9417, 946975])
        [
            {
                'source_page_id': 6220,
                'target_page_id': 18973446,
                'score': 0.6187232849309675
            },
            {
                'source_page_id': 6220,
                'target_page_id': 9417,
                'score': 0.627444454118999
            },
            {
                'source_page_id': 6220,
                'target_page_id': 946975,
                'score': 0.6114293794482915
            },
            {
                'source_page_id': 1196,
                'target_page_id': 18973446,
                'score': 0.4912413298812224
            },
            {
                'source_page_id': 1196,
                'target_page_id': 9417,
                'score': 0.3372925094508338
            },
            {
                'source_page_id': 1196,
                'target_page_id': 946975,
                'score': 0.36674985172844654
            }
        ]
    """
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
        score = compute_score(n_out_paths, n_in_paths)

        # Append result
        results.append({
            'source_page_id': s,
            'target_page_id': t,
            'score': score
        })

    return results


def compute_graph_similarity(source_page_ids, target_page_ids):
    graph_scores = compute_graph_scores(source_page_ids, target_page_ids)

    scores = [pair['score'] for pair in graph_scores]

    return np.median(scores)

