import pandas as pd
import numpy as np


def norm(x):
    """
    Computes the norm of the different configurations in x.

    Args:
        x (pd.DataFrame): DataFrame whose columns are key + ['PageID', 'Score']. Each configuration of scores is given
            by a unique tuple of values for columns in key. For example, if x has columns
            ['InvestorID', 'PageID', 'Score'], then each value of InvestorID is assumed to index a configuration,
            given by the columns ['PageID', 'Score'].

    Returns (pd.DataFrame): DataFrame with columns key + ['Norm'], containing the norm of each configuration, computed
        as \\sqrt{\\sum_{c \\in C} X(c)^2}, where C is the set of concepts and X: C \to [0, 1] is a given configuration.
    """

    # Extract key indexing each configuration
    key = [c for c in x.columns if c not in ['PageID', 'Score']]

    norms = x.copy()

    # Square all scores to sum them
    norms['Norm'] = np.square(norms['Score'])

    # Group by key and add the squares of the scores
    norms = norms.groupby(by=key).aggregate(Norm=('Norm', 'sum')).reset_index()

    # Take the square root to get the norm
    norms['Norm'] = np.sqrt(norms['Norm'])

    return norms


def normalise(x):
    """
    Returns the same set of configurations as in x, with their scores divided by each norm. Hence, all configurations
    in x have norm 1.

    Args:
        x (pd.DataFrame): DataFrame whose columns are key + ['PageID', 'Score']. Each configuration of scores is given
            by a unique tuple of values for columns in key.

    Returns (pd.DataFrame): DataFrame with the same columns as x, containing configurations indexed by the same set as
        x. The score of a concept in the returned configuration is the score it has in x divided by the norm of the
        configuration.
    """

    # Extract key
    key = [c for c in x.columns if c not in ['PageID', 'Score']]

    # Compute configuration norms
    norms = norm(x)

    # Add norms to x
    x = pd.merge(x, norms, how='inner', on=key)

    # Divide scores by norm
    x['Score'] = x['Score'] / x['Norm']

    # Keep only relevant columns
    x = x[key + ['PageID', 'Score']]

    return x


def mix(x, edges, min_ratio=0.05):
    """
    Mixes the configurations in x according to the edges in edges.

    Args:
        x (pd.DataFrame): DataFrame whose columns are key + ['PageID', 'Score']. Each configuration of scores is given
            by a unique tuple of values for columns in key. For example, if x has columns
            ['InvestorID', 'PageID', 'Score'], then each value of InvestorID is assumed to index a configuration, given
            by the columns ['PageID', 'Score'].
        edges (pd.DataFrame): DataFrame whose columns are ['SourcePageID', 'TargetPageID', 'Score'], which define the
            weighted edges of the concepts graph.
        min_ratio (float): For every resulting configuration, only concepts whose ratio of score over maximum score
            is above min_ratio are kept. If set to 0, then all concepts are kept.

    Returns (pd.DataFrame): DataFrame with the same columns as x, containing configurations indexed by the same set as
        x. The score of a concept in the mixed configuration is the arithmetic mean of the products of the
        configuration score and the edge score in the 1-ball of the concept, assuming every concept has a loop
        with score 1.
    """

    # Extract key
    key = [c for c in x.columns if c not in ['PageID', 'Score']]

    mixed = pd.merge(
        x.rename(columns={'PageID': 'SourcePageID', 'Score': 'VertexScore'}),
        edges.rename(columns={'TargetPageID': 'PageID', 'Score': 'EdgeScore'}),
        how='inner',
        on='SourcePageID'
    )

    # Multiply vertex score with edge score
    mixed['Score'] = mixed['VertexScore'] * mixed['EdgeScore']

    # Add ball sizes for the average to take all vertices into account
    ball_sizes = edges.groupby(by='SourcePageID').aggregate(BallSize=('TargetPageID', 'count')).reset_index()
    ball_sizes = ball_sizes.rename(columns={'SourcePageID': 'PageID'})
    mixed = pd.merge(
        mixed,
        ball_sizes,
        how='left',
        on='PageID'
    )

    # Average all scores in the 1-ball of each vertex
    mixed['Score'] = mixed['Score'] / mixed['BallSize']
    mixed = mixed.groupby(by=(key + ['PageID'])).aggregate(Score=('Score', 'sum')).reset_index()

    # Filter low scores to avoid memory issues
    if min_ratio > 0:
        mixed = pd.merge(
            mixed,
            mixed.groupby(by=key).aggregate(MaxScore=('Score', 'max')).reset_index(),
            how='inner',
            on=key
        )
        mixed = mixed[mixed['Score'] >= min_ratio * mixed['MaxScore']].reset_index(drop=True)
        mixed = mixed[key + ['PageID', 'Score']]

    return mixed


def normalise_graph(edges):
    """
    Adds missing reverse edges and averages scores. Adds loops on each vertex with a score of one.

    Args:
        edges (pd.DataFrame): DataFrame whose columns are ['SourcePageID', 'TargetPageID', 'Score'], which define the
            weighted edges of the concepts graph.

    Returns (pd.DataFrame): DataFrame whose columns are ['SourcePageID', 'TargetPageID', 'Score'], with each pair in
        both directions and with loops on every vertex with a score of 1.
    """

    # Remove present loops as they will be replaced with a score of one
    normalised_edges = edges[edges['SourcePageID'] != edges['TargetPageID']].reset_index(drop=True)

    # Extract unique vertices
    vertices = pd.concat([
        normalised_edges['SourcePageID'],
        normalised_edges['TargetPageID']
    ]).drop_duplicates().reset_index(drop=True)

    # Add reverse edges
    reverse_edges = normalised_edges.copy()
    reverse_edges[['SourcePageID', 'TargetPageID']] = normalised_edges[['TargetPageID', 'SourcePageID']]
    normalised_edges = pd.concat([normalised_edges, reverse_edges]).reset_index(drop=True)

    # Average scores of forward and backward edges
    normalised_edges = normalised_edges.groupby(by=['SourcePageID', 'TargetPageID']).aggregate({'Score': 'mean'}).reset_index()

    # Add loops with score of 1
    loops = pd.DataFrame({'SourcePageID': vertices, 'TargetPageID': vertices, 'Score': 1})
    normalised_edges = pd.concat([normalised_edges, loops]).reset_index(drop=True)

    return normalised_edges


def combine(x, y, pairs):
    """
    Combines the configurations in x and y based on the associations in pairs.

    Args:
        x (pd.DataFrame): DataFrame whose columns are key_x + ['PageID', 'Score']. Each configuration of scores is
            given by a unique tuple of values for columns in key_x. For example, if x has columns
            ['InvestorID', 'PageID', 'Score'], then each value of InvestorID is assumed to index a configuration,
            given by the columns ['PageID', 'Score'].
        y (pd.DataFrame): DataFrame whose columns are key_y + ['PageID', 'Score']. Each configuration of scores is
            given by a unique tuple of values for columns in key_y. For example, if y has columns
            ['InvestorID', 'PageID', 'Score'], then each value of InvestorID is assumed to index a configuration,
            given by the columns ['PageID', 'Score'].
        pairs (pd.DataFrame): DataFrame whose columns are key_x + key_y. Configurations in x and y are compared
            based on the associations in this DataFrame.

    Returns (pd.DataFrame): DataFrame with columns key_x + key_y + ['PageID', 'Score']. For each row in pairs, there is
        a configuration of scores in the returned DataFrame. The score for each concept is the geometric mean of the
        scores in x and y.
    """

    # Extract keys
    key_x = [c for c in x.columns if c not in ['PageID', 'Score']]
    key_y = [c for c in y.columns if c not in ['PageID', 'Score']]

    combination = pairs.copy()

    # Merge x and y
    combination = pd.merge(combination, x, how='inner', on=key_x)
    combination = pd.merge(combination, y, how='inner', on=(key_y + ['PageID']))

    # Compute scores
    combination['Score'] = np.sqrt(combination['Score_x'] * combination['Score_y'])

    return combination[key_x + key_y + ['PageID', 'Score']]


def compute_affinities(x, y, pairs, edges=None, mix_x=False, mix_y=False, normalise_before=False, method='cosine', k=1):
    """
    Computes affinity scores between the pairs configurations in x and y indexed in pairs, according to the concepts
    (edge-weighted) graph specified in edges.

    Args:
        x (pd.DataFrame): DataFrame whose columns are key_x + ['PageID', 'Score']. Each configuration of scores is
            given by a unique tuple of values for columns in key_x. For example, if x has columns
            ['InvestorID', 'PageID', 'Score'], then each value of InvestorID is assumed to index a configuration,
            given by the columns ['PageID', 'Score'].
        y (pd.DataFrame): DataFrame whose columns are key_y + ['PageID', 'Score']. Each configuration of scores is
            given by a unique tuple of values for columns in key_y. For example, if y has columns
            ['InvestorID', 'PageID', 'Score'], then each value of InvestorID is assumed to index a configuration,
            given by the columns ['PageID', 'Score'].
        pairs (pd.DataFrame): DataFrame whose columns are key_x + key_y. Configurations in x and y are compared
            based on the associations in this DataFrame.
        edges (pd.DataFrame): DataFrame whose columns are ['SourcePageID', 'TargetPageID', 'Score'], which define the
            weighted edges of the concepts graph. Only required if one of mix_x or mix_y are True.
        mix_x (bool): Whether to replace x with its mixing before affinity computation. Recommended to set to True if
            the configurations in x have a low number of concepts. If set to True, then edges is required.
        mix_y (bool): Whether to replace y with its mixing before affinity computation. Recommended to set to True if
            the configurations in y have a low number of concepts. If set to True, then edges is required.
        normalise_before (bool): Whether to normalise score configurations to have norm 1 before computing affinities.
        method (str): Which method to use to compute affinities. 'euclidean' uses a function based on the euclidean
            distance of each pair of configurations. 'cosine' uses a function based on cosine similarity of each pair
            of configurations. Notice that 'cosine' performs faster than 'euclidean'.
        k (float): Coefficient that controls the shape of the affinity function for method='euclidean'. It takes any
            value in (0, +inf), typical values range from 0.1 to 10. The higher the value of k, the higher the score
            the same pair of configurations will be assigned. Unused if method='cosine'.

    Returns (pd.DataFrame): DataFrame with columns key_x + key_y + ['Score'], containing the same rows as pairs.
        For each pair of configuration X and Y, their score is computed as follows:
            - If method='cosine', the score is the ratio of the norm of U*V squared (equivalently, the scalar product
                <U, V>) and the product of norms of U and V.
            - If method='euclidean', the score is 1 - tanh(k * ||U - V||), for some k > 0.
        If mix_x is True, then U is defined as the mixing of X with respect to edges, otherwise U is X.
        If mix_y is True, then V is defined as the mixing of Y with respect to edges, otherwise V is Y.
        Finally, U*V denotes the combination of U and V.
    """

    if mix_x or mix_y:
        # Check edges are provided if needed
        assert edges is not None, 'If mix_x or mix_y are set to True, edges must be provided.'

        # Make concepts graph undirected
        edges = normalise_graph(edges)

    # Extract keys
    key_x = [c for c in x.columns if c not in ['PageID', 'Score']]
    key_y = [c for c in y.columns if c not in ['PageID', 'Score']]

    # Determine U and V by computing mixings and normalising if needed
    if mix_x:
        u = mix(x, edges)
    else:
        u = x

    if mix_y:
        v = mix(y, edges)
    else:
        v = y

    if normalise_before:
        u = normalise(u)
        v = normalise(v)

    if method == 'cosine':
        # Compute combination U*V
        uv = combine(u, v, pairs)

        # Compute norms
        norms_u = norm(u)
        norms_v = norm(v)
        norms_uv = norm(uv)

        # Merge norms for U, V and U*V into pairs DataFrame
        pairs = pd.merge(pairs, norms_u.rename(columns={'Norm': 'NormU'}), how='left', on=key_x)
        pairs = pd.merge(pairs, norms_v.rename(columns={'Norm': 'NormV'}), how='left', on=key_y)
        pairs = pd.merge(pairs, norms_uv.rename(columns={'Norm': 'NormUV'}), how='left', on=(key_x + key_y))
        pairs = pairs.fillna(0)

        # Compute affinity scores
        pairs['Score'] = np.square(pairs['NormUV']) / (pairs['NormU'] * pairs['NormV'])

        # Keep only relevant columns
        pairs = pairs[key_x + key_y + ['Score']]

        return pairs

    elif method == 'euclidean':
        # Extend pairs with scores from U and V
        diff_u = pd.merge(pairs, u, how='inner', on=key_x)
        diff_v = pd.merge(pairs, v, how='inner', on=key_y)
        diff = pd.merge(diff_u, diff_v, how='outer', on=(key_x + key_y + ['PageID'])).fillna(0)

        # The score of the difference is the difference of scores
        diff['Score'] = diff['Score_x'] - diff['Score_y']

        # Compute norm of the differences
        diff = diff[key_x + key_y + ['PageID', 'Score']]
        pairs = norm(diff)

        # Score = 1 - tanh(||U - V|| / k) = 2 / (1 + exp(2 * ||U - V|| / k))
        pairs['Score'] = 2 / (1 + np.exp(2 * pairs['Norm'] / k))

        # Keep only relevant columns
        pairs = pairs[key_x + key_y + ['Score']]

        return pairs

    else:
        pairs['Score'] = 0
        return pairs
