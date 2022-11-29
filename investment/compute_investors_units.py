import pandas as pd
import numpy as np

from interfaces.db import DB

from utils.breadcrumb import Breadcrumb


def norm(x):
    """
    Computes the norm of the different configurations in x.

    Args:
        x (pd.DataFrame): DataFrame whose columns are key + ['PageID', 'Score']. Each configuration of scores is given
            by a unique tuple of values for columns in key. For example, if x has columns
            ['InvestorID', 'PageID', 'Score'], then each value of InvestorID is assumed to index a configuration,
            given by the columns ['PageID', 'Score'].

    Returns (pd.DataFrame): DataFrame with columns key + ['Norm'], containing the norm of each configuration, computed
        as \sqrt{\sum_{c \in C} X(c)^2}, where C is the set of concepts and X: C \to [0, 1] is a given configuration.
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


def mix(x, edges):
    """
    Mixes the configurations in x according to the edges in edges.

    Args:
        x (pd.DataFrame): DataFrame whose columns are key + ['PageID', 'Score']. Each configuration of scores is given
            by a unique tuple of values for columns in key. For example, if x has columns
            ['InvestorID', 'PageID', 'Score'], then each value of InvestorID is assumed to index a configuration, given
            by the columns ['PageID', 'Score'].
        edges (pd.DataFrame): DataFrame whose columns are ['SourcePageID', 'TargetPageID', 'Score'], which define the
            weighted edges of the concepts graph.

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


def compute_affinities(x, y, pairs, edges=None, mix_x=False, mix_y=False):
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

    Returns (pd.DataFrame): DataFrame with columns key_x + key_y + ['Score'], containing the same rows as pairs.
        For each pair of configuration X and Y, their score is computed as the ratio of the norm of
        U*V squared and the product of norms of U and V.
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

    # Determine U and V by computing mixings if needed
    if mix_x:
        u = mix(x, edges)
    else:
        u = x

    if mix_y:
        v = mix(y, edges)
    else:
        v = y

    # Compute combination U*V
    uv = combine(u, v, pairs)

    # Compute norms
    norms_u = norm(u)
    norms_v = norm(v)
    norms_uv = norm(uv)

    # Merge norms for U, V and U*V into pairs DataFrame
    pairs = pd.merge(
        pairs,
        norms_u.rename(columns={'Norm': 'NormU'}),
        how='left',
        on=key_x
    )

    pairs = pd.merge(
        pairs,
        norms_v.rename(columns={'Norm': 'NormV'}),
        how='left',
        on=key_y
    )

    pairs = pd.merge(
        pairs,
        norms_uv.rename(columns={'Norm': 'NormUV'}),
        how='left',
        on=(key_x + key_y)
    )

    # Compute affinity scores
    pairs = pairs.fillna(0)
    pairs['Score'] = np.square(pairs['NormUV']) / (pairs['NormU'] * pairs['NormV'])

    return pairs[key_x + key_y + ['Score']]


def compute_investors_units():

    # Initialize breadcrumb to log and keep track of time
    bc = Breadcrumb()

    # Instantiate db interface to communicate with database
    db = DB()

    ############################################################

    bc.log('Fetching unit-concept edges from database...')

    table_name = 'graph.Edges_N_Unit_N_Concept_T_Research'
    fields = ['UnitID', 'PageID', 'Score']
    units_concepts = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)

    # Renormalise units-concepts scores to add to 1
    units_concepts = pd.merge(
        units_concepts,
        units_concepts.groupby('UnitID').aggregate(SumScore=('Score', 'sum')).reset_index(),
        how='left',
        on='UnitID'
    )
    units_concepts['Score'] = units_concepts['Score'] / units_concepts['SumScore']
    units_concepts = units_concepts.drop(columns='SumScore')

    ############################################################

    bc.log('Fetching investor-concept Jaccard edges from database...')

    table_name = 'ca_temp.Edges_N_Investor_N_Concept_T_Jaccard'
    fields = ['InvestorID', 'PageID', 'Jaccard_000']
    investors_concepts_jaccard = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)

    ############################################################

    bc.log('Deriving investors-units edges from investors-concepts and units-concepts...')

    investors_units = pd.merge(investors_concepts_jaccard, units_concepts, how='inner', on='PageID')
    investors_units['Score'] = investors_units['Score'] * investors_units['Jaccard_000']
    investors_units = investors_units.groupby(by=['InvestorID', 'UnitID']).aggregate(Score=('Score', 'sum')).reset_index()
    investors_units = investors_units.sort_values(by=['UnitID', 'Score'], ascending=[True, False])
    investors_units = investors_units.groupby(by='UnitID').head(5)
    investor_ids = list(investors_units['InvestorID'].drop_duplicates())

    ############################################################

    bc.log('Fetching investor-concept yearly edges from database...')

    table_name = 'ca_temp.Edges_N_Investor_N_Concept_T_Years'
    fields = ['InvestorID', 'PageID', 'Year', 'CountAmount']
    conditions = {'InvestorID': investor_ids}
    investors_concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

    # Normalise investor-concepts scores
    investors_concepts = pd.merge(
        investors_concepts,
        investors_concepts.groupby(by=['InvestorID', 'Year']).aggregate(MaxScore=('CountAmount', 'max')).reset_index(),
        how='left',
        on=['InvestorID', 'Year']
    )
    investors_concepts['Score'] = investors_concepts['CountAmount'] / investors_concepts['MaxScore']
    investors_concepts = investors_concepts.drop(columns=['CountAmount', 'MaxScore'])

    ############################################################

    unit_concept_ids = list(units_concepts['PageID'].drop_duplicates())
    investor_concept_ids = list(investors_concepts['PageID'].drop_duplicates())
    concept_ids = list(set(unit_concept_ids + investor_concept_ids))

    ############################################################

    bc.log('Fetching concept-concept edges from database...')

    table_name = 'graph.Edges_N_Concept_N_Concept_T_GraphScore'
    fields = ['SourcePageID', 'TargetPageID', 'NormalisedScore']
    conditions = {'OR': {'SourcePageID': concept_ids, 'TargetPageID': concept_ids}}
    concepts_concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=['SourcePageID', 'TargetPageID', 'Score'])

    ############################################################

    bc.log('Adding year to investors-units...')

    investors_units = pd.merge(
        investors_concepts[['InvestorID', 'Year']].drop_duplicates(),
        investors_units,
        how='left',
        on='InvestorID'
    )

    ############################################################

    bc.log('Computing affinities investors-units...')

    investors_units = investors_units.drop(columns=['Score'])
    investors_units = compute_affinities(investors_concepts, units_concepts, investors_units, edges=concepts_concepts, mix_x=True, mix_y=True)

    ############################################################

    bc.log('Inserting investors-units edges into database...')

    table_name = 'ca_temp.Edges_N_Investor_N_Unit_T_Years'
    definition = [
        'InvestorID CHAR(64)',
        'Year SMALLINT',
        'UnitID CHAR(32)',
        'Score FLOAT',
        'KEY InvestorID (InvestorID)',
        'KEY Year (Year)',
        'KEY UnitID (UnitID)'
    ]
    db.drop_create_insert_table(table_name, definition, investors_units)

    ############################################################

    bc.report()


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    compute_investors_units()
