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
        x. The score of a concept in the mixed configuration is the arithmetic mean of the geometric means of the
        configuration score and the edge score in the neighbourhood of the concept.
    """

    # Extract key
    key = [c for c in x.columns if c not in ['PageID', 'Score']]

    mixed = x.copy()

    # Merge both DataFrames through concept-concept edges considering both forward and backward edges
    mixed = pd.concat([
        pd.merge(
            mixed,
            edges.rename(columns={'SourcePageID': 'PageID', 'Score': 'EdgeScore'}),
            how='inner',
            on='PageID'
        ),
        pd.merge(
            mixed,
            edges.rename(columns={'TargetPageID': 'PageID', 'SourcePageID': 'TargetPageID', 'Score': 'EdgeScore'}),
            how='inner',
            on='PageID'
        )
    ])

    # Take the geometric mean of both scores, from the configuration and from the edge
    mixed['Score'] = np.sqrt(mixed['Score'] * mixed['EdgeScore'])

    # Drop and rename columns to match again the original columns
    mixed = mixed.drop(columns=['PageID', 'EdgeScore'])
    mixed = mixed.rename(columns={'TargetPageID': 'PageID'})
    mixed = mixed[key + ['PageID', 'Score']].reset_index(drop=True)

    # Compute the scores for each of the new pages by averaging out the scores over all neighbours
    mixed = mixed.groupby(by=(key + ['PageID'])).aggregate(Score=('Score', 'mean')).reset_index()

    # # Filter scores below threshold to avoid memory issues with long tails
    # mixed = mixed[mixed['Score'] >= 0.2].reset_index(drop=True)

    return mixed


def compute_affinities(x, y, pairs, edges):
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
            weighted edges of the concepts graph.

    Returns (pd.DataFrame): DataFrame with columns key_x + key_y + ['Score'], containing the same rows as pairs.
        For each pair of configuration X and Y, their score is computed as the square root of the ratio of the norm
        of BX*BY squared and the product of norms of BX and BY. Here BX and BY denote the mixing of X and Y,
        respectively, and BX*BY denotes the combination of BX and BY.
    """

    # Extract keys
    key_x = [c for c in x.columns if c not in ['PageID', 'Score']]
    key_y = [c for c in y.columns if c not in ['PageID', 'Score']]

    # Compute mixings BX and BY, combination BX*BY and their norms
    bx = mix(x, edges)
    by = mix(y, edges)
    bxby = combine(bx, by, pairs)
    norms_bx = norm(bx)
    norms_by = norm(by)
    norms_bxby = norm(bxby)

    # Merge norms for BX, BY and BX*BY into pairs DataFrame
    pairs = pd.merge(
        pairs,
        norms_bx.rename(columns={'Norm': 'NormBX'}),
        how='left',
        on=key_x
    )

    pairs = pd.merge(
        pairs,
        norms_by.rename(columns={'Norm': 'NormBY'}),
        how='left',
        on=key_y
    )

    pairs = pd.merge(
        pairs,
        norms_bxby.rename(columns={'Norm': 'NormBXBY'}),
        how='left',
        on=(key_x + key_y)
    )

    # Compute affinity scores
    pairs = pairs.fillna(0)
    pairs['Score'] = np.sqrt(np.square(pairs['NormBXBY']) / (pairs['NormBX'] * pairs['NormBY']))

    return pairs[key_x + key_y + ['Score']]


def main():

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

    bc.log('Fetching concept-concept edges from database...')

    table_name = 'graph.Edges_N_Concept_N_Concept_T_GraphScore'
    fields = ['SourcePageID', 'TargetPageID', 'NormalisedScore']
    concepts_concepts = pd.DataFrame(db.find(table_name, fields=fields), columns=['SourcePageID', 'TargetPageID', 'Score'])

    # Consider only edges above a threshold to avoid too intensive memory usage
    # concepts_concepts = concepts_concepts[concepts_concepts['Score'] >= 0.1].reset_index(drop=True)

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
    investors_units = compute_affinities(investors_concepts, units_concepts, investors_units, concepts_concepts)

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

    main()
