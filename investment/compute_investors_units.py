import pandas as pd
import numpy as np

from interfaces.db import DB

from utils.breadcrumb import Breadcrumb


def mix(x, concepts_concepts):
    """
    One-step mixing of edges through concepts.

    Args:
        x (pd.DataFrame): DataFrame including columns PageID and Score
        concepts_concepts (pd.DataFrame): Dataframe including columns SourcePageID, TargetPageID and Score

    Returns (pd.DataFrame): DataFrame with the same columns as x, the contents of which are the one-step mixing
        of x following edges in concepts_concepts. This is typically larger than the original DataFrame.
    """

    # Extract key
    key = [c for c in x.columns if c != 'Score']

    # Merge both DataFrames through concept-concept edges considering both forward and backward edges
    x = pd.concat([
        pd.merge(
            x,
            concepts_concepts.rename(columns={'SourcePageID': 'PageID', 'Score': 'EdgeScore'}),
            how='inner',
            on='PageID'
        ),
        pd.merge(
            x,
            concepts_concepts.rename(
                columns={'TargetPageID': 'PageID', 'SourcePageID': 'TargetPageID', 'Score': 'EdgeScore'}),
            how='inner',
            on='PageID'
        )
    ])

    # Take the geometric mean of both scores, the original one XXXX-concept and the concept-concept one
    x['Score'] = np.sqrt(x['Score'] * x['EdgeScore'])

    # Drop and rename columns to match again the columns of x
    x.drop(columns=['PageID', 'EdgeScore'], inplace=True)
    x = x.rename(columns={'TargetPageID': 'PageID'})

    # Compute the scores for each of the new pages by averaging out the scores over all neighbours
    x.groupby(by=key).aggregate(Score=('Score', 'mean')).reset_index()

    return x


def combine(base, x, y):
    """
    Combines different edge scores through concepts. For the pairs specified in base, it computes a score by combining
    the scores in x and y.

    Args:
        base (pd.DataFrame): DataFrame including the columns in x and y, except PageID and Score
        x (pd.DataFrame): DataFrame including columns PageID and Score
        y (pd.DataFrame): DataFrame including columns PageID and Score

    Returns (pd.DataFrame): The base DataFrame with an extra column Score.
    """
    # Extract keys
    key_x = [c for c in x.columns if c not in ['PageID', 'Score']]
    key_y = [c for c in y.columns if c not in ['PageID', 'Score']]

    # Add squared norm for x (sum of all scores)
    x = pd.merge(
        x,
        x.groupby(by=key_x).aggregate(SqNorm=('Score', 'sum')).reset_index(),
        how='inner',
        on=key_x
    )

    # Add squared norm for y (sum of all scores)
    y = pd.merge(
        y,
        y.groupby(by=key_y).aggregate(SqNorm=('Score', 'sum')).reset_index(),
        how='inner',
        on=key_y
    )

    # Merge both into base
    base = pd.merge(
        base,
        x,
        how='inner',
        on=key_x
    )
    base = pd.merge(
        base,
        y,
        how='inner',
        on=(key_y + ['PageID'])
    )

    # Compute affinity score and renormalize it
    base['Score'] = np.sqrt(base['Score_x'] * base['Score_y']) / np.sqrt(base['SqNorm_x'] * base['SqNorm_y'])
    base = base.groupby(by=(key_x + key_y)).aggregate(Score=('Score', 'sum')).reset_index()

    base['Score'] = np.sqrt(base['Score'])

    return base


def compute_affinities(investors_units, investors_concepts, units_concepts, concepts_concepts):

    investors_units = investors_units.drop(columns=['Score'])

    aaa = combine(investors_units, investors_concepts, units_concepts)

    print(aaa)
    print(aaa.describe())

    mixed_investors_concepts = mix(investors_concepts, concepts_concepts)
    mixed_units_concepts = mix(units_concepts, concepts_concepts)

    bbb = combine(investors_units, mixed_investors_concepts, mixed_units_concepts)

    print(bbb)
    print(bbb.describe())


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

    # Normalise units-concepts scores
    units_concepts = pd.merge(
        units_concepts,
        units_concepts.groupby('UnitID').aggregate(MaxScore=('Score', 'max')).reset_index(),
        how='left',
        on='UnitID'
    )
    units_concepts['Score'] = units_concepts['Score'] / units_concepts['MaxScore']
    units_concepts = units_concepts.drop(columns='MaxScore')
    unit_ids = list(units_concepts['UnitID'].drop_duplicates())

    ############################################################

    bc.log('Fetching investor-concept Jaccard edges from database...')

    table_name = 'ca_temp.Edges_N_Investor_N_Concept_T_Jaccard'
    fields = ['InvestorID', 'PageID', 'Jaccard_110']
    investors_concepts_jaccard = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)

    ############################################################

    bc.log('Deriving investors-units edges from investors-concepts and units-concepts...')

    investors_units = pd.merge(investors_concepts_jaccard, units_concepts, how='inner', on='PageID')
    investors_units['Score'] = investors_units['Score'] * investors_units['Jaccard_110']
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

    ############################################################

    bc.log('Computing affinities investors-units...')

    compute_affinities(investors_units, investors_concepts, units_concepts, concepts_concepts)

    ############################################################

    bc.report()


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    main()
