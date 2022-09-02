import numpy as np
import pandas as pd

from interfaces.db import DB

from utils.text.io import log
from utils.time.stopwatch import Stopwatch


def weight(x, epsilon=0.01):
    # Weight function is tanh(k*x), where k is inferred from data in such a way that,
    # if m is the median of x, 2m is mapped to 1 - epsilon, for a given epsilon (e.g. 0.01).
    #
    # This dynamic choice provides a sensible mapping from the given values to [0, 1].
    # Fixing the same k for all weights would more likely map values unevenly onto [0, 1]
    # (e.g. with most values >0.95), which is not a desired behaviour.
    #
    # As an example, if the median of x is 120, then we have the following weights:
    #   - 0   |---> 0
    #   - 20  |---> 0.217
    #   - 60  |---> 0.579
    #   - 120 |---> 0.868
    #   - 180 |---> 0.963
    #   - 240 |---> 0.990
    #   - 300 |---> 0.997

    k = np.log(2/epsilon - 1) / (4 * x.median())
    return np.tanh(k * x)


def main():
    # Initialize stopwatch to keep track of time
    sw = Stopwatch()

    # Instantiate db interface to communicate with database
    db = DB()

    ############################################################
    # FETCH GRAPH FROM DATABASE                                #
    ############################################################

    log('Fetching investor-investor edges from database...')

    table_name = 'ca_temp.Edges_N_Investor_N_Investor'
    fields = ['SourceInvestorID', 'TargetInvestorID', 'ScoreQuadCount']
    df = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)

    # Add weight column
    df['Weight'] = weight(df['ScoreQuadCount'])
    df = df.drop('ScoreQuadCount', axis=1)

    # Duplicate as to have all edges and not only in one direction
    reversed_df = df.copy()
    reversed_df[['SourceInvestorID', 'TargetInvestorID']] = df[['TargetInvestorID', 'SourceInvestorID']]
    investors_investors = pd.concat([df, reversed_df]).reset_index(drop=True)

    log('investors_investors')
    log(investors_investors)

    log(f'    {sw.delta():.3f}s', color='green')

    ############################################################

    log('Fetching investor-concept edges from database...')

    table_name = 'ca_temp.Edges_N_Investor_N_Concept'
    fields = ['InvestorID', 'PageID', 'ScoreQuadCount']
    investors_concepts = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)

    # Add weight column
    investors_concepts['Weight'] = weight(investors_concepts['ScoreQuadCount'])
    investors_concepts = investors_concepts.drop('ScoreQuadCount', axis=1)

    log('investors_concepts')
    log(investors_concepts)

    log(f'    {sw.delta():.3f}s', color='green')

    ############################################################

    log('Fetching concept-concept edges from database...')

    table_name = 'graph.Edges_N_Concept_N_Concept_T_GraphScore'
    fields = ['SourcePageID', 'TargetPageID', 'NormalisedScore']
    concept_ids = list(investors_concepts['PageID'].drop_duplicates())
    conditions = {'SourcePageID': concept_ids, 'TargetPageID': concept_ids}
    concepts_concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

    # Add weight column
    concepts_concepts['Weight'] = weight(concepts_concepts['NormalisedScore'])
    concepts_concepts = concepts_concepts.drop('NormalisedScore', axis=1)

    log('concepts_concepts')
    log(concepts_concepts)

    log(f'    {sw.delta():.3f}s', color='green')

    ############################################################

    log('Fetching investor nodes from database...')

    table_name = 'ca_temp.Nodes_N_Investor'
    fields = ['InvestorID', 'ScoreQuadCount']
    investors = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)

    # Add weight column
    investors['Weight'] = weight(investors['ScoreQuadCount'])
    investors = investors.drop('ScoreQuadCount', axis=1)

    log('investors')
    log(investors)

    log(f'    {sw.delta():.3f}s', color='green')

    ############################################################

    log('Fetching concept nodes from database...')

    table_name = 'ca_temp.Nodes_N_Concept'
    fields = ['InvesteeID', 'ScoreQuadCount']   # TODO correct this, should be PageID instead of InvesteeID
    concepts = pd.DataFrame(db.find(table_name, fields=fields), columns=['PageID', 'ScoreQuadCount'])    # TODO correct this, should be fields instead of given list

    # Add weight column
    concepts['Weight'] = weight(concepts['ScoreQuadCount'])
    concepts = concepts.drop('ScoreQuadCount', axis=1)

    log('concepts')
    log(concepts)

    log(f'    {sw.delta():.3f}s', color='green')

    ############################################################
    # PREPARATION OF POTENTIAL EDGES                           #
    ############################################################

    # Potential edges are pairs investor-concept (i, c) at distance at most 2 in the graph.
    #   That means (i, c) is a potential edge for each investor i and for each
    #   concept c in the set C(I(i)) U C(C(i)), where I and C are defined as follows:
    #       - I(i) is the set of investors j such that (i, j) is an (investor-investor) edge
    #       - C(i) is the set of concepts c such that (i, c) is an (investor-concept) edge
    #       - C(c) is the set of concepts d such that (c, d) is an (concept-concept) edge

    # Prepare dataframe with all triples investor-investor-concept (i, j, c) such that i -> j -> c, in a uniform format

    # Add source investor weight
    df = pd.merge(investors_investors, investors, how='inner', left_on='SourceInvestorID', right_on='InvestorID')
    df = df.drop('InvestorID', axis=1)
    df = df.rename(columns={'SourceInvestorID': 'SourceID', 'TargetInvestorID': 'PivotID', 'Weight_x': 'WeightSourcePivot', 'Weight_y': 'WeightSource'})

    # Add target investor weight
    df = pd.merge(df, investors, how='inner', left_on='PivotID', right_on='InvestorID')
    df = df.drop('InvestorID', axis=1)
    df = df.rename(columns={'Weight': 'WeightPivot'})

    log('df')
    log(df)

    # TODO continue here merging df with investors_concepts
    # df = pd.merge(investors_investors, investors_concepts, how='inner', left_on='TargetInvestorID', right_on='InvestorID')

    log('df')
    log(df)

    # Prepare dataframe with all triples investor-concept-concept (i, d, c) such that i -> d -> c, in a uniform format
    # df = pd.merge(investors_concepts, concepts_concepts, how='inner', left_on='PageID', right_on='SourcePageID')


    log(f'    {sw.delta():.3f}s', color='green')

    ############################################################

    sw.report(laps=False)


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    main()
