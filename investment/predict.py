import numpy as np
import pandas as pd

from interfaces.db import DB

from utils.breadcrumb import Breadcrumb


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
    # Initialize breadcrumb to log and keep track of time
    bc = Breadcrumb()

    # Instantiate db interface to communicate with database
    db = DB()

    ############################################################
    # FETCH GRAPH FROM DATABASE                                #
    ############################################################

    bc.log('Fetching investor-investor edges from database...')

    table_name = 'ca_temp.Edges_N_Investor_N_Investor'
    fields = ['SourceInvestorID', 'TargetInvestorID', 'ScoreQuadCount']
    df = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)

    # Add weight column
    df['Weight'] = weight(df['ScoreQuadCount'])
    df = df.drop('ScoreQuadCount', axis=1)

    # Filter out edges with 0 weight
    df = df[df['Weight'] > 0]

    # Duplicate as to have all edges and not only in one direction
    reversed_df = df.copy()
    reversed_df[['SourceInvestorID', 'TargetInvestorID']] = df[['TargetInvestorID', 'SourceInvestorID']]
    investors_investors = pd.concat([df, reversed_df]).reset_index(drop=True)

    del df, reversed_df

    ############################################################

    bc.log('Fetching investor-concept edges from database...')

    table_name = 'ca_temp.Edges_N_Investor_N_Concept'
    fields = ['InvestorID', 'PageID', 'ScoreQuadCount']
    investors_concepts = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)
    investors_concepts['PageID'] = investors_concepts['PageID'].astype(str)

    # Add weight column and filter out edges with 0 weight
    investors_concepts['Weight'] = weight(investors_concepts['ScoreQuadCount'])
    investors_concepts = investors_concepts.drop('ScoreQuadCount', axis=1)

    # Filter out edges with 0 weight
    investors_concepts = investors_concepts[investors_concepts['Weight'] > 0]

    ############################################################

    bc.log('Fetching concept-concept edges from database...')

    table_name = 'graph.Edges_N_Concept_N_Concept_T_GraphScore'
    fields = ['SourcePageID', 'TargetPageID', 'NormalisedScore']
    concept_ids = list(investors_concepts['PageID'].drop_duplicates().astype(int))
    conditions = {'SourcePageID': concept_ids, 'TargetPageID': concept_ids}
    concepts_concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)
    concepts_concepts['SourcePageID'] = concepts_concepts['SourcePageID'].astype(str)
    concepts_concepts['TargetPageID'] = concepts_concepts['TargetPageID'].astype(str)

    # Add weight column
    concepts_concepts['Weight'] = weight(concepts_concepts['NormalisedScore'])
    concepts_concepts = concepts_concepts.drop('NormalisedScore', axis=1)

    # Filter out edges with 0 weight
    concepts_concepts = concepts_concepts[concepts_concepts['Weight'] > 0]

    ############################################################

    bc.log('Fetching investor nodes from database...')

    table_name = 'ca_temp.Nodes_N_Investor'
    fields = ['InvestorID', 'ScoreQuadCount']
    investors = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)

    # Add weight column
    investors['Weight'] = weight(investors['ScoreQuadCount'])
    investors = investors.drop('ScoreQuadCount', axis=1)

    # Filter out nodes with 0 weight
    investors = investors[investors['Weight'] > 0]

    ############################################################

    bc.log('Fetching concept nodes from database...')

    table_name = 'ca_temp.Nodes_N_Concept'
    fields = ['PageID', 'ScoreQuadCount']
    concepts = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)
    concepts['PageID'] = concepts['PageID'].astype(str)

    # Add weight column
    concepts['Weight'] = weight(concepts['ScoreQuadCount'])
    concepts = concepts.drop('ScoreQuadCount', axis=1)

    # Filter out nodes with 0 weight
    concepts = concepts[concepts['Weight'] > 0]

    ############################################################
    # POTENTIAL EDGES - PREPARATION                            #
    ############################################################

    # Potential edges are pairs investor-concept (i, c) at distance at most 2 in the graph.
    #   That means (i, c) is a potential edge for each investor i and for each
    #   concept c in the set C(I(i)) U C(C(i)), where I and C are defined as follows:
    #       - I(i) is the set of investors j such that (i, j) is an (investor-investor) edge
    #       - C(i) is the set of concepts c such that (i, c) is an (investor-concept) edge
    #       - C(c) is the set of concepts d such that (c, d) is an (concept-concept) edge

    ############################################################

    bc.log('Merging investor-investor edges with investor-concept edges...')

    inv_inv_cpt = pd.merge(investors_investors, investors_concepts, how='inner', left_on='TargetInvestorID', right_on='InvestorID')
    inv_inv_cpt = inv_inv_cpt.drop('InvestorID', axis=1)
    inv_inv_cpt = inv_inv_cpt.rename(columns={
        'SourceInvestorID': 'SourceID',
        'TargetInvestorID': 'PivotID',
        'PageID': 'TargetID',
        'Weight_x': 'SourcePivotWeight',
        'Weight_y': 'PivotTargetWeight'
    })

    ############################################################

    bc.log('Merging investor-concept edges with concept-concept edges...')

    inv_cpt_cpt = pd.merge(investors_concepts, concepts_concepts, how='inner', left_on='PageID', right_on='SourcePageID')
    inv_cpt_cpt = inv_cpt_cpt.drop('PageID', axis=1)
    inv_cpt_cpt = inv_cpt_cpt.rename(columns={
        'InvestorID': 'SourceID',
        'SourcePageID': 'PivotID',
        'TargetPageID': 'TargetID',
        'Weight_x': 'SourcePivotWeight',
        'Weight_y': 'PivotTargetWeight'
    })

    ############################################################

    bc.log('Preparing dataframe with potential edges...')

    potential_edges = pd.concat([inv_inv_cpt[['SourceID', 'TargetID']], inv_cpt_cpt[['SourceID', 'TargetID']]]).drop_duplicates()

    ############################################################
    # POTENTIAL EDGES - INTERSECTION TERMS                     #
    ############################################################

    bc.log('Adding investor pivot weight...')

    inv_inv_cpt = pd.merge(inv_inv_cpt, investors, how='left', left_on='PivotID', right_on='InvestorID')
    inv_inv_cpt = inv_inv_cpt.drop('InvestorID', axis=1)
    inv_inv_cpt = inv_inv_cpt.rename(columns={'Weight': 'PivotWeight'})

    ############################################################

    bc.log('Adding concept pivot weight...')

    inv_cpt_cpt = pd.merge(inv_cpt_cpt, concepts, how='left', left_on='PivotID', right_on='PageID')
    inv_cpt_cpt = inv_cpt_cpt.drop('PageID', axis=1)
    inv_cpt_cpt = inv_cpt_cpt.rename(columns={'Weight': 'PivotWeight'})

    ############################################################

    bc.log('Computing intersection term T^I(i, c)...')

    # For given investor i and concept c, their intersection term T^I(i, c) is defined as
    #   T^I(i, c) = sum_{i~>j~>c} w(j) * (w(i, j) + w(j, c)) / 2

    inv_inv_cpt['IntersectionTerm_TIic'] = inv_inv_cpt['PivotWeight'] * (inv_inv_cpt['SourcePivotWeight'] + inv_inv_cpt['PivotTargetWeight']) / 2
    df = inv_inv_cpt[['SourceID', 'TargetID', 'IntersectionTerm_TIic']].groupby(by=['SourceID', 'TargetID']).sum().reset_index()
    potential_edges = pd.merge(potential_edges, df, how='left', on=['SourceID', 'TargetID'])
    potential_edges = potential_edges.fillna(0)

    del df

    ############################################################

    bc.log('Computing intersection term T^C(i, c)...')

    # For given investor i and concept c, their intersection term T^C(i, c) is defined as
    #   T^C(i, c) = sum_{i~>d~>c} w(d) * (w(i, d) + w(d, c)) / 2

    inv_cpt_cpt['IntersectionTerm_TCic'] = inv_cpt_cpt['PivotWeight'] * (inv_cpt_cpt['SourcePivotWeight'] + inv_cpt_cpt['PivotTargetWeight']) / 2
    df = inv_cpt_cpt[['SourceID', 'TargetID', 'IntersectionTerm_TCic']].groupby(by=['SourceID', 'TargetID']).sum().reset_index()
    potential_edges = pd.merge(potential_edges, df, how='left', on=['SourceID', 'TargetID'])
    potential_edges = potential_edges.fillna(0)

    del df

    ############################################################
    # POTENTIAL EDGES - BASE TERMS                             #
    ############################################################

    bc.log('Adding target node weight to investor-investor edges...')

    investors_investors = pd.merge(investors_investors, investors, how='inner', left_on='TargetInvestorID', right_on='InvestorID')
    investors_investors = investors_investors.drop('InvestorID', axis=1)
    investors_investors = investors_investors.rename(columns={'Weight_x': 'EdgeWeight', 'Weight_y': 'TargetWeight'})

    ############################################################

    bc.log('Adding investor and concept node weights to investor-concept edges...')

    investors_concepts = pd.merge(investors_concepts, investors, how='inner', on='InvestorID')
    investors_concepts = investors_concepts.rename(columns={'Weight_x': 'EdgeWeight', 'Weight_y': 'SourceWeight'})

    investors_concepts = pd.merge(investors_concepts, concepts, how='inner', on='PageID')
    investors_concepts = investors_concepts.rename(columns={'Weight': 'TargetWeight'})

    ############################################################

    bc.log('Adding source node weight to concept-concept edges...')

    concepts_concepts = pd.merge(concepts_concepts, concepts, how='inner', left_on='SourcePageID', right_on='PageID')
    concepts_concepts = concepts_concepts.drop('PageID', axis=1)
    concepts_concepts = concepts_concepts.rename(columns={'Weight_x': 'EdgeWeight', 'Weight_y': 'SourceWeight'})

    ############################################################

    bc.log('Computing base term T^I(i)...')

    # For a given investor i, its base term T^I(i) is defined as
    #   T^I(i) = sum_{i~>j} w(j) * w(i, j)

    investors_investors['BaseTerm_TIi'] = investors_investors['EdgeWeight'] * investors_investors['TargetWeight']
    df = investors_investors[['SourceInvestorID', 'BaseTerm_TIi']].groupby(by='SourceInvestorID').sum().reset_index()
    df = df.rename(columns={'SourceInvestorID': 'SourceID'})
    potential_edges = pd.merge(potential_edges, df, how='left', on='SourceID')
    potential_edges = potential_edges.fillna(0)

    del df

    ############################################################

    bc.log('Computing base term T^C(i)...')

    # For a given investor i, its base term T^C(i) is defined as
    #   T^C(i) = sum_{i~>d} w(d) * w(i, d)

    investors_concepts['BaseTerm_TCi'] = investors_concepts['EdgeWeight'] * investors_concepts['TargetWeight']
    df = investors_concepts[['InvestorID', 'BaseTerm_TCi']].groupby(by='InvestorID').sum().reset_index()
    df = df.rename(columns={'InvestorID': 'SourceID'})
    potential_edges = pd.merge(potential_edges, df, how='left', on='SourceID')
    potential_edges = potential_edges.fillna(0)

    del df

    ############################################################

    bc.log('Computing base term T^I(c)...')

    # For a given concept c, its base term T^I(c) is defined as
    #   T^I(c) = sum_{j~>c} w(j) * w(j, c)

    investors_concepts['BaseTerm_TIc'] = investors_concepts['EdgeWeight'] * investors_concepts['SourceWeight']
    df = investors_concepts[['PageID', 'BaseTerm_TIc']].groupby(by='PageID').sum().reset_index()
    df = df.rename(columns={'PageID': 'TargetID'})
    potential_edges = pd.merge(potential_edges, df, how='left', on='TargetID')
    potential_edges = potential_edges.fillna(0)

    del df

    ############################################################

    bc.log('Computing base term T^C(c)...')

    # For a given concept c, its base term T^C(c) is defined as
    #   T^C(c) = sum_{d~>c} w(d) * w(d, c)

    concepts_concepts['BaseTerm_TCc'] = concepts_concepts['EdgeWeight'] * concepts_concepts['SourceWeight']
    df = concepts_concepts[['TargetPageID', 'BaseTerm_TCc']].groupby(by='TargetPageID').sum().reset_index()
    df = df.rename(columns={'TargetPageID': 'TargetID'})
    potential_edges = pd.merge(potential_edges, df, how='left', on='TargetID')
    potential_edges = potential_edges.fillna(0)

    del df

    ############################################################
    # POTENTIAL EDGES - WEIGHTED JACCARD COEFFICIENT           #
    ############################################################

    bc.log('Computing Jaccard coefficient...')

    # For given investor i and concept c, their weighted Jaccard coefficient WJC(i, c) is defined as
    #   WJC(i, c) = (T^I(i, c) + T^C(i, c)) / (T^I(i) + T^C(i) + T^I(c) + T^C(c) - T^I(i, c) - T^C(i, c))

    intersection_terms = potential_edges['IntersectionTerm_TIic'] + potential_edges['IntersectionTerm_TCic']
    base_terms = potential_edges['BaseTerm_TIi'] + potential_edges['BaseTerm_TCi'] + potential_edges['BaseTerm_TIc'] + potential_edges['BaseTerm_TCc']
    potential_edges['JaccardCoef'] = intersection_terms / (base_terms - intersection_terms)
    potential_edges = potential_edges.fillna(0)
    potential_edges = potential_edges.replace([np.inf, -np.inf], 1)

    ############################################################

    print(potential_edges.sort_values(by='JaccardCoef', ascending=False))

    ############################################################

    bc.report()


if __name__ == '__main__':
    pd.set_option('display.max_rows', 400)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    main()
