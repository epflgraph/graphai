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

    # Add weight column
    investors_concepts['Weight'] = weight(investors_concepts['ScoreQuadCount'])
    investors_concepts = investors_concepts.drop('ScoreQuadCount', axis=1)

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

    ############################################################

    bc.log('Fetching investor nodes from database...')

    table_name = 'ca_temp.Nodes_N_Investor'
    fields = ['InvestorID', 'ScoreQuadCount']
    investors = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)

    # Add weight column
    investors['Weight'] = weight(investors['ScoreQuadCount'])
    investors = investors.drop('ScoreQuadCount', axis=1)

    ############################################################

    bc.log('Fetching concept nodes from database...')

    table_name = 'ca_temp.Nodes_N_Concept'
    fields = ['PageID', 'ScoreQuadCount']
    concepts = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)
    concepts['PageID'] = concepts['PageID'].astype(str)

    # Add weight column
    concepts['Weight'] = weight(concepts['ScoreQuadCount'])
    concepts = concepts.drop('ScoreQuadCount', axis=1)

    ############################################################
    # PREPARATION OF POTENTIAL EDGES                           #
    ############################################################

    # Potential edges are pairs investor-concept (i, c) at distance at most 2 in the graph.
    #   That means (i, c) is a potential edge for each investor i and for each
    #   concept c in the set C(I(i)) U C(C(i)), where I and C are defined as follows:
    #       - I(i) is the set of investors j such that (i, j) is an (investor-investor) edge
    #       - C(i) is the set of concepts c such that (i, c) is an (investor-concept) edge
    #       - C(c) is the set of concepts d such that (c, d) is an (concept-concept) edge

    ############################################################

    bc.log('Adding source and target weight to investor-investor edges...')

    investors_investors = pd.merge(investors_investors, investors, how='inner', left_on='SourceInvestorID', right_on='InvestorID')
    investors_investors = investors_investors.drop('InvestorID', axis=1)
    investors_investors = investors_investors.rename(columns={'Weight_x': 'EdgeWeight', 'Weight_y': 'SourceWeight'})

    investors_investors = pd.merge(investors_investors, investors, how='inner', left_on='TargetInvestorID', right_on='InvestorID')
    investors_investors = investors_investors.drop('InvestorID', axis=1)
    investors_investors = investors_investors.rename(columns={'Weight': 'TargetWeight'})

    ############################################################

    bc.log('Adding investor and concept weight to investor-concept edges...')

    investors_concepts = pd.merge(investors_concepts, investors, how='inner', on='InvestorID')
    investors_concepts = investors_concepts.rename(columns={'Weight_x': 'EdgeWeight', 'Weight_y': 'InvestorWeight'})

    investors_concepts = pd.merge(investors_concepts, concepts, how='inner', on='PageID')
    investors_concepts = investors_concepts.rename(columns={'Weight': 'ConceptWeight'})

    ############################################################

    bc.log('Adding source and target weight to concept-concept edges...')

    concepts_concepts = pd.merge(concepts_concepts, concepts, how='inner', left_on='SourcePageID', right_on='PageID')
    concepts_concepts = concepts_concepts.drop('PageID', axis=1)
    concepts_concepts = concepts_concepts.rename(columns={'Weight_x': 'EdgeWeight', 'Weight_y': 'SourceWeight'})

    concepts_concepts = pd.merge(concepts_concepts, concepts, how='inner', left_on='TargetPageID', right_on='PageID')
    concepts_concepts = concepts_concepts.drop('PageID', axis=1)
    concepts_concepts = concepts_concepts.rename(columns={'Weight': 'TargetWeight'})

    # TODO Rewrite the code below to correctly compute the weights. Need to do:
    #   - Compute base term T(i) and T(c) for both investor and concept pivots.
    #       - investors_investors -> groupby SourceInvestorID -> T^I(i) = sum_{p} w(p) w(i, p)
    #       - investors_concepts -> groupby InvestorID -> T^C(i) = sum_{p} w(p) w(i, p)
    #       - investors_concepts -> groupby PageID -> T^I(c) = sum_{p} w(p) w(p, c)
    #       - concepts_concepts -> groupby TargetPageID -> T^C(c) = sum_{p} w(p) w(p, c)
    #   - Compute intersection term T(i, c) for both investor and concept pivots.
    #       - investors_investors x investors_concepts -> groupby [S, T] ->
    #                                                   -> T^I(i, c) = sum_{p} w(p) (w(i, p) + w(p, c)) / 2
    #       - investors_concepts x concepts_concepts -> groupby [S, T] ->
    #                                                   -> T^C(i, c) = sum_{p} w(p) (w(i, p) + w(p, c)) / 2
    #   - Compute Jaccard coefficient as T(i, c) / (T(i) + T(c) - T(i, c))

    ############################################################
    # PREPARATION OF POTENTIAL EDGES - INV INV CPT             #
    ############################################################

    # Prepare dataframe with all triples investor-investor-concept (i, j, c) such that i -> j -> c, in a uniform format,
    # whose columns are:
    #
    # SourceID, PivotID, TargetID, WeightSource, WeightPivot, WeightTarget, WeightSourcePivot, WeightPivotTarget
    # ---------Node IDs----------  --------------Node weights-------------  ------------Edge weights------------

    bc.log('Preparing potential edges with investor as pivot...')
    bc.indent()

    ############################################################

    bc.log('Merging investor-investor edges with investor-concept edges...')

    inv_inv_cpt = pd.merge(investors_investors, investors_concepts.rename(columns={'InvestorID': 'TargetInvestorID'}), how='inner', on='TargetInvestorID')
    inv_inv_cpt = inv_inv_cpt.rename(columns={'SourceInvestorID': 'SourceID', 'TargetInvestorID': 'PivotID', 'PageID': 'TargetID', 'Weight_x': 'WeightSourcePivot', 'Weight_y': 'WeightPivotTarget'})

    ############################################################

    bc.log('Adding pivot investor weight...')

    inv_inv_cpt = pd.merge(inv_inv_cpt, investors.rename(columns={'InvestorID': 'PivotID'}), how='left', on='PivotID')
    inv_inv_cpt = inv_inv_cpt.rename(columns={'Weight': 'WeightPivot'})

    ############################################################

    bc.log('Creating column with row type...')

    inv_inv_cpt['L'] = inv_inv_cpt['SourceID'].notna()
    inv_inv_cpt['R'] = inv_inv_cpt['TargetID'].notna()
    inv_inv_cpt = inv_inv_cpt[['SourceID', 'PivotID', 'TargetID', 'WeightSourcePivot', 'WeightPivotTarget', 'WeightPivot', 'L', 'R']]

    bc.outdent()

    ############################################################
    # PREPARATION OF POTENTIAL EDGES - INV CPT CPT             #
    ############################################################

    # Prepare dataframe with all triples investor-concept-concept (i, d, c) such that i -> d -> c, in a uniform format,
    # whose columns are:
    #
    # SourceID, PivotID, TargetID, WeightSource, WeightPivot, WeightTarget, WeightSourcePivot, WeightPivotTarget
    # ---------Node IDs----------  --------------Node weights-------------  ------------Edge weights------------

    bc.log('Preparing potential edges with concept as pivot...')
    bc.indent()

    ############################################################

    bc.log('Merging investor-concept edges with concept-concept edges...')

    inv_cpt_cpt = pd.merge(investors_concepts.rename(columns={'PageID': 'SourcePageID'}), concepts_concepts, how='outer', on='SourcePageID')
    inv_cpt_cpt = inv_cpt_cpt.rename(columns={'InvestorID': 'SourceID', 'SourcePageID': 'PivotID', 'TargetPageID': 'TargetID', 'Weight_x': 'WeightSourcePivot', 'Weight_y': 'WeightPivotTarget'})

    ############################################################

    bc.log('Adding pivot concept weight...')

    inv_cpt_cpt = pd.merge(inv_cpt_cpt, concepts.rename(columns={'PageID': 'PivotID'}), how='left', on='PivotID')
    inv_cpt_cpt = inv_cpt_cpt.rename(columns={'Weight': 'WeightPivot'})

    ############################################################

    bc.log('Creating column with row type...')

    inv_cpt_cpt['L'] = inv_cpt_cpt['SourceID'].notna()
    inv_cpt_cpt['R'] = inv_cpt_cpt['TargetID'].notna()
    inv_cpt_cpt = inv_cpt_cpt[['SourceID', 'PivotID', 'TargetID', 'WeightSourcePivot', 'WeightPivotTarget', 'WeightPivot', 'L', 'R']]

    bc.outdent()

    ############################################################
    # PREPARATION OF POTENTIAL EDGES - CONCAT AND GROUP        #
    ############################################################

    bc.log('Combining both types of potential edges...')

    inv_pvt_cpt = pd.concat([inv_inv_cpt, inv_cpt_cpt]).reset_index(drop=True)

    ############################################################

    bc.log('Grouping by [SourceID, TargetID, L, R]...')

    def aggregate_stlr(group):
        left = group['L'].all()
        right = group['R'].all()

        if len(group) > 1 and not (left and right):
            print('pato')
            print(group)
            print('cuac')
            print(left)
            print('buru')
            print(right)

        if left and right:
            x = (group['WeightPivot'] * (group['WeightSourcePivot'] + group['WeightPivotTarget'])).sum() / 2
        elif left:
            x = group['WeightPivot'] * group['WeightSourcePivot']
        elif right:
            x = group['WeightPivot'] * group['WeightPivotTarget']
        else:
            # Should never get here
            x = 0

        return pd.Series({'Weight': x})

    df = inv_pvt_cpt.groupby(by=['SourceID', 'TargetID', 'L', 'R']).apply(aggregate_stlr).reset_index()

    ############################################################

    bc.log('Grouping by [SourceID, TargetID]...')

    def aggregate_st(group):
        if len(group) > 1:
            print('pato')
            print(group)
            aaa = group[group['L'] & group['R']]
            print('cuac')
            print(aaa)

        return 0

    df = df.groupby(by=['SourceID', 'TargetID']).apply(aggregate_st).reset_index()

    ############################################################

    # TODO Compute the modified Jaccard coefficient by dividing the contribution of (T, T) over the sum of the three.

    bc.report()


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    main()
