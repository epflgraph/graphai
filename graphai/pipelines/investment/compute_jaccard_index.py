import numpy as np
import pandas as pd

from graphai.core.interfaces.db import DB

from graphai.core.utils.breadcrumb import Breadcrumb


def compute_jaccard_index():

    ############################################################
    # INITIALIZATION                                           #
    ############################################################

    # Initialize breadcrumb to log and keep track of time
    bc = Breadcrumb()

    # Instantiate db interface to communicate with database
    db = DB()

    ############################################################
    # FETCH GRAPH FROM DATABASE                                #
    ############################################################

    bc.log('Fetching tables from database...')
    bc.indent()

    ############################################################

    bc.log('Fetching investor nodes from database...')

    table_name = 'aitor.Nodes_N_Investor_T_Aggregated'
    fields = ['InvestorID', 'Score']
    investors = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)

    ############################################################

    bc.log('Fetching concept nodes from database...')

    table_name = 'aitor.Nodes_N_Concept_T_Aggregated'
    fields = ['PageID', 'Score']
    concepts = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)
    concepts['PageID'] = concepts['PageID'].astype(str)

    ############################################################

    bc.log('Fetching investor-investor edges from database...')

    table_name = 'aitor.Edges_N_Investor_N_Investor_T_Aggregated'
    fields = ['SourceInvestorID', 'TargetInvestorID', 'Score']
    investors_investors = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)

    ############################################################

    bc.log('Fetching investor-concept edges from database...')

    table_name = 'aitor.Edges_N_Investor_N_Concept_T_Aggregated'
    fields = ['InvestorID', 'PageID', 'Score']
    investors_concepts = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)
    investors_concepts['PageID'] = investors_concepts['PageID'].astype(str)

    ############################################################

    bc.log('Fetching concept-concept edges from database...')

    table_name = 'aitor.Edges_N_Concept_N_Concept_T_Aggregated'
    fields = ['SourcePageID', 'TargetPageID', 'Score']
    concepts_concepts = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)
    concepts_concepts['SourcePageID'] = concepts_concepts['SourcePageID'].astype(str)
    concepts_concepts['TargetPageID'] = concepts_concepts['TargetPageID'].astype(str)

    ############################################################

    bc.outdent()

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

    bc.log('Preparing potential edges...')
    bc.indent()

    ############################################################

    bc.log('Merging investor-investor edges with investor-concept edges...')

    inv_inv_cpt = pd.merge(
        investors_investors.rename(columns={'SourceInvestorID': 'SourceID', 'TargetInvestorID': 'PivotID', 'Score': 'SourcePivotWeight'}),
        investors_concepts.rename(columns={'InvestorID': 'PivotID', 'PageID': 'TargetID', 'Score': 'PivotTargetWeight'}),
        how='inner',
        on='PivotID'
    )
    inv_inv_cpt = pd.merge(inv_inv_cpt, investors.rename(columns={'InvestorID': 'PivotID', 'Score': 'PivotWeight'}), how='left', on='PivotID')

    ############################################################

    bc.log('Merging investor-concept edges with concept-concept edges...')

    inv_cpt_cpt = pd.merge(
        investors_concepts.rename(columns={'InvestorID': 'SourceID', 'PageID': 'PivotID', 'Score': 'SourcePivotWeight'}),
        concepts_concepts.rename(columns={'SourcePageID': 'PivotID', 'TargetPageID': 'TargetID', 'Score': 'PivotTargetWeight'}),
        how='inner',
        on='PivotID'
    )
    inv_cpt_cpt = pd.merge(inv_cpt_cpt, concepts.rename(columns={'PageID': 'PivotID', 'Score': 'PivotWeight'}), how='left', on='PivotID')

    ############################################################

    bc.log('Preparing dataframe with potential edges...')

    potential_edges = pd.concat([inv_inv_cpt[['SourceID', 'TargetID']], inv_cpt_cpt[['SourceID', 'TargetID']]]).drop_duplicates()
    potential_edges = pd.merge(potential_edges, investors.rename(columns={'InvestorID': 'SourceID', 'Score': 'SourceWeight'}), how='left', on='SourceID')
    potential_edges = pd.merge(potential_edges, concepts.rename(columns={'PageID': 'TargetID', 'Score': 'TargetWeight'}), how='left', on='TargetID')
    potential_edges = pd.merge(potential_edges, investors_concepts.rename(columns={'InvestorID': 'SourceID', 'PageID': 'TargetID', 'Score': 'SourceTargetWeight'}), how='left', on=['SourceID', 'TargetID'])
    potential_edges = potential_edges.fillna(0)

    ############################################################

    bc.outdent()

    ############################################################
    # POTENTIAL EDGES - INTERSECTION TERMS                     #
    ############################################################

    bc.log('Computing intersection terms...')
    bc.indent()

    ############################################################

    bc.log('Computing intersection terms W^i(int_I) and W^c(int_I)...')

    # For given investor i and concept c, their intersection terms W^i(int_I) and W^c(int_I) are defined as
    #   W^i(int_I) = sum_{j in int_I} sqrt(w(j) * w(i, j))
    #   W^c(int_I) = sum_{j in int_I} sqrt(w(j) * w(j, c))
    # where int_I = I(i) \cap I^{-1}(c)

    inv_inv_cpt['W^i(int_I)'] = np.sqrt(inv_inv_cpt['PivotWeight'] * inv_inv_cpt['SourcePivotWeight'])
    inv_inv_cpt['W^c(int_I)'] = np.sqrt(inv_inv_cpt['PivotWeight'] * inv_inv_cpt['PivotTargetWeight'])
    df = inv_inv_cpt[['SourceID', 'TargetID', 'W^i(int_I)', 'W^c(int_I)']].groupby(by=['SourceID', 'TargetID']).sum().reset_index()
    potential_edges = pd.merge(potential_edges, df, how='left', on=['SourceID', 'TargetID'])
    potential_edges = potential_edges.fillna(0)

    del df

    ############################################################

    bc.log('Computing intersection terms W^i(int_C) and W^c(int_C)...')

    # For given investor i and concept c, their intersection terms W^i(int_C) and W^c(int_C) are defined as
    #   W^i(int_C) = sum_{d in int_C} sqrt(w(d) * w(i, d))
    #   W^c(int_C) = sum_{d in int_C} sqrt(w(d) * w(d, c))
    # where int_C = C(i) \cap C^{-1}(c)

    inv_cpt_cpt['W^i(int_C)'] = np.sqrt(inv_cpt_cpt['PivotWeight'] * inv_cpt_cpt['SourcePivotWeight'])
    inv_cpt_cpt['W^c(int_C)'] = np.sqrt(inv_cpt_cpt['PivotWeight'] * inv_cpt_cpt['PivotTargetWeight'])
    df = inv_cpt_cpt[['SourceID', 'TargetID', 'W^i(int_C)', 'W^c(int_C)']].groupby(by=['SourceID', 'TargetID']).sum().reset_index()
    potential_edges = pd.merge(potential_edges, df, how='left', on=['SourceID', 'TargetID'])
    potential_edges = potential_edges.fillna(0)

    del df

    ############################################################

    bc.outdent()

    ############################################################
    # POTENTIAL EDGES - LATERAL TERMS                          #
    ############################################################

    bc.log('Computing lateral terms...')
    bc.indent()

    ############################################################

    bc.log('Adding target node weight to investor-investor edges...')

    investors_investors = pd.merge(
        investors_investors.rename(columns={'Score': 'SourceTargetWeight'}),
        investors.rename(columns={'InvestorID': 'TargetInvestorID', 'Score': 'TargetWeight'}),
        how='left',
        on='TargetInvestorID'
    )

    ############################################################

    bc.log('Adding investor and concept node weights to investor-concept edges...')

    investors_concepts = investors_concepts.rename(columns={'Score': 'SourceTargetWeight'})

    investors_concepts = pd.merge(
        investors_concepts,
        investors.rename(columns={'Score': 'SourceWeight'}),
        how='left',
        on='InvestorID'
    )

    investors_concepts = pd.merge(
        investors_concepts,
        concepts.rename(columns={'Score': 'TargetWeight'}),
        how='left',
        on='PageID'
    )

    ############################################################

    bc.log('Adding source node weight to concept-concept edges...')

    concepts_concepts = pd.merge(
        concepts_concepts.rename(columns={'Score': 'SourceTargetWeight'}),
        concepts.rename(columns={'PageID': 'SourcePageID', 'Score': 'SourceWeight'}),
        how='left',
        on='SourcePageID'
    )

    ############################################################

    bc.log('Computing lateral term W^i(left_I)...')

    # For given investor i and concept c, their lateral term W^i(left_I) is defined as
    #   W^i(left_I) = sum_{j in left_I} sqrt(w(j) * w(i, j))
    # where left_I = I(i) \setminus I^{-1}(c)
    #
    # To compute it, we first compute the term W^i(I(i)), defined as
    #   W^i(I(i)) = sum_{j in I(i)} sqrt(w(j) * w(i, j))
    # and then substract the intersection term W^i(int_I) computed above:
    #   W^i(left_I) = W^i(I(i)) - W^i(int_I)

    investors_investors['W^i(I(i))'] = np.sqrt(investors_investors['TargetWeight'] * investors_investors['SourceTargetWeight'])
    df = investors_investors[['SourceInvestorID', 'W^i(I(i))']].groupby(by='SourceInvestorID').sum().reset_index()
    potential_edges = pd.merge(
        potential_edges,
        df.rename(columns={'SourceInvestorID': 'SourceID'}),
        how='left',
        on='SourceID'
    )

    potential_edges['W^i(left_I)'] = (potential_edges['W^i(I(i))'] - potential_edges['W^i(int_I)']).clip(lower=0)
    potential_edges = potential_edges.drop('W^i(I(i))', axis=1)
    potential_edges = potential_edges.fillna(0)

    del df

    ############################################################

    bc.log('Computing lateral term W^c(right_I)...')

    # For given investor i and concept c, their lateral term W^c(right_I) is defined as
    #   W^c(right_I) = sum_{j in right_I} sqrt(w(j) * w(j, c))
    # where right_I = I^{-1}(c) \setminus I(i)
    #
    # To compute it, we first compute the term W^c(I^{-1}(c)), defined as
    #   W^c(I^{-1}(c)) = sum_{j in I^{-1}(c)} sqrt(w(j) * w(j, c))
    # and then substract the intersection term W^c(int_I) computed above:
    #   W^c(right_I) = W^c(I^{-1}(c)) - W^c(int_I)

    investors_concepts['W^c(I^{-1}(c))'] = np.sqrt(investors_concepts['SourceWeight'] * investors_concepts['SourceTargetWeight'])
    df = investors_concepts[['PageID', 'W^c(I^{-1}(c))']].groupby(by='PageID').sum().reset_index()
    potential_edges = pd.merge(
        potential_edges,
        df.rename(columns={'PageID': 'TargetID'}),
        how='left',
        on='TargetID'
    )

    potential_edges['W^c(right_I)'] = (potential_edges['W^c(I^{-1}(c))'] - potential_edges['W^c(int_I)']).clip(lower=0)
    potential_edges = potential_edges.drop('W^c(I^{-1}(c))', axis=1)
    potential_edges = potential_edges.fillna(0)

    del df

    ############################################################

    bc.log('Computing lateral term W^i(left_C)...')

    # For given investor i and concept c, their lateral term W^i(left_C) is defined as
    #   W^i(left_C) = sum_{d in left_C} sqrt(w(d) * w(i, d))
    # where left_C = C(i) \setminus C^{-1}(c)
    #
    # To compute it, we first compute the term W^i(C(i)), defined as
    #   W^i(C(i)) = sum_{d in C(i)} sqrt(w(d) * w(i, d))
    # and then substract the intersection term W^i(int_C) computed above:
    #   W^i(left_C) = W^i(C(i)) - W^i(int_C)

    investors_concepts['W^i(C(i))'] = np.sqrt(investors_concepts['TargetWeight'] * investors_concepts['SourceTargetWeight'])
    df = investors_concepts[['InvestorID', 'W^i(C(i))']].groupby(by='InvestorID').sum().reset_index()
    potential_edges = pd.merge(
        potential_edges,
        df.rename(columns={'InvestorID': 'SourceID'}),
        how='left',
        on='SourceID'
    )

    potential_edges['W^i(left_C)'] = (potential_edges['W^i(C(i))'] - potential_edges['W^i(int_C)']).clip(lower=0)
    potential_edges = potential_edges.drop('W^i(C(i))', axis=1)
    potential_edges = potential_edges.fillna(0)

    del df

    ############################################################

    bc.log('Computing lateral term W^c(right_C)...')

    # For given investor i and concept c, their lateral term W^c(right_C) is defined as
    #   W^c(right_C) = sum_{d in right_C} sqrt(w(d) * w(d, c))
    # where right_C = C^{-1}(c) \setminus C(i)
    #
    # To compute it, we first compute the term W^c(C^{-1}(c)), defined as
    #   W^c(C^{-1}(c)) = sum_{d in C^{-1}(c)} sqrt(w(d) * w(d, c))
    # and then substract the intersection term W^c(int_C) computed above:
    #   W^c(right_C) = W^c(C^{-1}(c)) - W^c(int_C)

    concepts_concepts['W^c(C^{-1}(c))'] = np.sqrt(concepts_concepts['SourceWeight'] * concepts_concepts['SourceTargetWeight'])
    df = concepts_concepts[['TargetPageID', 'W^c(C^{-1}(c))']].groupby(by='TargetPageID').sum().reset_index()
    potential_edges = pd.merge(
        potential_edges,
        df.rename(columns={'TargetPageID': 'TargetID'}),
        how='left',
        on='TargetID'
    )

    potential_edges['W^c(right_C)'] = (potential_edges['W^c(C^{-1}(c))'] - potential_edges['W^c(int_C)']).clip(lower=0)
    potential_edges = potential_edges.drop('W^c(C^{-1}(c))', axis=1)
    potential_edges = potential_edges.fillna(0)

    del df

    ############################################################

    bc.outdent()

    ############################################################
    # POTENTIAL EDGES - WEIGHTED JACCARD INDICES               #
    ############################################################

    bc.log('Computing weighted Jaccard indices...')

    # For given investor i and concept c, their weighted Jaccard index WJ(i, c) is defined as
    #                                             (W^i(int_I) + W^c(int_I) + W^i(int_C) + W^c(int_C)) / 2
    #   WJ(i, c) = -------------------------------------------------------------------------------------------------------------------
    #              (W^i(int_I) + W^c(int_I) + W^i(int_C) + W^c(int_C)) / 2  +  W^i(left_I) + W^c(right_I) + W^i(left_C) + W^c(right_C)

    intersection_terms = (potential_edges['W^i(int_I)'] + potential_edges['W^c(int_I)'] + potential_edges['W^i(int_C)'] + potential_edges['W^c(int_C)']) / 2
    lateral_terms = potential_edges['W^i(left_I)'] + potential_edges['W^c(right_I)'] + potential_edges['W^i(left_C)'] + potential_edges['W^c(right_C)']

    potential_edges['Jaccard_000'] = intersection_terms / (intersection_terms + lateral_terms)
    potential_edges['Jaccard_100'] = potential_edges['Jaccard_000'] * np.sqrt(potential_edges['SourceWeight'] * potential_edges['TargetWeight'])
    potential_edges['Jaccard_010'] = potential_edges['Jaccard_000'] * (potential_edges['SourceTargetWeight'] + 1) / 2
    potential_edges['Jaccard_110'] = potential_edges['Jaccard_100'] * (potential_edges['SourceTargetWeight'] + 1) / 2
    potential_edges = potential_edges.fillna(0)

    ############################################################
    # POTENTIAL EDGES - INSERT INTO DATABASE                   #
    ############################################################

    # Filter potential edges based on relevant Jaccard index.
    # Otherwise data is too heavy (e.g. 16M potential edges)
    epsilon = 0.01
    potential_edges = potential_edges[potential_edges['Jaccard_110'] >= epsilon]

    bc.log('Inserting potential edges into database...')

    # Drop, recreate table and fill with df
    table_name = 'aitor.Edges_N_Investor_N_Concept_T_Jaccard'
    definition = [
        'InvestorID CHAR(64)',
        'PageID INT UNSIGNED',
        'SourceWeight FLOAT',
        'TargetWeight FLOAT',
        'SourceTargetWeight FLOAT',
        'W_i_int_I FLOAT',
        'W_c_int_I FLOAT',
        'W_i_int_C FLOAT',
        'W_c_int_C FLOAT',
        'W_i_left_I FLOAT',
        'W_c_right_I FLOAT',
        'W_i_left_C FLOAT',
        'W_c_right_C FLOAT',
        'Jaccard_000 FLOAT',
        'Jaccard_100 FLOAT',
        'Jaccard_010 FLOAT',
        'Jaccard_110 FLOAT',
        'KEY InvestorID (InvestorID)',
        'KEY PageID (PageID)'
    ]
    db.drop_create_insert_table(table_name, definition, potential_edges)

    ############################################################

    bc.report()


if __name__ == '__main__':
    pd.set_option('display.max_rows', 400)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    compute_jaccard_index()
