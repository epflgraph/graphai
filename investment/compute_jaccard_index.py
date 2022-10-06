import numpy as np
import pandas as pd

from interfaces.db import DB

from utils.time.date import *
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


def compute_year_coefficients(min_date, max_date, min_year, max_year):
    # Create DataFrame with start and end dates per year
    years = pd.DataFrame({'Year': range(min_year, max_year + 1)})
    years['StartDate'] = pd.to_datetime(years['Year'].apply(lambda y: f'{y}-01-01'))
    years['EndDate'] = pd.to_datetime(years['Year'].apply(lambda y: f'{y + 1}-01-01'))

    # Compute ratio of time differences
    # CoefPresent = number of days in given year / total number of days in time window
    # CoefPast = number of days in time window before given year / total number of days in time window
    years['CoefPresent'] = (years['EndDate'] - years['StartDate']) / (pd.to_datetime(max_date) - pd.to_datetime(min_date))
    years['CoefPast'] = (years['StartDate'] - pd.to_datetime(min_date)) / (pd.to_datetime(max_date) - pd.to_datetime(min_date))

    # Keep only relevant columns
    years = years[['Year', 'CoefPresent', 'CoefPast']]

    return years


def filter_and_combine_years(df, years, groupby_columns):
    # Define function to aggregate metrics over different years
    def aggregate_years(group):
        group = pd.merge(group, years, how='left', on='Year')

        lin_term0 = group['CoefPast'] * group['CountAmount']
        lin_term1 = group['CoefPresent'] * group['ScoreLinCount']

        score_lin_count = (lin_term0 + lin_term1).sum()

        quad_term0 = (group['CoefPast'] ** 2) * group['CountAmount']
        quad_term1 = 2 * group['CoefPast'] * group['CoefPresent'] * group['ScoreLinCount']
        quad_term2 = (group['CoefPresent'] ** 2) * group['ScoreQuadCount']

        score_quad_count = (quad_term0 + quad_term1 + quad_term2).sum()

        return pd.Series({'ScoreLinCount': score_lin_count, 'ScoreQuadCount': score_quad_count})

    # Filter edges older than min_year
    df = df[df['Year'] >= years['Year'].min()].reset_index(drop=True)

    # Aggregate yearly metrics correctly
    df = df.groupby(by=groupby_columns).apply(aggregate_years).reset_index()

    return df


def normalize_scores(df, score_column):
    # Normalize weight
    df['Weight'] = weight(df[score_column])
    df = df.drop(score_column, axis=1)

    # Filter out edges with 0 weight
    df = df[df['Weight'] > 0].reset_index(drop=True)

    return df


def main():

    ############################################################
    # INITIALIZATION                                           #
    ############################################################

    # Initialize breadcrumb to log and keep track of time
    bc = Breadcrumb()

    # Instantiate db interface to communicate with database
    db = DB()

    # Minimum year to consider for the computation of the Jaccard indices.
    # Older funding rounds will be ignored.
    max_date = str(now().date())
    max_year = int(max_date.split('-')[0])
    min_year = max_year - 4
    min_date = f'{min_year}-01-01'

    # Prepare year coefficients to combine metrics from different years
    years = compute_year_coefficients(min_date, max_date, min_year, max_year)

    ############################################################
    # FETCH GRAPH FROM DATABASE                                #
    ############################################################

    bc.log(f'Fetching tables from database...')
    bc.indent()

    ############################################################

    bc.log('Fetching investor-investor edges from database...')

    # Fetch table from database
    table_name = 'ca_temp.Edges_N_Investor_N_Investor'
    fields = ['SourceInvestorID', 'TargetInvestorID', 'Year', 'CountAmount', 'ScoreLinCount', 'ScoreQuadCount']
    df = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)

    # Filter years older than time window and aggregate metrics accordingly
    df = filter_and_combine_years(df, years, groupby_columns=['SourceInvestorID', 'TargetInvestorID'])

    # Normalize scores to have them in [0, 1]
    df = normalize_scores(df[['SourceInvestorID', 'TargetInvestorID', 'ScoreQuadCount']], score_column='ScoreQuadCount')

    # Duplicate as to have all edges and not only in one direction
    reversed_df = df.copy()
    reversed_df[['SourceInvestorID', 'TargetInvestorID']] = df[['TargetInvestorID', 'SourceInvestorID']]
    investors_investors = pd.concat([df, reversed_df]).reset_index(drop=True)

    del df, reversed_df

    ############################################################

    bc.log('Fetching investor-concept edges from database...')

    # Fetch table from database
    table_name = 'ca_temp.Edges_N_Investor_N_Concept'
    fields = ['InvestorID', 'PageID', 'Year', 'CountAmount', 'ScoreLinCount', 'ScoreQuadCount']
    df = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)
    df['PageID'] = df['PageID'].astype(str)

    # Filter years older than time window and aggregate metrics accordingly
    df = filter_and_combine_years(df, years, groupby_columns=['InvestorID', 'PageID'])

    # Normalize scores to have them in [0, 1]
    investors_concepts = normalize_scores(df[['InvestorID', 'PageID', 'ScoreQuadCount']], score_column='ScoreQuadCount')

    del df

    ############################################################

    bc.log('Fetching concept-concept edges from database...')

    # Fetch table from database
    table_name = 'graph.Edges_N_Concept_N_Concept_T_GraphScore'
    fields = ['SourcePageID', 'TargetPageID', 'NormalisedScore']
    concept_ids = list(investors_concepts['PageID'].drop_duplicates().astype(int))
    conditions = {'SourcePageID': concept_ids, 'TargetPageID': concept_ids}
    df = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)
    df['SourcePageID'] = df['SourcePageID'].astype(str)
    df['TargetPageID'] = df['TargetPageID'].astype(str)

    # Normalize scores to have them in [0, 1]
    concepts_concepts = normalize_scores(df[['SourcePageID', 'TargetPageID', 'NormalisedScore']], score_column='NormalisedScore')

    del df

    ############################################################

    bc.log('Fetching investor nodes from database...')

    # Fetch table from database
    table_name = 'ca_temp.Nodes_N_Investor'
    fields = ['InvestorID', 'Year', 'CountAmount', 'ScoreLinCount', 'ScoreQuadCount']
    df = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)

    # Filter years older than time window and aggregate metrics accordingly
    df = filter_and_combine_years(df, years, groupby_columns=['InvestorID'])

    # Normalize scores to have them in [0, 1]
    investors = normalize_scores(df[['InvestorID', 'ScoreQuadCount']], score_column='ScoreQuadCount')

    del df

    ############################################################

    bc.log('Fetching concept nodes from database...')

    # Fetch table from database
    table_name = 'ca_temp.Nodes_N_Concept'
    fields = ['PageID', 'Year', 'CountAmount', 'ScoreLinCount', 'ScoreQuadCount']
    df = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)
    df['PageID'] = df['PageID'].astype(str)

    # Filter years older than time window and aggregate metrics accordingly
    df = filter_and_combine_years(df, years, groupby_columns=['PageID'])

    # Normalize scores to have them in [0, 1]
    concepts = normalize_scores(df[['PageID', 'ScoreQuadCount']], score_column='ScoreQuadCount')

    del df

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

    bc.log(f'Preparing potential edges...')
    bc.indent()

    ############################################################

    bc.log('Merging investor-investor edges with investor-concept edges...')

    inv_inv_cpt = pd.merge(
        investors_investors.rename(columns={'SourceInvestorID': 'SourceID', 'TargetInvestorID': 'PivotID', 'Weight': 'SourcePivotWeight'}),
        investors_concepts.rename(columns={'InvestorID': 'PivotID', 'PageID': 'TargetID', 'Weight': 'PivotTargetWeight'}),
        how='inner',
        on='PivotID'
    )
    inv_inv_cpt = pd.merge(inv_inv_cpt, investors.rename(columns={'InvestorID': 'PivotID', 'Weight': 'PivotWeight'}), how='left', on='PivotID')

    ############################################################

    bc.log('Merging investor-concept edges with concept-concept edges...')

    inv_cpt_cpt = pd.merge(
        investors_concepts.rename(columns={'InvestorID': 'SourceID', 'PageID': 'PivotID', 'Weight': 'SourcePivotWeight'}),
        concepts_concepts.rename(columns={'SourcePageID': 'PivotID', 'TargetPageID': 'TargetID', 'Weight': 'PivotTargetWeight'}),
        how='inner',
        on='PivotID'
    )
    inv_cpt_cpt = pd.merge(inv_cpt_cpt, concepts.rename(columns={'PageID': 'PivotID', 'Weight': 'PivotWeight'}), how='left', on='PivotID')

    ############################################################

    bc.log('Preparing dataframe with potential edges...')

    potential_edges = pd.concat([inv_inv_cpt[['SourceID', 'TargetID']], inv_cpt_cpt[['SourceID', 'TargetID']]]).drop_duplicates()
    potential_edges = pd.merge(potential_edges, investors.rename(columns={'InvestorID': 'SourceID', 'Weight': 'SourceWeight'}), how='left', on='SourceID')
    potential_edges = pd.merge(potential_edges, concepts.rename(columns={'PageID': 'TargetID', 'Weight': 'TargetWeight'}), how='left', on='TargetID')
    potential_edges = pd.merge(potential_edges, investors_concepts.rename(columns={'InvestorID': 'SourceID', 'PageID': 'TargetID', 'Weight': 'SourceTargetWeight'}), how='left', on=['SourceID', 'TargetID'])
    potential_edges = potential_edges.fillna(0)

    ############################################################

    bc.outdent()

    ############################################################
    # POTENTIAL EDGES - INTERSECTION TERMS                     #
    ############################################################

    bc.log(f'Computing intersection terms...')
    bc.indent()

    ############################################################

    bc.log('Computing intersection terms W^i(int_I) and W^c(int_I)...')

    # For given investor i and concept c, their intersection terms W^i(int_I) and W^c(int_I) are defined as
    #   W^i(int_I) = sum_{j in int_I} w(j) * w(i, j)
    #   W^c(int_I) = sum_{j in int_I} w(j) * w(j, c)
    # where int_I = I(i) \cap I^{-1}(c)

    inv_inv_cpt['W^i(int_I)'] = inv_inv_cpt['PivotWeight'] * inv_inv_cpt['SourcePivotWeight']
    inv_inv_cpt['W^c(int_I)'] = inv_inv_cpt['PivotWeight'] * inv_inv_cpt['PivotTargetWeight']
    df = inv_inv_cpt[['SourceID', 'TargetID', 'W^i(int_I)', 'W^c(int_I)']].groupby(by=['SourceID', 'TargetID']).sum().reset_index()
    potential_edges = pd.merge(potential_edges, df, how='left', on=['SourceID', 'TargetID'])
    potential_edges = potential_edges.fillna(0)

    del df

    ############################################################

    bc.log('Computing intersection terms W^i(int_C) and W^c(int_C)...')

    # For given investor i and concept c, their intersection terms W^i(int_C) and W^c(int_C) are defined as
    #   W^i(int_C) = sum_{d in int_C} w(d) * w(i, d)
    #   W^c(int_C) = sum_{d in int_C} w(d) * w(d, c)
    # where int_C = C(i) \cap C^{-1}(c)

    inv_cpt_cpt['W^i(int_C)'] = inv_cpt_cpt['PivotWeight'] * inv_cpt_cpt['SourcePivotWeight']
    inv_cpt_cpt['W^c(int_C)'] = inv_cpt_cpt['PivotWeight'] * inv_cpt_cpt['PivotTargetWeight']
    df = inv_cpt_cpt[['SourceID', 'TargetID', 'W^i(int_C)', 'W^c(int_C)']].groupby(by=['SourceID', 'TargetID']).sum().reset_index()
    potential_edges = pd.merge(potential_edges, df, how='left', on=['SourceID', 'TargetID'])
    potential_edges = potential_edges.fillna(0)

    del df

    ############################################################

    bc.outdent()

    ############################################################
    # POTENTIAL EDGES - LATERAL TERMS                          #
    ############################################################

    bc.log(f'Computing lateral terms...')
    bc.indent()

    ############################################################

    bc.log('Adding target node weight to investor-investor edges...')

    investors_investors = pd.merge(
        investors_investors.rename(columns={'Weight': 'SourceTargetWeight'}),
        investors.rename(columns={'InvestorID': 'TargetInvestorID', 'Weight': 'TargetWeight'}),
        how='left',
        on='TargetInvestorID'
    )

    ############################################################

    bc.log('Adding investor and concept node weights to investor-concept edges...')

    investors_concepts = investors_concepts.rename(columns={'Weight': 'SourceTargetWeight'})

    investors_concepts = pd.merge(
        investors_concepts,
        investors.rename(columns={'Weight': 'SourceWeight'}),
        how='left',
        on='InvestorID'
    )

    investors_concepts = pd.merge(
        investors_concepts,
        concepts.rename(columns={'Weight': 'TargetWeight'}),
        how='left',
        on='PageID'
    )

    ############################################################

    bc.log('Adding source node weight to concept-concept edges...')

    concepts_concepts = pd.merge(
        concepts_concepts.rename(columns={'Weight': 'SourceTargetWeight'}),
        concepts.rename(columns={'PageID': 'SourcePageID', 'Weight': 'SourceWeight'}),
        how='left',
        on='SourcePageID'
    )

    ############################################################

    bc.log('Computing lateral term W^i(left_I)...')

    # For given investor i and concept c, their lateral term W^i(left_I) is defined as
    #   W^i(left_I) = sum_{j in left_I} w(j) * w(i, j)
    # where left_I = I(i) \setminus I^{-1}(c)
    #
    # To compute it, we first compute the term W^i(I(i)), defined as
    #   W^i(I(i)) = sum_{j in I(i)} w(j) * w(i, j)
    # and then substract the intersection term W^i(int_I) computed above:
    #   W^i(left_I) = W^i(I(i)) - W^i(int_I)

    investors_investors['W^i(I(i))'] = investors_investors['TargetWeight'] * investors_investors['SourceTargetWeight']
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
    #   W^c(right_I) = sum_{j in right_I} w(j) * w(j, c)
    # where right_I = I^{-1}(c) \setminus I(i)
    #
    # To compute it, we first compute the term W^c(I^{-1}(c)), defined as
    #   W^c(I^{-1}(c)) = sum_{j in I^{-1}(c)} w(j) * w(j, c)
    # and then substract the intersection term W^c(int_I) computed above:
    #   W^c(right_I) = W^c(I^{-1}(c)) - W^c(int_I)

    investors_concepts['W^c(I^{-1}(c))'] = investors_concepts['SourceWeight'] * investors_concepts['SourceTargetWeight']
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
    #   W^i(left_C) = sum_{d in left_C} w(d) * w(i, d)
    # where left_C = C(i) \setminus C^{-1}(c)
    #
    # To compute it, we first compute the term W^i(C(i)), defined as
    #   W^i(C(i)) = sum_{d in C(i)} w(d) * w(i, d)
    # and then substract the intersection term W^i(int_C) computed above:
    #   W^i(left_C) = W^i(C(i)) - W^i(int_C)

    investors_concepts['W^i(C(i))'] = investors_concepts['TargetWeight'] * investors_concepts['SourceTargetWeight']
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
    #   W^c(right_C) = sum_{d in right_C} w(d) * w(d, c)
    # where right_C = C^{-1}(c) \setminus C(i)
    #
    # To compute it, we first compute the term W^c(C^{-1}(c)), defined as
    #   W^c(C^{-1}(c)) = sum_{d in C^{-1}(c)} w(d) * w(d, c)
    # and then substract the intersection term W^c(int_C) computed above:
    #   W^c(right_C) = W^c(C^{-1}(c)) - W^c(int_C)

    concepts_concepts['W^c(C^{-1}(c))'] = concepts_concepts['SourceWeight'] * concepts_concepts['SourceTargetWeight']
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
    potential_edges['Jaccard_100'] = potential_edges['Jaccard_000'] * potential_edges['SourceWeight'] * potential_edges['TargetWeight']
    potential_edges['Jaccard_010'] = potential_edges['Jaccard_000'] * (potential_edges['SourceTargetWeight'] + 1) / 2
    potential_edges['Jaccard_110'] = potential_edges['Jaccard_100'] * (potential_edges['SourceTargetWeight'] + 1) / 2
    potential_edges = potential_edges.fillna(0)

    ############################################################
    # POTENTIAL EDGES - INSERT INTO DATABASE                   #
    ############################################################

    # Filter potential edges based on relevant Jaccard index.
    # Otherwise data is too heavy (e.g. >400GB for potential edges from just one year)
    epsilon = 0.01
    condition = (potential_edges['Jaccard_000'] >= epsilon) & (potential_edges['Jaccard_100'] >= epsilon) & (potential_edges['Jaccard_010'] >= epsilon) & (potential_edges['Jaccard_110'] >= epsilon)
    potential_edges = potential_edges[condition]

    bc.log('Inserting potential edges into database...')

    # Drop, recreate table and fill with df
    table_name = 'ca_temp.Edges_N_Investor_N_Concept_T_Jaccard'
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

    main()
