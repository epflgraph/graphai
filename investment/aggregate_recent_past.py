import numpy as np
import pandas as pd

from interfaces.db import DB

from utils.time.date import *
from utils.breadcrumb import Breadcrumb


def compute_year_coefficients():
    # Minimum year to consider for the computation of the Jaccard indices.
    # Older funding rounds will be ignored.
    max_date = str(now().date())
    max_year = int(max_date.split('-')[0])
    min_year = max_year - 4
    min_date = f'{min_year}-01-01'

    # Create DataFrame with start and end dates per year
    years = pd.DataFrame({'Year': range(min_year, max_year + 1)})

    years['StartDate'] = pd.to_datetime(years['Year'].astype(str) + '-01-01')
    years['EndDate'] = pd.to_datetime((years['Year'] + 1).astype(str) + '-01-01')

    # Compute ratio of time differences
    # CoefPresent = number of days in given year / total number of days in time window
    # CoefPast = number of days in time window before given year / total number of days in time window
    years['CoefPresent'] = (years['EndDate'] - years['StartDate']) / (pd.to_datetime(max_date) - pd.to_datetime(min_date))
    years['CoefPast'] = (years['StartDate'] - pd.to_datetime(min_date)) / (pd.to_datetime(max_date) - pd.to_datetime(min_date))

    # Keep only relevant columns
    years = years[['Year', 'CoefPresent', 'CoefPast']]

    return years


def rescale_scores(df, years):
    # Filter edges older than min_year
    df = df[df['Year'] >= years['Year'].min()].reset_index(drop=True)

    # Add time coefficients
    df = pd.merge(df, years, how='left', on='Year')

    # Compute rescaled scores according to time coefficients
    slc_term_0 = df['CoefPast'] * df['CountAmount']
    slc_term_1 = df['CoefPresent'] * df['ScoreLinCount']

    sla_term_0 = df['CoefPast'] * df['SumAmount']
    sla_term_1 = df['CoefPresent'] * df['ScoreLinAmount']

    sqc_term_0 = (df['CoefPast'] ** 2) * df['CountAmount']
    sqc_term_1 = 2 * df['CoefPast'] * df['CoefPresent'] * df['ScoreLinCount']
    sqc_term_2 = (df['CoefPresent'] ** 2) * df['ScoreQuadCount']

    sqa_term_0 = (df['CoefPast'] ** 2) * df['SumAmount']
    sqa_term_1 = 2 * df['CoefPast'] * df['CoefPresent'] * df['ScoreLinAmount']
    sqa_term_2 = (df['CoefPresent'] ** 2) * df['ScoreQuadAmount']

    df['ScoreLinCount'] = slc_term_0 + slc_term_1
    df['ScoreLinAmount'] = sla_term_0 + sla_term_1
    df['ScoreQuadCount'] = sqc_term_0 + sqc_term_1 + sqc_term_2
    df['ScoreQuadAmount'] = sqa_term_0 + sqa_term_1 + sqa_term_2
    df = df.drop(['CoefPast', 'CoefPresent'], axis=1)

    return df


def aggregate_years(df, groupby_columns):
    return df.groupby(by=groupby_columns).aggregate({
        'CountAmount': 'sum',
        'MinAmount': 'min',
        'MaxAmount': 'max',
        'SumAmount': 'sum',
        'ScoreLinCount': 'sum',
        'ScoreLinAmount': 'sum',
        'ScoreQuadCount': 'sum',
        'ScoreQuadAmount': 'sum'
    }).reset_index()


def normalize_scores(df, score_column, epsilon=0.01):
    # Score normalization function is tanh(k*x), where k is inferred from data in such a way that,
    # if m is the median of x, 2m is mapped to 1 - epsilon, for a given epsilon (e.g. 0.01).
    #
    # This dynamic choice provides a sensible mapping from the given values to [0, 1].
    # Not doing this would more likely map values unevenly onto [0, 1]
    # (e.g. with most values >0.95), which is not a desired behaviour.
    #
    # As an example, if the median of x is 120, then we have the following mapping:
    #   - 0   |---> 0
    #   - 20  |---> 0.217
    #   - 60  |---> 0.579
    #   - 120 |---> 0.868
    #   - 180 |---> 0.963
    #   - 240 |---> 0.990
    #   - 300 |---> 0.997

    m = df[score_column].median()
    k = np.log(2 / epsilon - 1) / (4 * m)
    df['Score'] = np.tanh(k * df[score_column])
    df = df.drop(score_column, axis=1)

    # Filter out edges with 0 score
    df = df[df['Score'] > 0].reset_index(drop=True)

    return df


def main():

    ############################################################
    # INITIALIZATION                                           #
    ############################################################

    # Initialize breadcrumb to log and keep track of time
    bc = Breadcrumb()

    # Instantiate db interface to communicate with database
    db = DB()

    # Prepare year coefficients to combine metrics from different years
    years = compute_year_coefficients()

    ############################################################

    bc.log('Fetching investor nodes from database...')

    # Fetch table from database
    table_name = 'ca_temp.Nodes_N_Investor_T_Years'
    fields = ['InvestorID', 'Year', 'CountAmount', 'MinAmount', 'MaxAmount', 'SumAmount', 'ScoreLinCount', 'ScoreLinAmount', 'ScoreQuadCount', 'ScoreQuadAmount']
    df = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)

    # Rescale scores to fit time window, filter older years and aggregate accordingly
    df = rescale_scores(df, years)
    df = aggregate_years(df, groupby_columns=['InvestorID'])

    # Normalize scores to have them in [0, 1]
    df = df[['InvestorID', 'ScoreQuadCount']]
    investors = normalize_scores(df, score_column='ScoreQuadCount')

    del df

    ############################################################

    bc.log('Fetching concept nodes from database...')

    # Fetch table from database
    table_name = 'ca_temp.Nodes_N_Concept_T_Years'
    fields = ['PageID', 'Year', 'CountAmount', 'MinAmount', 'MaxAmount', 'SumAmount', 'ScoreLinCount', 'ScoreLinAmount', 'ScoreQuadCount', 'ScoreQuadAmount']
    df = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)

    # Rescale scores to fit time window, filter older years and aggregate accordingly
    df = rescale_scores(df, years)
    df = aggregate_years(df, groupby_columns=['PageID'])

    # Normalize scores to have them in [0, 1]
    df = df[['PageID', 'ScoreQuadCount']]
    concepts = normalize_scores(df, score_column='ScoreQuadCount')

    del df

    ############################################################

    bc.log('Fetching investor-investor edges from database...')

    # Fetch table from database
    table_name = 'ca_temp.Edges_N_Investor_N_Investor_T_Years'
    fields = ['SourceInvestorID', 'TargetInvestorID', 'Year', 'CountAmount', 'MinAmount', 'MaxAmount', 'SumAmount', 'ScoreLinCount', 'ScoreLinAmount', 'ScoreQuadCount', 'ScoreQuadAmount']
    df = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)

    # Rescale scores to fit time window, filter older years and aggregate accordingly
    df = rescale_scores(df, years)
    df = aggregate_years(df, groupby_columns=['SourceInvestorID', 'TargetInvestorID'])

    # Normalize scores to have them in [0, 1]
    df = df[['SourceInvestorID', 'TargetInvestorID', 'ScoreQuadCount']]
    df = normalize_scores(df, score_column='ScoreQuadCount')

    # Duplicate as to have all edges and not only in one direction
    reversed_df = df.copy()
    reversed_df[['SourceInvestorID', 'TargetInvestorID']] = df[['TargetInvestorID', 'SourceInvestorID']]
    investors_investors = pd.concat([df, reversed_df]).reset_index(drop=True)

    del df, reversed_df

    ############################################################

    bc.log('Fetching investor-concept edges from database...')

    # Fetch table from database
    table_name = 'ca_temp.Edges_N_Investor_N_Concept_T_Years'
    fields = ['InvestorID', 'PageID', 'Year', 'CountAmount', 'MinAmount', 'MaxAmount', 'SumAmount', 'ScoreLinCount', 'ScoreLinAmount', 'ScoreQuadCount', 'ScoreQuadAmount', 'Concentration']
    df = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)

    # Extract concentrations and aggregate them per year
    concentrations = df[['InvestorID', 'PageID', 'Year', 'Concentration']]
    concentrations = concentrations.groupby(by=['InvestorID', 'PageID']).aggregate({'Concentration': 'mean'})

    # Rescale scores to fit time window, filter older years and aggregate accordingly
    df = rescale_scores(df, years)
    df = aggregate_years(df, groupby_columns=['InvestorID', 'PageID'])

    # Add concentrations and use them to dilute big investors who invest in all concepts
    df = pd.merge(df, concentrations, how='left', on=['InvestorID', 'PageID'])
    df['DilutedScore'] = df['ScoreQuadCount'] * df['Concentration']

    # Normalize scores to have them in [0, 1]
    df = df[['InvestorID', 'PageID', 'DilutedScore']]
    investors_concepts = normalize_scores(df, score_column='DilutedScore')

    del df

    ############################################################

    bc.log('Fetching concept-concept edges from database...')

    # Fetch table from database
    table_name = 'graph.Edges_N_Concept_N_Concept_T_GraphScore'
    fields = ['SourcePageID', 'TargetPageID', 'NormalisedScore']
    concept_ids = list(investors_concepts['PageID'].drop_duplicates().astype(int))
    conditions = {'SourcePageID': concept_ids, 'TargetPageID': concept_ids}
    df = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

    # Normalize scores to have them in [0, 1]
    df = df[['SourcePageID', 'TargetPageID', 'NormalisedScore']]
    concepts_concepts = normalize_scores(df, score_column='NormalisedScore')

    del df

    ############################################################
    # INSERT AGGREGATED DATA INTO DATABASE                     #
    ############################################################

    bc.log('Inserting investors into database...')

    table_name = 'ca_temp.Nodes_N_Investor_T_Aggregated'
    definition = [
        'InvestorID CHAR(64)',
        'Score FLOAT',
        'KEY InvestorID (InvestorID)'
    ]
    db.drop_create_insert_table(table_name, definition, investors)

    ############################################################

    bc.log('Inserting concepts into database...')

    table_name = 'ca_temp.Nodes_N_Concept_T_Aggregated'
    definition = [
        'PageID INT UNSIGNED',
        'Score FLOAT',
        'KEY PageID (PageID)'
    ]
    db.drop_create_insert_table(table_name, definition, concepts)

    ############################################################

    bc.log('Inserting investor-investor edges into database...')

    table_name = 'ca_temp.Edges_N_Investor_N_Investor_T_Aggregated'
    definition = [
        'SourceInvestorID CHAR(64)',
        'TargetInvestorID CHAR(64)',
        'Score FLOAT',
        'KEY SourceInvestorID (SourceInvestorID)',
        'KEY TargetInvestorID (TargetInvestorID)'
    ]
    db.drop_create_insert_table(table_name, definition, investors_investors)

    ############################################################

    bc.log('Inserting investor-concept edges into database...')

    table_name = 'ca_temp.Edges_N_Investor_N_Concept_T_Aggregated'
    definition = [
        'InvestorID CHAR(64)',
        'PageID INT UNSIGNED',
        'Score FLOAT',
        'KEY InvestorID (InvestorID)',
        'KEY PageID (PageID)'
    ]
    db.drop_create_insert_table(table_name, definition, investors_concepts)

    ############################################################

    bc.log('Inserting investor-concept edges into database...')

    table_name = 'ca_temp.Edges_N_Concept_N_Concept_T_Aggregated'
    definition = [
        'SourcePageID INT UNSIGNED',
        'TargetPageID INT UNSIGNED',
        'Score FLOAT',
        'KEY SourcePageID (SourcePageID)',
        'KEY TargetPageID (TargetPageID)'
    ]
    db.drop_create_insert_table(table_name, definition, concepts_concepts)

    ############################################################

    bc.report()


if __name__ == '__main__':
    pd.set_option('display.max_rows', 400)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    main()
