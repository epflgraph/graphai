import pandas as pd
import numpy as np

from interfaces.db import DB

from investment.data import *
from investment.create_investments_graph import derive_historical_data

from utils.breadcrumb import Breadcrumb


def main():
    # Initialize breadcrumb to log and keep track of time
    bc = Breadcrumb()

    # Instantiate db interface to communicate with database
    db = DB()

    # Define all history time window, investments outside it will be ignored.
    min_date = '2018-01-01'
    max_date = '2022-01-01'

    min_year = int(min_date.split('-')[0])
    max_year = int(max_date.split('-')[0])

    # Digital signal processing, Sensor, Machine learning
    lab_concepts = pd.DataFrame({'PageID': [8525, 235757, 233488], 'Score': [0.639206, 0.431793, 0.389501]})

    bc.log(f'Creating per-year investments graph for years {min_year} to {max_year}')

    ############################################################
    # BUILD DATAFRAME                                          #
    ############################################################

    dfs = {}
    for year in range(min_year, max_year):
        bc.log(f'Retrieving funding rounds for year {year}')
        bc.indent()

        frs = get_frs(db, f'{year}-01-01', f'{year + 1}-01-01')

        fr_ids = list(frs['FundingRoundID'])
        # fr_ids = fr_ids[:100]

        ############################################################

        bc.log('Retrieving investors...')

        investors_frs = get_investors_frs(db, fr_ids)

        ############################################################

        bc.log('Retrieving investees...')

        frs_investees = get_frs_investees(db, fr_ids)
        investee_ids = list(frs_investees['InvesteeID'])

        ############################################################

        bc.log('Retrieving concepts...')

        investees_concepts = get_investees_concepts(db, investee_ids)

        ############################################################
        # COMPUTE DERIVED DATA                                     #
        ############################################################

        bc.log('Computing historical data for investor-concept edges...')

        df = pd.merge(investors_frs, frs_investees, how='inner', on='FundingRoundID')
        df = pd.merge(df, investees_concepts, how='inner', on='InvesteeID')
        df = pd.merge(df, frs, how='inner', on='FundingRoundID')

        investors_concepts = derive_historical_data(df, groupby_columns=['InvestorID', 'PageID'], date_column='FundingRoundDate', amount_column='FundingAmountPerInvestor_USD', min_date=min_date, max_date=max_date)

        ############################################################

        dfs[year] = {
            'frs': frs,
            'investors_frs': investors_frs[['InvestorID', 'FundingRoundID']],
            'frs_investees': frs_investees[['FundingRoundID', 'InvesteeID']],
            'investees_concepts': investees_concepts,
            'investors_concepts': investors_concepts
        }

        bc.outdent()

    ############################################################

    bc.log('Fetching concept-concept edges from database...')

    table_name = 'graph.Edges_N_Organisation_N_Concept'
    fields = ['DISTINCT PageID']
    cb_concept_ids = list(pd.DataFrame(db.find(table_name, fields=fields), columns=['PageID'])['PageID'])

    table_name = 'graph.Edges_N_Concept_N_Concept_T_GraphScore'
    fields = ['SourcePageID', 'TargetPageID']
    conditions = {'SourcePageID': cb_concept_ids, 'TargetPageID': cb_concept_ids}
    concepts_concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

    ############################################################
    ############################################################

    ############################################################
    ############################################################

    ############################################################
    ############################################################

    ############################################################
    ############################################################

    bc.log('Normalising scores...')

    for year in range(min_year, max_year):
        dfs[year]['investors_concepts']['Score'] = np.tanh(dfs[year]['investors_concepts']['CountAmount'])

    ############################################################

    bc.log("Computing 1-ball for lab research domains")

    def get_ball(concepts):
        forward_sphere = pd.merge(concepts[['PageID']].rename(columns={'PageID': 'SourcePageID'}), concepts_concepts, how='inner', on='SourcePageID')[['TargetPageID']].rename(columns={'TargetPageID': 'PageID'})
        backward_sphere = pd.merge(concepts[['PageID']].rename(columns={'PageID': 'TargetPageID'}), concepts_concepts, how='inner', on='TargetPageID')[['SourcePageID']].rename(columns={'SourcePageID': 'PageID'})
        sphere = pd.concat([forward_sphere, backward_sphere]).drop_duplicates()

        ball = pd.concat([concepts[['PageID']], sphere]).drop_duplicates()
        ball = pd.merge(ball, concepts, how='left', on='PageID').fillna(0)

        return ball

    lab_ball_concepts = get_ball(lab_concepts)

    ############################################################

    bc.log("Extracting investors' historical investment spaces...")

    historical_investment_spaces = dfs[min_year]['investors_concepts'][['InvestorID', 'PageID', 'Score']]

    ############################################################

    def compute_distance_factor(concepts_left, concepts_right):
        intersection = pd.merge(concepts_left.rename(columns={'Score': 'ScoreLeft'}), concepts_right.rename(columns={'Score': 'ScoreRight'}), how='inner', on='PageID')
        min_intersection = intersection[['ScoreLeft', 'ScoreRight']].min(axis=1).sum()

        if min_intersection > 0:
            union = pd.merge(concepts_left.rename(columns={'Score': 'ScoreLeft'}), concepts_right.rename(columns={'Score': 'ScoreRight'}), how='outer', on='PageID')
            max_union = union[['ScoreLeft', 'ScoreRight']].max(axis=1).sum()
            factor = (1 - min_intersection / max_union) if max_union else 1
        else:
            factor = 1

        return factor

    def aggregate(group):
        investor_id = group['InvestorID'].iloc[0]

        year_concepts = group[['PageID', 'Score']]
        his_concepts = historical_investment_spaces.loc[historical_investment_spaces['InvestorID'] == investor_id, ['PageID', 'Score']]

        year_ball_concepts = get_ball(year_concepts)
        his_ball_concepts = get_ball(his_concepts)

        ##############################

        # Distance year-his (historical investment space)
        factor0 = compute_distance_factor(year_concepts, his_concepts)
        factor1 = compute_distance_factor(year_ball_concepts, his_ball_concepts)
        distance_year_his = factor0 * factor1

        ##############################

        # Distance year-lab
        factor0 = compute_distance_factor(year_concepts, lab_concepts)
        factor1 = compute_distance_factor(year_ball_concepts, lab_ball_concepts)
        distance_year_lab = factor0 * factor1

        ##############################

        aggregated_values = {
            'DistYearHis': distance_year_his,
            'DistYearLab': distance_year_lab
        }

        return pd.Series(aggregated_values)

    bc.log('Computing distances his-years and years-lab...')

    for year in range(min_year + 3, max_year):
        bc.log(f'Computing distances his-{year} and {year}-lab...')
        bc.indent()

        dfs[year]['investors_distances'] = dfs[year]['investors_concepts'][['InvestorID', 'PageID', 'Score']].groupby(by='InvestorID').apply(aggregate).reset_index()
        dfs[year]['investors_distances']['HisLabPosition'] = dfs[year]['investors_distances']['DistYearHis'] / (dfs[year]['investors_distances']['DistYearHis'] + dfs[year]['investors_distances']['DistYearLab'])

        print(year)

        unfiltered_distances = dfs[year]['investors_distances']
        filtered_distances = dfs[year]['investors_distances'][dfs[year]['investors_distances']['HisLabPosition'] != 0.5]

        print(unfiltered_distances)
        print(filtered_distances)

        print(unfiltered_distances.info())
        print(filtered_distances.info())

        print(unfiltered_distances.describe(percentiles=np.linspace(0, 1, 11)[1:-1]))
        print(filtered_distances.describe(percentiles=np.linspace(0, 1, 11)[1:-1]))

        bc.outdent()

    ############################################################

    bc.report()


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    main()
