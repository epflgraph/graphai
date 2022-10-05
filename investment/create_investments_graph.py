import warnings
import pandas as pd

from interfaces.db import DB

from investment.data import *

from utils.breadcrumb import Breadcrumb
from utils.time.date import *


def derive_historical_data(df, groupby_columns, date_column, amount_column):

    # if max_date == 'today':
    #     max_date = str(now().date())

    def aggregate(group):
        # Catch warnings from median of series containing only NA values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            count_amount = len(group[amount_column])
            min_amount = group[amount_column].min()
            max_amount = group[amount_column].max()
            median_amount = group[amount_column].median()
            sum_amount = group[amount_column].sum()

        aggregated_values = {
            'CountAmount': count_amount,
            'MinAmount': min_amount,
            'MaxAmount': max_amount,
            'MedianAmount': median_amount,
            'SumAmount': sum_amount
        }

        # Compute rescaled dates
        year = group['Year'].min()
        min_date = f'{year}-01-01'
        max_date = f'{year + 1}-01-01'
        rescaled_dates = rescale(group[date_column], min_date, max_date)

        # Derive scores using rescaled dates
        aggregated_values['ScoreLinCount'] = rescaled_dates.sum()
        aggregated_values['ScoreLinAmount'] = (rescaled_dates * group[amount_column]).sum()
        aggregated_values['ScoreQuadCount'] = (rescaled_dates ** 2).sum()
        aggregated_values['ScoreQuadAmount'] = ((rescaled_dates ** 2) * group[amount_column]).sum()

        return pd.Series(aggregated_values)

    return df.groupby(groupby_columns).apply(aggregate).reset_index()


def main():

    ############################################################
    # INITIALIZATION                                           #
    ############################################################

    # Initialize breadcrumb to log and keep track of time
    bc = Breadcrumb()

    # Instantiate db interface to communicate with database
    db = DB()

    bc.log(f'Creating investments graph...')

    ############################################################
    # BUILD DATAFRAME                                          #
    ############################################################

    bc.log('Retrieving funding rounds in time window...')

    frs = get_frs(db)

    # TO BE DELETED
    # # Compute max_date in the funding rounds and define min_date to be 4 years before
    # max_date = str(frs['FundingRoundDate'].max())
    # min_date = add_years(max_date, -4)

    # Extract year
    dates = pd.to_datetime(frs['FundingRoundDate'], errors='coerce')
    dates = dates.fillna(dates.min())
    frs['Year'] = dates.dt.year

    # frs = frs[:1000]

    ############################################################

    bc.log('Retrieving investors...')

    investors_frs = get_investors_frs(db)

    ############################################################

    bc.log('Retrieving fundraisers...')

    frs_fundraisers = get_frs_fundraisers(db)

    ############################################################

    bc.log('Retrieving concepts...')

    fundraisers_concepts = get_fundraisers_concepts(db)

    ############################################################
    # COMPUTE DERIVED DATA                                     #
    ############################################################

    bc.log('Computing historical data for investor nodes...')

    df = pd.merge(investors_frs, frs, how='inner', on='FundingRoundID')
    investors = derive_historical_data(df, groupby_columns=['InvestorID', 'InvestorType', 'Year'], date_column='FundingRoundDate', amount_column='FundingAmountPerInvestor_USD')

    ############################################################

    bc.log('Computing historical data for concept nodes...')

    df = pd.merge(frs_fundraisers, fundraisers_concepts, how='inner', on='FundraiserID')
    df = pd.merge(df, frs, how='inner', on='FundingRoundID')
    concepts = derive_historical_data(df, groupby_columns=['PageID', 'Year'], date_column='FundingRoundDate', amount_column='FundingAmount_USD')

    ############################################################

    bc.log('Computing historical data for investor-investor edges...')

    # Merge the dataframe investors_frs with itself to obtain all pairs of investors
    investors_investors = pd.merge(
        investors_frs[['InvestorID', 'FundingRoundID']].rename(columns={'InvestorID': 'SourceInvestorID'}),
        investors_frs[['InvestorID', 'FundingRoundID']].rename(columns={'InvestorID': 'TargetInvestorID'}),
        how='inner',
        on='FundingRoundID'
    )

    # Keep only lexicographically sorted pairs to avoid duplicates
    investors_investors = investors_investors[investors_investors['SourceInvestorID'] < investors_investors['TargetInvestorID']].reset_index(drop=True)

    # Attach fr info and extract historical data
    df = pd.merge(investors_investors, frs, how='inner', on='FundingRoundID')
    investors_investors = derive_historical_data(df, groupby_columns=['SourceInvestorID', 'TargetInvestorID', 'Year'], date_column='FundingRoundDate', amount_column='FundingAmountPerInvestor_USD')

    ############################################################

    bc.log('Computing historical data for investor-concept edges...')

    df = pd.merge(investors_frs, frs_fundraisers, how='inner', on='FundingRoundID')
    df = pd.merge(df, fundraisers_concepts, how='inner', on='FundraiserID')
    df = pd.merge(df, frs, how='inner', on='FundingRoundID')

    investors_concepts = derive_historical_data(df, groupby_columns=['InvestorID', 'PageID', 'Year'], date_column='FundingRoundDate', amount_column='FundingAmountPerInvestor_USD')

    ############################################################
    # INSERT UNTREATED DATA INTO DATABASE                      #
    ############################################################

    bc.log('Inserting funding rounds into database...')

    save_frs(db, frs)

    ############################################################

    bc.log('Inserting investors-frs edges into database...')

    save_investors_frs(db, investors_frs[['InvestorID', 'FundingRoundID']])

    ############################################################

    bc.log('Inserting frs-fundraisers edges into database...')

    save_frs_fundraisers(db, frs_fundraisers[['FundingRoundID', 'FundraiserID']])

    ############################################################

    bc.log('Inserting fundraisers-concepts edges into database...')

    save_fundraisers_concepts(db, fundraisers_concepts)

    ############################################################

    bc.log('Inserting fundraisers into database...')

    fundraisers = frs_fundraisers[['FundraiserID']].drop_duplicates()
    save_fundraisers(db, fundraisers)

    ############################################################
    # INSERT COMPUTED DATA INTO DATABASE                       #
    ############################################################

    bc.log('Inserting investors into database...')

    save_investors(db, investors)

    ############################################################

    bc.log('Inserting concepts into database...')

    save_concepts(db, concepts)

    ############################################################

    bc.log('Inserting investor-investor edges into database...')

    save_investors_investors(db, investors_investors)

    ############################################################

    bc.log('Inserting investors-concepts edges into database...')

    save_investors_concepts(db, investors_concepts)

    ############################################################

    bc.report()


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    main()
