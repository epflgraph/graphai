import pandas as pd

from interfaces.db import DB

from investment.data import *

from utils.breadcrumb import Breadcrumb
from utils.time.date import now, rescale


def derive_historical_data(df, groupby_columns, date_column, amount_column, min_date, max_date):

    if max_date == 'today':
        max_date = str(now().date())

    def aggregate(group):
        # TODO perhaps ignore NA values (converted to zero here already) in the computation of these statistics
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

        rescaled_dates = group[date_column].apply(lambda d: rescale(str(d), min_date, max_date))

        hs = {
            'Lin': lambda x: x,
            'Quad': lambda x: x ** 2,
            'Const': lambda x: 1
        }

        Fs = {
            'Count': lambda x: 1,
            'Amount': lambda x: x
        }

        for h_id, h in hs.items():
            for F_id, F in Fs.items():
                aggregated_values[f'Score{h_id}{F_id}'] = (rescaled_dates.apply(h) * group[amount_column].apply(F)).sum()

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

    # Define all history time window, investments outside it will be ignored.
    min_date = '2018-01-01'
    max_date = '2022-01-01'

    bc.log(f'Creating investments graph for time window [{min_date}, {max_date})')

    ############################################################
    # BUILD DATAFRAME                                          #
    ############################################################

    bc.log('Retrieving funding rounds in time window...')

    frs = get_frs(db, min_date, max_date)

    fr_ids = list(frs['FundingRoundID'])

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

    bc.log('Computing historical data for investor nodes...')

    df = pd.merge(investors_frs, frs, how='inner', on='FundingRoundID')
    investors = derive_historical_data(df, groupby_columns=['InvestorID', 'InvestorType'], date_column='FundingRoundDate', amount_column='FundingAmountPerInvestor_USD', min_date=min_date, max_date=max_date)

    ############################################################

    bc.log('Computing historical data for concept nodes...')

    df = pd.merge(frs_investees, investees_concepts, how='inner', on='InvesteeID')
    df = pd.merge(df, frs, how='inner', on='FundingRoundID')
    concepts = derive_historical_data(df, groupby_columns='PageID', date_column='FundingRoundDate', amount_column='FundingAmount_USD', min_date=min_date, max_date=max_date)

    ############################################################

    bc.log('Computing historical data for investor-investor edges...')

    # Merge the dataframe investors_frs with itself to obtain all pairs of investors
    investors_investors = pd.merge(investors_frs[['InvestorID', 'FundingRoundID']], investors_frs[['InvestorID', 'FundingRoundID']], how='inner', on='FundingRoundID')
    investors_investors.columns = ['SourceInvestorID', 'FundingRoundID', 'TargetInvestorID']

    # Keep only lexicographically sorted pairs to avoid duplicates
    investors_investors = investors_investors[investors_investors['SourceInvestorID'] < investors_investors['TargetInvestorID']].reset_index(drop=True)

    # Attach fr info and extract historical data
    df = pd.merge(investors_investors, frs, how='inner', on='FundingRoundID')
    investors_investors = derive_historical_data(df, groupby_columns=['SourceInvestorID', 'TargetInvestorID'], date_column='FundingRoundDate', amount_column='FundingAmountPerInvestor_USD', min_date=min_date, max_date=max_date)

    ############################################################

    bc.log('Computing historical data for investor-concept edges...')

    df = pd.merge(investors_frs, frs_investees, how='inner', on='FundingRoundID')
    df = pd.merge(df, investees_concepts, how='inner', on='InvesteeID')
    df = pd.merge(df, frs, how='inner', on='FundingRoundID')

    investors_concepts = derive_historical_data(df, groupby_columns=['InvestorID', 'PageID'], date_column='FundingRoundDate', amount_column='FundingAmountPerInvestor_USD', min_date=min_date, max_date=max_date)

    ############################################################
    # INSERT NODES INTO DATABASE                               #
    ############################################################

    bc.log('Inserting funding rounds into database...')

    save_frs(db, frs)

    ############################################################

    bc.log('Inserting investors into database...')

    save_investors(db, investors)

    ############################################################

    bc.log('Inserting investees into database...')

    investees = frs_investees[['InvesteeID']].drop_duplicates()
    save_investees(db, investees)

    ############################################################

    bc.log('Inserting concepts into database...')

    save_concepts(db, concepts)

    ############################################################
    # INSERT EDGES INTO DATABASE                               #
    ############################################################

    bc.log('Inserting investors-frs edges into database...')

    save_investors_frs(db, investors_frs[['InvestorID', 'FundingRoundID']])

    ############################################################

    bc.log('Inserting frs-investees edges into database...')

    save_frs_investees(db, frs_investees[['FundingRoundID', 'InvesteeID']])

    ############################################################

    bc.log('Inserting investees-concepts edges into database...')

    save_investees_concepts(db, investees_concepts)

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
