import pandas as pd

from interfaces.db import DB

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
    min_date = '2021-01-01'
    max_date = '2022-01-01'

    bc.log(f'Creating investments graph for time window [{min_date}, {max_date})')

    ############################################################
    # BUILD DATAFRAME                                          #
    ############################################################

    bc.log('Retrieving funding rounds...')

    # Fetch funding rounds in time window from database
    table_name = 'graph.Nodes_N_FundingRound'
    fields = ['FundingRoundID', 'FundingRoundDate', 'FundingAmount_USD', 'FundingAmount_USD / CB_InvestorCount']
    columns = ['FundingRoundID', 'FundingRoundDate', 'FundingAmount_USD', 'FundingAmountPerInvestor_USD']
    conditions = {'FundingRoundDate': {'>=': min_date, '<': max_date}}
    frs = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=columns)

    # Replace na values with 0
    frs = frs.fillna(0)

    fr_ids = list(frs['FundingRoundID'])

    ############################################################

    bc.log('Retrieving investors...')

    # Fetch organization investors from database
    table_name = 'graph.Edges_N_Organisation_N_FundingRound'
    fields = ['OrganisationID', 'FundingRoundID']
    conditions = {'Action': 'Invested in', 'FundingRoundID': fr_ids}
    org_investors_frs = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions),
                                     columns=['InvestorID', 'FundingRoundID'])

    # Fetch person investors from database
    table_name = 'graph.Edges_N_Person_N_FundingRound'
    fields = ['PersonID', 'FundingRoundID']
    conditions = {'Action': 'Invested in', 'FundingRoundID': fr_ids}
    person_investors_frs = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions),
                                        columns=['InvestorID', 'FundingRoundID'])

    # Add extra column with investor type
    org_investors_frs['InvestorType'] = 'Organization'
    person_investors_frs['InvestorType'] = 'Person'

    # Combine organization investors and person investors in a single DataFrame
    investors_frs = pd.concat([org_investors_frs, person_investors_frs])
    investors_frs = investors_frs[['InvestorID', 'InvestorType', 'FundingRoundID']]

    ############################################################

    bc.log('Retrieving investees...')

    # Fetch investees from database
    table_name = 'graph.Edges_N_Organisation_N_FundingRound'
    fields = ['FundingRoundID', 'OrganisationID']
    conditions = {'Action': 'Raised from', 'FundingRoundID': fr_ids}
    frs_investees = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions),
                                 columns=['FundingRoundID', 'InvesteeID'])
    investee_ids = list(frs_investees['InvesteeID'])

    ############################################################

    bc.log('Retrieving concepts...')

    # Fetch concepts from database
    table_name = 'graph.Edges_N_Organisation_N_Concept'
    fields = ['OrganisationID', 'PageID']
    conditions = {'OrganisationID': investee_ids}
    investees_concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions),
                                      columns=['InvesteeID', 'PageID'])

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

    # Drop, recreate table and fill with frs
    table_name = 'ca_temp.Nodes_N_FundingRound'
    definition = ['FundingRoundID CHAR(64)', 'FundingRoundDate DATE', 'FundingAmount_USD FLOAT',
                  'FundingAmountPerInvestor_USD FLOAT', 'PRIMARY KEY FundingRoundID (FundingRoundID)']
    db.drop_create_insert_table(table_name, definition, frs)

    ############################################################

    bc.log('Inserting investors into database...')

    # Drop, recreate table and fill with df
    table_name = 'ca_temp.Nodes_N_Investor'
    definition = [
        'InvestorID CHAR(64)',
        'InvestorType CHAR(32)',
        'CountAmount FLOAT',
        'MinAmount FLOAT',
        'MaxAmount FLOAT',
        'MedianAmount FLOAT',
        'SumAmount FLOAT',
        'ScoreLinCount FLOAT',
        'ScoreLinAmount FLOAT',
        'ScoreQuadCount FLOAT',
        'ScoreQuadAmount FLOAT',
        'ScoreConstCount FLOAT',
        'ScoreConstAmount FLOAT',
        'PRIMARY KEY InvestorID (InvestorID)'
    ]
    db.drop_create_insert_table(table_name, definition, investors)

    ############################################################

    bc.log('Inserting investees into database...')

    # Drop, recreate table and fill with df
    table_name = 'ca_temp.Nodes_N_Investee'
    definition = ['InvesteeID CHAR(64)', 'PRIMARY KEY InvesteeID (InvesteeID)']
    df = frs_investees[['InvesteeID']].drop_duplicates()
    db.drop_create_insert_table(table_name, definition, df)

    ############################################################

    bc.log('Inserting concepts into database...')

    # Drop, recreate table and fill with df
    table_name = 'ca_temp.Nodes_N_Concept'
    definition = [
        'PageID INT UNSIGNED',
        'CountAmount FLOAT',
        'MinAmount FLOAT',
        'MaxAmount FLOAT',
        'MedianAmount FLOAT',
        'SumAmount FLOAT',
        'ScoreLinCount FLOAT',
        'ScoreLinAmount FLOAT',
        'ScoreQuadCount FLOAT',
        'ScoreQuadAmount FLOAT',
        'ScoreConstCount FLOAT',
        'ScoreConstAmount FLOAT',
        'PRIMARY KEY PageID (PageID)'
    ]
    db.drop_create_insert_table(table_name, definition, concepts)

    ############################################################
    # INSERT EDGES INTO DATABASE                               #
    ############################################################

    bc.log('Inserting investors-frs edges into database...')

    # Drop, recreate table and fill with df
    table_name = 'ca_temp.Edges_N_Investor_N_FundingRound'
    definition = ['InvestorID CHAR(64)', 'FundingRoundID CHAR(64)', 'KEY InvestorID (InvestorID)',
                  'KEY FundingRoundID (FundingRoundID)']
    df = investors_frs[['InvestorID', 'FundingRoundID']]
    db.drop_create_insert_table(table_name, definition, df)

    ############################################################

    bc.log('Inserting frs-investees edges into database...')

    # Drop, recreate table and fill with df
    table_name = 'ca_temp.Edges_N_FundingRound_N_Investee'
    definition = ['FundingRoundID CHAR(64)', 'InvesteeID CHAR(64)', 'KEY FundingRoundID (FundingRoundID)',
                  'KEY InvesteeID (InvesteeID)']
    df = frs_investees[['FundingRoundID', 'InvesteeID']]
    db.drop_create_insert_table(table_name, definition, df)

    ############################################################

    bc.log('Inserting investees-concepts edges into database...')

    # Drop, recreate table and fill with df
    table_name = 'ca_temp.Edges_N_Investee_N_Concept'
    definition = ['InvesteeID CHAR(64)', 'PageID INT UNSIGNED', 'KEY InvesteeID (InvesteeID)', 'KEY PageID (PageID)']
    db.drop_create_insert_table(table_name, definition, investees_concepts)

    ############################################################

    bc.log('Inserting investor-investor edges into database...')

    # Drop, recreate table and fill with df
    table_name = 'ca_temp.Edges_N_Investor_N_Investor'
    definition = [
        'SourceInvestorID CHAR(64)',
        'TargetInvestorID CHAR(64)',
        'CountAmount FLOAT',
        'MinAmount FLOAT',
        'MaxAmount FLOAT',
        'MedianAmount FLOAT',
        'SumAmount FLOAT',
        'ScoreLinCount FLOAT',
        'ScoreLinAmount FLOAT',
        'ScoreQuadCount FLOAT',
        'ScoreQuadAmount FLOAT',
        'ScoreConstCount FLOAT',
        'ScoreConstAmount FLOAT',
        'KEY SourceInvestorID (SourceInvestorID)',
        'KEY TargetInvestorID (TargetInvestorID)'
    ]
    db.drop_create_insert_table(table_name, definition, investors_investors)

    ############################################################

    bc.log('Inserting investors-concepts edges into database...')

    # Drop, recreate table and fill with df
    table_name = 'ca_temp.Edges_N_Investor_N_Concept'
    definition = [
        'InvestorID CHAR(64)',
        'PageID INT UNSIGNED',
        'CountAmount FLOAT',
        'MinAmount FLOAT',
        'MaxAmount FLOAT',
        'MedianAmount FLOAT',
        'SumAmount FLOAT',
        'ScoreLinCount FLOAT',
        'ScoreLinAmount FLOAT',
        'ScoreQuadCount FLOAT',
        'ScoreQuadAmount FLOAT',
        'ScoreConstCount FLOAT',
        'ScoreConstAmount FLOAT',
        'KEY InvestorID (InvestorID)',
        'KEY PageID (PageID)'
    ]
    db.drop_create_insert_table(table_name, definition, investors_concepts)

    ############################################################

    bc.report()


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    main()
