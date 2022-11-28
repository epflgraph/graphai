import pandas as pd

from interfaces.db import DB

from utils.breadcrumb import Breadcrumb
from utils.time.date import *


def get_frs(db, params):
    # Fetch funding rounds from database
    table_name = 'graph.Nodes_N_FundingRound'
    fields = ['FundingRoundID', 'FundingRoundDate', 'FundingRoundType', 'FundingAmount_USD', 'FundingAmount_USD / CB_InvestorCount']
    columns = ['FundingRoundID', 'FundingRoundDate', 'FundingRoundType', 'FundingAmount_USD', 'FundingAmountPerInvestor_USD']

    conditions = {}

    if params.countries is not None:
        conditions['CountryISO3'] = params.countries

    if params.fr_types is not None:
        conditions['FundingRoundType'] = params.fr_types

    frs = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=columns)

    return frs


def get_investors_frs(db, params, fr_ids=None):
    if 'Organization' in params.investor_types:
        # Fetch organization investors from database
        table_name = 'graph.Edges_N_Organisation_N_FundingRound'
        fields = ['OrganisationID', 'FundingRoundID']

        if fr_ids is None:
            conditions = {'Action': 'Invested in'}
        else:
            conditions = {'FundingRoundID': fr_ids, 'Action': 'Invested in'}

        org_investors_frs = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions),
                                         columns=['InvestorID', 'FundingRoundID'])
        org_investors_frs['InvestorType'] = 'Organization'

        if 'Person' not in params.investor_types:
            return org_investors_frs

    if 'Person' in params.investor_types:
        # Fetch person investors from database
        table_name = 'graph.Edges_N_Person_N_FundingRound'
        fields = ['PersonID', 'FundingRoundID']

        if fr_ids is None:
            conditions = {'Action': 'Invested in'}
        else:
            conditions = {'FundingRoundID': fr_ids, 'Action': 'Invested in'}

        person_investors_frs = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions),
                                            columns=['InvestorID', 'FundingRoundID'])
        person_investors_frs['InvestorType'] = 'Person'

        if 'Organization' not in params.investor_types:
            return person_investors_frs

    # Combine organization investors and person investors in a single DataFrame
    investors_frs = pd.concat([org_investors_frs, person_investors_frs])
    investors_frs = investors_frs[['InvestorID', 'InvestorType', 'FundingRoundID']]

    return investors_frs


def get_frs_fundraisers(db, fr_ids=None):
    # Fetch fundraisers from database
    table_name = 'graph.Edges_N_Organisation_N_FundingRound'
    fields = ['FundingRoundID', 'OrganisationID']

    if fr_ids is None:
        conditions = {'Action': 'Raised from'}
    else:
        conditions = {'FundingRoundID': fr_ids, 'Action': 'Raised from'}

    frs_fundraisers = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions),
                                 columns=['FundingRoundID', 'FundraiserID'])
    return frs_fundraisers


def get_fundraisers_concepts(db, fundraiser_ids=None):
    # Fetch concepts from database
    table_name = 'graph.Edges_N_Organisation_N_Concept'
    fields = ['OrganisationID', 'PageID']

    if fundraiser_ids is None:
        conditions = {}
    else:
        conditions = {'OrganisationID': fundraiser_ids}

    fundraisers_concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=['FundraiserID', 'PageID'])
    return fundraisers_concepts


def derive_yearly_data(df, groupby_columns, date_column, amount_column):

    df['YearMin'] = df['Year'].astype(str) + '-01-01'
    df['YearMax'] = (df['Year'] + 1).astype(str) + '-01-01'

    df['ScoreLinCount'] = rescale(df[date_column], df['YearMin'], df['YearMax'])
    df['ScoreLinAmount'] = df['ScoreLinCount'] * df[amount_column]
    df['ScoreQuadCount'] = df['ScoreLinCount'] ** 2
    df['ScoreQuadAmount'] = df['ScoreQuadCount'] * df[amount_column]

    df = df.groupby(groupby_columns).aggregate(
        CountAmount=(amount_column, 'size'),
        MinAmount=(amount_column, 'min'),
        MaxAmount=(amount_column, 'max'),
        MedianAmount=(amount_column, 'median'),
        SumAmount=(amount_column, 'sum'),
        ScoreLinCount=('ScoreLinCount', 'sum'),
        ScoreLinAmount=('ScoreLinAmount', 'sum'),
        ScoreQuadCount=('ScoreQuadCount', 'sum'),
        ScoreQuadAmount=('ScoreQuadAmount', 'sum')
    ).reset_index()

    return df


def create_investments_graph(params):

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

    frs = get_frs(db, params)
    fr_ids = list(frs['FundingRoundID'])

    # Extract year
    dates = pd.to_datetime(frs['FundingRoundDate'], errors='coerce')
    dates = dates.fillna(dates.min())
    frs['Year'] = dates.dt.year

    ############################################################

    bc.log('Retrieving investors...')

    investors_frs = get_investors_frs(db, params, fr_ids)

    investors = investors_frs[['InvestorID', 'InvestorType']].drop_duplicates().reset_index(drop=True)

    ############################################################

    bc.log('Retrieving fundraisers...')

    frs_fundraisers = get_frs_fundraisers(db, fr_ids)

    fundraisers = frs_fundraisers[['FundraiserID']].drop_duplicates().reset_index(drop=True)
    fundraiser_ids = list(fundraisers['FundraiserID'])

    ############################################################

    bc.log('Retrieving concepts...')

    fundraisers_concepts = get_fundraisers_concepts(db, fundraiser_ids)

    ############################################################
    # COMPUTE DERIVED DATA                                     #
    ############################################################

    bc.log('Computing yearly data for investor nodes...')

    df = pd.merge(investors_frs, frs, how='inner', on='FundingRoundID')
    investors_years = derive_yearly_data(df, groupby_columns=['InvestorID', 'InvestorType', 'Year'], date_column='FundingRoundDate', amount_column='FundingAmountPerInvestor_USD')

    ############################################################

    bc.log('Computing yearly data for concept nodes...')

    df = pd.merge(frs_fundraisers, fundraisers_concepts, how='inner', on='FundraiserID')
    df = pd.merge(df, frs, how='inner', on='FundingRoundID')
    concepts_years = derive_yearly_data(df, groupby_columns=['PageID', 'Year'], date_column='FundingRoundDate', amount_column='FundingAmount_USD')

    ############################################################

    bc.log('Computing yearly data for investor-investor edges...')

    # Merge the dataframe investors_frs with itself to obtain all pairs of investors
    df = pd.merge(
        investors_frs[['InvestorID', 'FundingRoundID']].rename(columns={'InvestorID': 'SourceInvestorID'}),
        investors_frs[['InvestorID', 'FundingRoundID']].rename(columns={'InvestorID': 'TargetInvestorID'}),
        how='inner',
        on='FundingRoundID'
    )

    # Keep only lexicographically sorted pairs to avoid duplicates
    df = df[df['SourceInvestorID'] < df['TargetInvestorID']].reset_index(drop=True)

    # Attach fr info and extract yearly data
    df = pd.merge(df, frs, how='inner', on='FundingRoundID')
    investors_investors_years = derive_yearly_data(df, groupby_columns=['SourceInvestorID', 'TargetInvestorID', 'Year'], date_column='FundingRoundDate', amount_column='FundingAmountPerInvestor_USD')

    ############################################################

    bc.log('Computing yearly data for investor-concept edges...')

    df = pd.merge(investors_frs, frs_fundraisers, how='inner', on='FundingRoundID')
    df = pd.merge(df, fundraisers_concepts, how='inner', on='FundraiserID')
    df = pd.merge(df, frs, how='inner', on='FundingRoundID')

    investors_concepts_years = derive_yearly_data(df, groupby_columns=['InvestorID', 'PageID', 'Year'], date_column='FundingRoundDate', amount_column='FundingAmountPerInvestor_USD')

    # Add investor-concept concentration (number of investments in concept / number of investments)
    investors_concepts_years = pd.merge(
        investors_concepts_years,
        investors_years[['InvestorID', 'Year', 'CountAmount']].rename(columns={'CountAmount': 'SumCountAmount'}),
        how='left',
        on=['InvestorID', 'Year']
    )
    investors_concepts_years['Concentration'] = investors_concepts_years['CountAmount'] / investors_concepts_years['SumCountAmount']
    investors_concepts_years = investors_concepts_years.drop(columns='SumCountAmount')

    ############################################################
    # INSERT UNTREATED DATA INTO DATABASE                      #
    ############################################################

    bc.log('Inserting funding rounds into database...')

    table_name = 'ca_temp.Nodes_N_FundingRound'
    definition = ['FundingRoundID CHAR(64)', 'FundingRoundDate DATE', 'FundingRoundType CHAR(32)',
                  'FundingAmount_USD FLOAT', 'FundingAmountPerInvestor_USD FLOAT', 'Year SMALLINT',
                  'PRIMARY KEY FundingRoundID (FundingRoundID)']
    db.drop_create_insert_table(table_name, definition, frs)

    ############################################################

    bc.log('Inserting investors into database...')

    table_name = 'ca_temp.Nodes_N_Investor'
    definition = ['InvestorID CHAR(64)', 'InvestorType CHAR(32)', 'PRIMARY KEY InvestorID (InvestorID)']
    db.drop_create_insert_table(table_name, definition, investors)

    ############################################################

    bc.log('Inserting investors-frs edges into database...')

    table_name = 'ca_temp.Edges_N_Investor_N_FundingRound'
    definition = ['InvestorID CHAR(64)', 'FundingRoundID CHAR(64)', 'KEY InvestorID (InvestorID)',
                  'KEY FundingRoundID (FundingRoundID)']
    db.drop_create_insert_table(table_name, definition, investors_frs[['InvestorID', 'FundingRoundID']])

    ############################################################

    bc.log('Inserting frs-fundraisers edges into database...')

    table_name = 'ca_temp.Edges_N_FundingRound_N_Fundraiser'
    definition = ['FundingRoundID CHAR(64)', 'FundraiserID CHAR(64)', 'KEY FundingRoundID (FundingRoundID)',
                  'KEY FundraiserID (FundraiserID)']
    db.drop_create_insert_table(table_name, definition, frs_fundraisers)

    ############################################################

    bc.log('Inserting fundraisers-concepts edges into database...')

    table_name = 'ca_temp.Edges_N_Fundraiser_N_Concept'
    definition = ['FundraiserID CHAR(64)', 'PageID INT UNSIGNED', 'KEY FundraiserID (FundraiserID)', 'KEY PageID (PageID)']
    db.drop_create_insert_table(table_name, definition, fundraisers_concepts)

    ############################################################

    bc.log('Inserting fundraisers into database...')

    table_name = 'ca_temp.Nodes_N_Fundraiser'
    definition = ['FundraiserID CHAR(64)', 'PRIMARY KEY FundraiserID (FundraiserID)']
    db.drop_create_insert_table(table_name, definition, fundraisers)

    ############################################################
    # INSERT COMPUTED DATA INTO DATABASE                       #
    ############################################################

    bc.log('Inserting investors into database...')

    table_name = 'ca_temp.Nodes_N_Investor_T_Years'
    definition = [
        'InvestorID CHAR(64)',
        'InvestorType CHAR(32)',
        'Year SMALLINT',
        'CountAmount FLOAT',
        'MinAmount FLOAT',
        'MaxAmount FLOAT',
        'MedianAmount FLOAT',
        'SumAmount FLOAT',
        'ScoreLinCount FLOAT',
        'ScoreLinAmount FLOAT',
        'ScoreQuadCount FLOAT',
        'ScoreQuadAmount FLOAT',
        'KEY InvestorID (InvestorID)',
        'KEY Year (Year)'
    ]
    db.drop_create_insert_table(table_name, definition, investors_years)

    ############################################################

    bc.log('Inserting concepts into database...')

    table_name = 'ca_temp.Nodes_N_Concept_T_Years'
    definition = [
        'PageID INT UNSIGNED',
        'Year SMALLINT',
        'CountAmount FLOAT',
        'MinAmount FLOAT',
        'MaxAmount FLOAT',
        'MedianAmount FLOAT',
        'SumAmount FLOAT',
        'ScoreLinCount FLOAT',
        'ScoreLinAmount FLOAT',
        'ScoreQuadCount FLOAT',
        'ScoreQuadAmount FLOAT',
        'KEY PageID (PageID)',
        'KEY Year (Year)'
    ]
    db.drop_create_insert_table(table_name, definition, concepts_years)

    ############################################################

    bc.log('Inserting investor-investor edges into database...')

    table_name = 'ca_temp.Edges_N_Investor_N_Investor_T_Years'
    definition = [
        'SourceInvestorID CHAR(64)',
        'TargetInvestorID CHAR(64)',
        'Year SMALLINT',
        'CountAmount FLOAT',
        'MinAmount FLOAT',
        'MaxAmount FLOAT',
        'MedianAmount FLOAT',
        'SumAmount FLOAT',
        'ScoreLinCount FLOAT',
        'ScoreLinAmount FLOAT',
        'ScoreQuadCount FLOAT',
        'ScoreQuadAmount FLOAT',
        'KEY SourceInvestorID (SourceInvestorID)',
        'KEY TargetInvestorID (TargetInvestorID)',
        'KEY Year (Year)'
    ]
    db.drop_create_insert_table(table_name, definition, investors_investors_years)

    ############################################################

    bc.log('Inserting investors-concepts edges into database...')

    table_name = 'ca_temp.Edges_N_Investor_N_Concept_T_Years'
    definition = [
        'InvestorID CHAR(64)',
        'PageID INT UNSIGNED',
        'Year SMALLINT',
        'CountAmount FLOAT',
        'MinAmount FLOAT',
        'MaxAmount FLOAT',
        'MedianAmount FLOAT',
        'SumAmount FLOAT',
        'ScoreLinCount FLOAT',
        'ScoreLinAmount FLOAT',
        'ScoreQuadCount FLOAT',
        'ScoreQuadAmount FLOAT',
        'Concentration FLOAT',
        'KEY InvestorID (InvestorID)',
        'KEY PageID (PageID)',
        'KEY Year (Year)'
    ]
    db.drop_create_insert_table(table_name, definition, investors_concepts_years)

    ############################################################

    bc.report()


if __name__ == '__main__':
    import investment.parameters as params

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    create_investments_graph(params)
