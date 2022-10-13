import pandas as pd


def get_frs(db):
    # Fetch funding rounds in time window from database
    table_name = 'graph.Nodes_N_FundingRound'
    fields = ['FundingRoundID', 'FundingRoundDate', 'FundingRoundType', 'FundingAmount_USD', 'FundingAmount_USD / CB_InvestorCount']
    columns = ['FundingRoundID', 'FundingRoundDate', 'FundingRoundType', 'FundingAmount_USD', 'FundingAmountPerInvestor_USD']
    frs = pd.DataFrame(db.find(table_name, fields=fields), columns=columns)

    return frs


def get_investors_frs(db):
    # Fetch organization investors from database
    table_name = 'graph.Edges_N_Organisation_N_FundingRound'
    fields = ['OrganisationID', 'FundingRoundID']
    conditions = {'Action': 'Invested in'}
    org_investors_frs = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions),
                                     columns=['InvestorID', 'FundingRoundID'])

    # Fetch person investors from database
    table_name = 'graph.Edges_N_Person_N_FundingRound'
    fields = ['PersonID', 'FundingRoundID']
    conditions = {'Action': 'Invested in'}
    person_investors_frs = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions),
                                        columns=['InvestorID', 'FundingRoundID'])

    # Add extra column with investor type
    org_investors_frs['InvestorType'] = 'Organization'
    person_investors_frs['InvestorType'] = 'Person'

    # Combine organization investors and person investors in a single DataFrame
    investors_frs = pd.concat([org_investors_frs, person_investors_frs])
    investors_frs = investors_frs[['InvestorID', 'InvestorType', 'FundingRoundID']]

    return investors_frs


def get_frs_fundraisers(db):
    # Fetch fundraisers from database
    table_name = 'graph.Edges_N_Organisation_N_FundingRound'
    fields = ['FundingRoundID', 'OrganisationID']
    conditions = {'Action': 'Raised from'}
    frs_fundraisers = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions),
                                 columns=['FundingRoundID', 'FundraiserID'])
    return frs_fundraisers


def get_fundraisers_concepts(db):
    # Fetch concepts from database
    table_name = 'graph.Edges_N_Organisation_N_Concept'
    fields = ['OrganisationID', 'PageID']
    fundraisers_concepts = pd.DataFrame(db.find(table_name, fields=fields), columns=['FundraiserID', 'PageID'])
    return fundraisers_concepts


############################################################


def save_frs(db, frs):
    # Drop, recreate table and fill with frs
    table_name = 'ca_temp.Nodes_N_FundingRound'
    definition = ['FundingRoundID CHAR(64)', 'FundingRoundDate DATE', 'FundingRoundType CHAR(32)',
                  'FundingAmount_USD FLOAT', 'FundingAmountPerInvestor_USD FLOAT', 'Year SMALLINT',
                  'PRIMARY KEY FundingRoundID (FundingRoundID)']
    db.drop_create_insert_table(table_name, definition, frs)


def save_investors(db, investors):
    # Drop, recreate table and fill with df
    table_name = 'ca_temp.Nodes_N_Investor'
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
    db.drop_create_insert_table(table_name, definition, investors)


def save_fundraisers(db, fundraisers):
    # Drop, recreate table and fill with df
    table_name = 'ca_temp.Nodes_N_Fundraiser'
    definition = ['FundraiserID CHAR(64)', 'PRIMARY KEY FundraiserID (FundraiserID)']
    db.drop_create_insert_table(table_name, definition, fundraisers)


def save_concepts(db, concepts):
    # Drop, recreate table and fill with df
    table_name = 'ca_temp.Nodes_N_Concept'
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
    db.drop_create_insert_table(table_name, definition, concepts)


def save_investors_frs(db, investors_frs):
    # Drop, recreate table and fill with df
    table_name = 'ca_temp.Edges_N_Investor_N_FundingRound'
    definition = ['InvestorID CHAR(64)', 'FundingRoundID CHAR(64)', 'KEY InvestorID (InvestorID)',
                  'KEY FundingRoundID (FundingRoundID)']
    db.drop_create_insert_table(table_name, definition, investors_frs)


def save_frs_fundraisers(db, frs_fundraisers):
    # Drop, recreate table and fill with df
    table_name = 'ca_temp.Edges_N_FundingRound_N_Fundraiser'
    definition = ['FundingRoundID CHAR(64)', 'FundraiserID CHAR(64)', 'KEY FundingRoundID (FundingRoundID)',
                  'KEY FundraiserID (FundraiserID)']
    db.drop_create_insert_table(table_name, definition, frs_fundraisers)


def save_fundraisers_concepts(db, fundraisers_concepts):
    # Drop, recreate table and fill with df
    table_name = 'ca_temp.Edges_N_Fundraiser_N_Concept'
    definition = ['FundraiserID CHAR(64)', 'PageID INT UNSIGNED', 'KEY FundraiserID (FundraiserID)', 'KEY PageID (PageID)']
    db.drop_create_insert_table(table_name, definition, fundraisers_concepts)


def save_investors_investors(db, investors_investors):
    # Drop, recreate table and fill with df
    table_name = 'ca_temp.Edges_N_Investor_N_Investor'
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
    db.drop_create_insert_table(table_name, definition, investors_investors)


def save_investors_concepts(db, investors_concepts):
    # Drop, recreate table and fill with df
    table_name = 'ca_temp.Edges_N_Investor_N_Concept'
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
        'SumCountAmount FLOAT',
        'CountAmountRatio FLOAT',
        'KEY InvestorID (InvestorID)',
        'KEY PageID (PageID)',
        'KEY Year (Year)'
    ]
    db.drop_create_insert_table(table_name, definition, investors_concepts)






