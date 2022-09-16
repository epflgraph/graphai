import pandas as pd


def get_frs(db, min_date, max_date):
    # Fetch funding rounds in time window from database
    table_name = 'graph.Nodes_N_FundingRound'
    fields = ['FundingRoundID', 'FundingRoundDate', 'FundingAmount_USD', 'FundingAmount_USD / CB_InvestorCount']
    columns = ['FundingRoundID', 'FundingRoundDate', 'FundingAmount_USD', 'FundingAmountPerInvestor_USD']
    conditions = {'FundingRoundDate': {'>=': min_date, '<': max_date}}
    frs = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=columns)

    # Replace na values with 0
    frs = frs.fillna(0)

    return frs


def get_investors_frs(db, fr_ids):
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

    return investors_frs


def get_frs_investees(db, fr_ids):
    # Fetch investees from database
    table_name = 'graph.Edges_N_Organisation_N_FundingRound'
    fields = ['FundingRoundID', 'OrganisationID']
    conditions = {'Action': 'Raised from', 'FundingRoundID': fr_ids}
    frs_investees = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions),
                                 columns=['FundingRoundID', 'InvesteeID'])
    return frs_investees


def get_investees_concepts(db, investee_ids):
    # Fetch concepts from database
    table_name = 'graph.Edges_N_Organisation_N_Concept'
    fields = ['OrganisationID', 'PageID']
    conditions = {'OrganisationID': investee_ids}
    investees_concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions),
                                      columns=['InvesteeID', 'PageID'])
    return investees_concepts


############################################################


def save_frs(db, frs):
    # Drop, recreate table and fill with frs
    table_name = 'ca_temp.Nodes_N_FundingRound'
    definition = ['FundingRoundID CHAR(64)', 'FundingRoundDate DATE', 'FundingAmount_USD FLOAT',
                  'FundingAmountPerInvestor_USD FLOAT', 'PRIMARY KEY FundingRoundID (FundingRoundID)']
    db.drop_create_insert_table(table_name, definition, frs)


def save_investors(db, investors):
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


def save_investees(db, investees):
    # Drop, recreate table and fill with df
    table_name = 'ca_temp.Nodes_N_Investee'
    definition = ['InvesteeID CHAR(64)', 'PRIMARY KEY InvesteeID (InvesteeID)']
    db.drop_create_insert_table(table_name, definition, investees)


def save_concepts(db, concepts):
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


def save_investors_frs(db, investors_frs):
    # Drop, recreate table and fill with df
    table_name = 'ca_temp.Edges_N_Investor_N_FundingRound'
    definition = ['InvestorID CHAR(64)', 'FundingRoundID CHAR(64)', 'KEY InvestorID (InvestorID)',
                  'KEY FundingRoundID (FundingRoundID)']
    db.drop_create_insert_table(table_name, definition, investors_frs)


def save_frs_investees(db, frs_investees):
    # Drop, recreate table and fill with df
    table_name = 'ca_temp.Edges_N_FundingRound_N_Investee'
    definition = ['FundingRoundID CHAR(64)', 'InvesteeID CHAR(64)', 'KEY FundingRoundID (FundingRoundID)',
                  'KEY InvesteeID (InvesteeID)']
    db.drop_create_insert_table(table_name, definition, frs_investees)


def save_investees_concepts(db, investees_concepts):
    # Drop, recreate table and fill with df
    table_name = 'ca_temp.Edges_N_Investee_N_Concept'
    definition = ['InvesteeID CHAR(64)', 'PageID INT UNSIGNED', 'KEY InvesteeID (InvesteeID)', 'KEY PageID (PageID)']
    db.drop_create_insert_table(table_name, definition, investees_concepts)


def save_investors_investors(db, investors_investors):
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


def save_investors_concepts(db, investors_concepts):
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






