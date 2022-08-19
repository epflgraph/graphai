import pandas as pd
from itertools import combinations

from interfaces.db import DB
from utils.time.stopwatch import Stopwatch


def retrieve_funding_rounds(min_date, max_date):
    # Instantiate db interface to communicate with database
    db = DB()

    # Retrieve funding rounds in time window
    table_name = 'graph.Nodes_N_FundingRound'
    fields = ['FundingRoundID', 'FundingRoundDate', 'FundingAmount_USD']
    conditions = {'FundingRoundDate': {'>=': min_date, '<': max_date}}
    frs = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

    return frs


def retrieve_investors(frs):
    # Extract list of funding round ids
    fr_ids = list(frs['FundingRoundID'])

    # Instantiate db interface to communicate with database
    db = DB()

    # Retrieve organization investors
    table_name = 'graph.Edges_N_Organisation_N_FundingRound'
    fields = ['OrganisationID', 'FundingRoundID']
    conditions = {'Action': 'Invested in', 'FundingRoundID': fr_ids}
    org_investors_frs = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=['InvestorID', 'FundingRoundID'])

    # Retrieve person investors
    table_name = 'graph.Edges_N_Person_N_FundingRound'
    fields = ['PersonID', 'FundingRoundID']
    conditions = {'Action': 'Invested in', 'FundingRoundID': fr_ids}
    person_investors_frs = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=['InvestorID', 'FundingRoundID'])

    # Add extra column with investor type
    org_investors_frs['InvestorType'] = 'Organization'
    person_investors_frs['InvestorType'] = 'Person'

    # Combine organization investors and person investors in a single DataFrame
    investors_frs = pd.concat([org_investors_frs, person_investors_frs])
    investors_frs = investors_frs[['InvestorID', 'InvestorType', 'FundingRoundID']]

    return investors_frs


def retrieve_investees(frs):
    # Extract list of funding round ids
    fr_ids = list(frs['FundingRoundID'])

    # Instantiate db interface to communicate with database
    db = DB()

    # Retrieve investees
    table_name = 'graph.Edges_N_Organisation_N_FundingRound'
    fields = ['FundingRoundID', 'OrganisationID']
    conditions = {'Action': 'Raised from', 'FundingRoundID': fr_ids}
    frs_investees = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=['FundingRoundID', 'InvesteeID'])

    return frs_investees


def retrieve_concepts(frs_investees):
    # Extract list of investee ids
    investee_ids = list(frs_investees['InvesteeID'])

    # Instantiate db interface to communicate with database
    db = DB()

    # Retrieve investees' concepts
    table_name = 'graph.Edges_N_Organisation_N_Concept'
    fields = ['OrganisationID', 'PageID']
    conditions = {'OrganisationID': investee_ids}
    investees_concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=['InvesteeID', 'PageID'])

    return investees_concepts


def compute_investor_pairs(investors_frs):
    # Create DataFrame with all investor relationships.
    # Two investors are related if they have participated in at least one funding round together
    # in the given time period.
    def combine(group):
        return pd.DataFrame.from_records(combinations(group['InvestorID'], 2))

    investor_pairs = investors_frs.groupby('FundingRoundID').apply(combine).reset_index(level=0)
    investor_pairs.columns = ['FundingRoundID', 'SourceInvestorID', 'TargetInvestorID']

    # Duplicate DataFrame flipping source and target, since the relation is symmetrical
    first_half = pd.DataFrame(investor_pairs[['SourceInvestorID', 'FundingRoundID', 'TargetInvestorID']].values, columns=['SourceInvestorID', 'FundingRoundID', 'TargetInvestorID'])
    second_half = pd.DataFrame(investor_pairs[['TargetInvestorID', 'FundingRoundID', 'SourceInvestorID']].values, columns=['SourceInvestorID', 'FundingRoundID', 'TargetInvestorID'])
    return pd.concat([first_half, second_half]).reset_index(drop=True)


def insert_funding_rounds(frs):
    # Instantiate db interface to communicate with database
    db = DB()

    # Insert into DB
    db.create_table_Nodes_N_FundingRound(frs)


def insert_investors(investors_frs):
    # Drop duplicate investors
    nodes = investors_frs[['InvestorID', 'InvestorType']].drop_duplicates()

    # Instantiate db interface to communicate with database
    db = DB()

    # Insert into DB
    db.create_table_Nodes_N_Investor(nodes)


def insert_investees(frs_investees):
    # Drop duplicate investees
    nodes = frs_investees[['InvesteeID']].drop_duplicates()

    # Instantiate db interface to communicate with database
    db = DB()

    # Insert into DB
    db.create_table_Nodes_N_Investee(nodes)


def insert_investors_frs(investors_frs):
    edges = investors_frs[['InvestorID', 'FundingRoundID']]

    # Instantiate db interface to communicate with database
    db = DB()

    # Insert into DB
    db.create_table_Edges_N_Investor_N_FundingRound(edges)


def insert_investors_investors(investors_investors):
    # Drop duplicate investor pairs
    edges = investors_investors[['SourceInvestorID', 'TargetInvestorID']].drop_duplicates()

    # Instantiate db interface to communicate with database
    db = DB()

    # Insert into DB
    db.create_table_Edges_N_Investor_N_Investor(edges)


def insert_investors_investees(investors_frs, frs_investees):
    # Merge investors and investees through frs and drop duplicates
    edges = pd.merge(investors_frs, frs_investees, how='inner', on='FundingRoundID')[['InvestorID', 'InvesteeID']].drop_duplicates()

    # Instantiate db interface to communicate with database
    db = DB()

    # Insert into DB
    db.create_table_Edges_N_Investor_N_Investee(edges)


def insert_investors_concepts(investors_frs, frs_investees, investees_concepts):
    # Merge investors and concepts through frs and investees and drop duplicates
    edges = pd.merge(investors_frs, frs_investees, how='inner', on='FundingRoundID')
    edges = pd.merge(edges, investees_concepts, how='inner', on='InvesteeID')
    edges = edges[['InvestorID', 'PageID']].drop_duplicates()

    # Instantiate db interface to communicate with database
    db = DB()

    # Insert into DB
    db.create_table_Edges_N_Investor_N_Concept(edges)


def insert_frs_investees(frs_investees):
    edges = frs_investees[['FundingRoundID', 'InvesteeID']]

    # Instantiate db interface to communicate with database
    db = DB()

    # Insert into DB
    db.create_table_Edges_N_FundingRound_N_Investee(edges)


def insert_frs_concepts(frs_investees, investees_concepts):
    # Merge frs and concepts through investees and drop duplicates
    edges = pd.merge(frs_investees, investees_concepts, how='inner', on='InvesteeID')[['FundingRoundID', 'PageID']].drop_duplicates()

    # Instantiate db interface to communicate with database
    db = DB()

    # Insert into DB
    db.create_table_Edges_N_FundingRound_N_Concept(edges)


def insert_investees_concepts(investees_concepts):
    # Instantiate db interface to communicate with database
    db = DB()

    # Insert into DB
    db.create_table_Edges_N_Investee_N_Concept(investees_concepts)


def main():
    sw = Stopwatch()

    min_date = '2021-01-01'
    max_date = '2022-01-01'

    print('Retrieving funding rounds...')
    frs = retrieve_funding_rounds(min_date, max_date)
    sw.tick()

    print('Inserting funding rounds into database...')
    insert_funding_rounds(frs)
    sw.tick()

    print('Retrieving investors...')
    investors_frs = retrieve_investors(frs)
    sw.tick()

    print('Inserting investors into database...')
    insert_investors(investors_frs)
    sw.tick()

    print('Inserting investors-frs edges into database...')
    insert_investors_frs(investors_frs)
    sw.tick()

    print('Computing investor pairs...')
    investors_investors = compute_investor_pairs(investors_frs)
    sw.tick()

    print('Inserting investor-investor edges into database...')
    insert_investors_investors(investors_investors)
    sw.tick()

    print('Retrieving investees...')
    frs_investees = retrieve_investees(frs)
    sw.tick()

    print('Inserting investees into database...')
    insert_investees(frs_investees)
    sw.tick()

    print('Inserting frs-investees edges into database...')
    insert_frs_investees(frs_investees)
    sw.tick()

    print('Inserting investors-investees edges into database...')
    insert_investors_investees(investors_frs, frs_investees)
    sw.tick()

    print('Retrieving concepts...')
    investees_concepts = retrieve_concepts(frs_investees)
    sw.tick()

    print('Inserting investees-concepts edges into database...')
    insert_investees_concepts(investees_concepts)
    sw.tick()

    print('Inserting frs-concepts edges into database...')
    insert_frs_concepts(frs_investees, investees_concepts)
    sw.tick()

    print('Inserting investors-concepts edges into database...')
    insert_investors_concepts(investors_frs, frs_investees, investees_concepts)
    sw.report()


if __name__ == '__main__':
    main()
