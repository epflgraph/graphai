import pandas as pd
from itertools import combinations

from interfaces.db import DB
from utils.time.stopwatch import Stopwatch


def retrieve_funding_rounds(min_date, max_date):

    # Instantiate db interface to communicate with database
    db = DB()

    # Retrieve funding rounds in time window
    fields = ['FundingRoundID', 'FundingRoundDate', 'FundingAmount_USD']
    columns = ['fr_id', 'date', 'amount']
    frs = pd.DataFrame(db.get_funding_rounds(min_date=min_date, max_date=max_date, fields=fields), columns=columns)

    return frs


def retrieve_investors(frs):
    # Instantiate db interface to communicate with database
    db = DB()

    # Extract list of funding round ids
    fr_ids = list(frs['fr_id'])

    # Retrieve organization and person investors and combine them in a single DataFrame
    org_investors_frs = pd.DataFrame(db.get_org_investors_funding_rounds(fr_ids=fr_ids), columns=['investor_id', 'fr_id'])
    org_investors_frs['investor_type'] = 'Organization'
    person_investors_frs = pd.DataFrame(db.get_person_investors_funding_rounds(fr_ids=fr_ids), columns=['investor_id', 'fr_id'])
    person_investors_frs['investor_type'] = 'Person'
    investors_frs = pd.concat([org_investors_frs, person_investors_frs])
    investors_frs = investors_frs[['investor_id', 'investor_type', 'fr_id']]

    return investors_frs


def retrieve_investees(frs):
    # Instantiate db interface to communicate with database
    db = DB()

    # Extract list of funding round ids
    fr_ids = list(frs['fr_id'])

    # Retrieve investees
    frs_investees = pd.DataFrame(db.get_funding_rounds_investees(fr_ids=fr_ids), columns=['fr_id', 'investee_id'])

    return frs_investees


def retrieve_concepts(frs_investees):
    # Instantiate db interface to communicate with database
    db = DB()

    # Extract list of investee ids
    investee_ids = list(frs_investees['investee_id'])

    # Retrieve investees' concepts
    investees_concepts = pd.DataFrame(db.get_investees_concepts(org_ids=investee_ids), columns=['investee_id', 'concept_id'])

    return investees_concepts


def compute_investor_pairs(investors_frs):
    # Create DataFrame with all investor relationships.
    # Two investors are related if they have participated in at least one funding round together
    # in the given time period.
    def combine(group):
        return pd.DataFrame.from_records(combinations(group['investor_id'], 2))

    investor_pairs = investors_frs.groupby('fr_id').apply(combine).reset_index(level=0)
    investor_pairs.columns = ['fr_id', 'source_investor_id', 'target_investor_id']

    # Duplicate DataFrame flipping source and target, since the relation is symmetrical
    first_half = pd.DataFrame(investor_pairs[['source_investor_id', 'fr_id', 'target_investor_id']].values, columns=['source_investor_id', 'fr_id', 'target_investor_id'])
    second_half = pd.DataFrame(investor_pairs[['target_investor_id', 'fr_id', 'source_investor_id']].values, columns=['source_investor_id', 'fr_id', 'target_investor_id'])
    return pd.concat([first_half, second_half]).reset_index(drop=True)


def insert_funding_rounds(frs):
    # Instantiate db interface to communicate with database
    db = DB()

    # Insert into DB
    db.create_table_Nodes_N_FundingRound(frs)


def insert_investors(investors_frs):
    # Instantiate db interface to communicate with database
    db = DB()

    # Drop duplicate investors
    nodes = investors_frs[['investor_id', 'investor_type']].drop_duplicates()

    # Insert into DB
    db.create_table_Nodes_N_Investor(nodes)


def insert_investees(frs_investees):
    # Instantiate db interface to communicate with database
    db = DB()

    # Drop duplicate investees
    nodes = frs_investees[['investee_id']].drop_duplicates()

    # Insert into DB
    db.create_table_Nodes_N_Investee(nodes)


def insert_investors_frs(investors_frs):
    # Instantiate db interface to communicate with database
    db = DB()

    edges = investors_frs[['investor_id', 'fr_id']]

    # Insert into DB
    db.create_table_Edges_N_Investor_N_FundingRound(edges)


def insert_investors_investors(investors_investors):
    # Instantiate db interface to communicate with database
    db = DB()

    # Drop duplicate investor pairs
    edges = investors_investors[['source_investor_id', 'target_investor_id']].drop_duplicates()

    # Insert into DB
    db.create_table_Edges_N_Investor_N_Investor(edges)


def insert_investors_investees(investors_frs, frs_investees):
    # Instantiate db interface to communicate with database
    db = DB()

    # Merge investors and investees through frs and drop duplicates
    edges = pd.merge(investors_frs, frs_investees, how='inner', on='fr_id')[['investor_id', 'investee_id']].drop_duplicates()

    # Insert into DB
    db.create_table_Edges_N_Investor_N_Investee(edges)


def insert_investors_concepts(investors_frs, frs_investees, investees_concepts):
    # Instantiate db interface to communicate with database
    db = DB()

    # Merge investors and concepts through frs and investees and drop duplicates
    edges = pd.merge(investors_frs, frs_investees, how='inner', on='fr_id')
    edges = pd.merge(edges, investees_concepts, how='inner', on='investee_id')
    edges = edges[['investor_id', 'concept_id']].drop_duplicates()

    # Insert into DB
    db.create_table_Edges_N_Investor_N_Concept(edges)


def insert_frs_investees(frs_investees):
    # Instantiate db interface to communicate with database
    db = DB()

    edges = frs_investees[['fr_id', 'investee_id']]

    # Insert into DB
    db.create_table_Edges_N_FundingRound_N_Investee(edges)


def insert_frs_concepts(frs_investees, investees_concepts):
    # Instantiate db interface to communicate with database
    db = DB()

    # Merge frs and concepts through investees and drop duplicates
    edges = pd.merge(frs_investees, investees_concepts, how='inner', on='investee_id')[['fr_id', 'concept_id']].drop_duplicates()

    # Insert into DB
    db.create_table_Edges_N_FundingRound_N_Concept(edges)


def insert_investees_concepts(investees_concepts):
    # Instantiate db interface to communicate with database
    db = DB()

    # Insert into DB
    db.create_table_Edges_N_Investee_N_Concept(investees_concepts)


def main():
    sw = Stopwatch()

    min_date = '2018-01-01'
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
