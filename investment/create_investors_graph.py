import pandas as pd
import numpy as np

from itertools import combinations

from interfaces.db import DB
from utils.time.date import now, rescale
from utils.time.stopwatch import Stopwatch


def retrieve_funding_rounds(min_date, max_date):
    # Instantiate db interface to communicate with database
    db = DB()

    # Retrieve funding rounds in time window
    table_name = 'graph.Nodes_N_FundingRound'
    fields = ['FundingRoundID', 'FundingRoundDate', 'FundingAmount_USD', 'FundingAmount_USD / CB_InvestorCount']
    columns = ['FundingRoundID', 'FundingRoundDate', 'FundingAmount_USD', 'FundingAmountPerInvestor_USD']
    conditions = {'FundingRoundDate': {'>=': min_date, '<': max_date}}
    frs = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=columns)

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


def compute_investors_concepts_history(df):
    # Extract list of investor-concept pairs
    investor_concept_pairs = df[['InvestorID', 'PageID']].drop_duplicates().itertuples(index=False, name=None)

    # Compute investor-concept history for each pair
    history = {}
    for investor_id, concept_id in investor_concept_pairs:
        if investor_id not in history:
            history[investor_id] = {}

        history[investor_id][concept_id] = {}

        # Restrict df to current investor and concept
        investor_concept_df = df[(df['InvestorID'] == investor_id) & (df['PageID'] == concept_id)]

        # Iterate over all dates and gather all investments made by current investor in current concept each day.
        for fr_date in investor_concept_df['FundingRoundDate'].drop_duplicates():
            history[investor_id][concept_id][str(fr_date)] = {}

            # Restrict df to current concept, investor and date
            investor_concept_date_df = investor_concept_df[investor_concept_df['FundingRoundDate'] == fr_date]

            # Iterate over frs and gather all investments made by current investor in current concept on given day.
            total_amount = 0
            for fr_id in investor_concept_date_df['FundingRoundID']:
                # Restrict df to current investor, concept, date and funding round
                investor_concept_date_fr_df = investor_concept_date_df[investor_concept_date_df['FundingRoundID'] == fr_id]

                # There should be exactly one row for each combination of parameters
                assert len(investor_concept_date_fr_df) == 1, f'There should be exactly one row for investor {investor_id} and funding round {fr_id}, {len(investor_concept_date_fr_df)} found.'

                # Extract amount and build history
                amount = investor_concept_date_fr_df['FundingAmountPerInvestor_USD'].iloc[0]
                history[investor_id][concept_id][str(fr_date)][fr_id] = {'amount': amount}
                total_amount += np.nan_to_num(amount)

            # Compute number of investments for the given investor, concept and date
            n_investments = len(history[investor_id][concept_id][str(fr_date)])

            # Store number of investments and total amount for the given investor, concept and date
            history[investor_id][concept_id][str(fr_date)]['n_investments'] = n_investments
            history[investor_id][concept_id][str(fr_date)]['total_amount'] = total_amount

        # Compute number of investments and total investment amount for the given investor and concept
        n_investments = sum(
            [history[investor_id][concept_id][d]['n_investments'] for d in history[investor_id][concept_id]]
        )
        total_amount = sum(
            [history[investor_id][concept_id][d]['total_amount'] for d in history[investor_id][concept_id]]
        )

        # Store number of investments and total investment amount for the given investor and concept
        history[investor_id][concept_id]['n_investments'] = n_investments
        history[investor_id][concept_id]['total_amount'] = total_amount

    # Compute number of investments and total investment amount for the given investor
    n_investments = sum([history[investor_id][concept_id]['n_investments'] for concept_id in history[investor_id]])
    total_amount = sum([history[investor_id][concept_id]['total_amount'] for concept_id in history[investor_id]])

    # Store number of investments and total investment amount for the given investor
    history[investor_id]['n_investments'] = n_investments
    history[investor_id]['total_amount'] = total_amount

    return history


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


def insert_investors_concepts(df, investors_concepts_history, min_date='1990-01-01', max_date='today'):

    if max_date == 'today':
        max_date = str(now().date())

    df = df[['InvestorID', 'PageID']].drop_duplicates()

    def compute_scores(x):
        investor_id = x['InvestorID']
        concept_id = x['PageID']

        dates = [date for date in investors_concepts_history[investor_id][concept_id] if date != 'n_investments' and date != 'total_amount']
        n_investments = [investors_concepts_history[investor_id][concept_id][date]['n_investments'] for date in dates]
        total_amounts = [investors_concepts_history[investor_id][concept_id][date]['total_amount'] for date in dates]
        dates_scaled = [rescale(date, min_date, max_date) for date in dates]

        sum_investments = sum(n_investments)
        sum_amounts = sum(total_amounts)

        hs = {
            'lin': lambda k: dates_scaled[k],
            'quad': lambda k: dates_scaled[k]**2,
            'const': lambda k: 1
        }

        Fs = {
            'n_inv': lambda k: n_investments[k],
            'amount': lambda k: total_amounts[k],
            'n_inv_norm': lambda k: n_investments[k] / sum_investments if sum_investments else 0,
            'amount_norm': lambda k: total_amounts[k] / sum_amounts if sum_amounts else 0
        }

        d = {}
        for h_id, h in hs.items():
            for F_id, F in Fs.items():
                d[f'{h_id}__{F_id}'] = sum([h(k) * F(k) for k in range(len(dates_scaled))])

        return pd.Series(d)

    scores = df.apply(compute_scores, axis=1)
    df = pd.concat([df, scores], axis=1)

    # Instantiate db interface to communicate with database
    db = DB()

    # Insert into DB
    db.create_table_Edges_N_Investor_N_Concept(df)


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

    # Merge dataframes investors-frs-investees-concepts to have all information together
    df = investors_frs \
        .merge(frs_investees, how='inner', on='FundingRoundID') \
        .merge(investees_concepts, how='inner', on='InvesteeID') \
        .merge(frs, how='inner', on='FundingRoundID')

    print('Computing investors-concepts history...')
    investors_concepts_history = compute_investors_concepts_history(df)
    sw.tick()

    print('Inserting investors-concepts edges into database...')
    insert_investors_concepts(df, investors_concepts_history)
    sw.report()


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    main()
