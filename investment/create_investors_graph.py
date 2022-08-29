import pandas as pd
import numpy as np

from itertools import combinations

from interfaces.db import DB

from utils.text.io import log
from utils.time.date import now, rescale
from utils.time.stopwatch import Stopwatch


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


def compute_scores_investors_concepts(df, investors_concepts_history, min_date='1990-01-01', max_date='today'):

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

    return df


def main():
    # Initialize stopwatch to keep track of time
    sw = Stopwatch()

    # Instantiate db interface to communicate with database
    db = DB()

    # Define all history time window, investments outside it will be ignored.
    min_date = '2021-01-01'
    max_date = '2022-01-01'

    ###################
    # BUILD DATAFRAME #
    ###################

    log('Retrieving funding rounds...')

    # Fetch funding rounds in time window from database
    table_name = 'graph.Nodes_N_FundingRound'
    fields = ['FundingRoundID', 'FundingRoundDate', 'FundingAmount_USD', 'FundingAmount_USD / CB_InvestorCount']
    columns = ['FundingRoundID', 'FundingRoundDate', 'FundingAmount_USD', 'FundingAmountPerInvestor_USD']
    conditions = {'FundingRoundDate': {'>=': min_date, '<': max_date}}
    frs = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=columns)
    fr_ids = list(frs['FundingRoundID'])

    log(f'    {sw.delta():.3f}s', color='green')

    ############################################################

    log('Retrieving investors...')

    # Fetch organization investors from database
    table_name = 'graph.Edges_N_Organisation_N_FundingRound'
    fields = ['OrganisationID', 'FundingRoundID']
    conditions = {'Action': 'Invested in', 'FundingRoundID': fr_ids}
    org_investors_frs = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=['InvestorID', 'FundingRoundID'])

    # Fetch person investors from database
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

    log(f'    {sw.delta():.3f}s', color='green')

    ############################################################

    log('Inserting funding rounds into database...')

    # Drop, recreate table and fill with frs
    table_name = 'ca_temp.Nodes_N_FundingRound'
    definition = ['FundingRoundID CHAR(64)', 'FundingRoundDate DATE', 'FundingAmount_USD FLOAT',
                  'FundingAmountPerInvestor_USD FLOAT', 'PRIMARY KEY FundingRoundID (FundingRoundID)']
    db.drop_create_insert_table(table_name, definition, frs)

    log(f'    {sw.delta():.3f}s', color='green')

    ############################################################

    log('Inserting investors into database...')

    # Drop, recreate table and fill with df
    table_name = 'ca_temp.Nodes_N_Investor'
    definition = ['InvestorID CHAR(64)', 'InvestorType CHAR(32)', 'PRIMARY KEY InvestorID (InvestorID)']
    df = investors_frs[['InvestorID', 'InvestorType']].drop_duplicates()
    db.drop_create_insert_table(table_name, definition, df)

    log(f'    {sw.delta():.3f}s', color='green')

    ############################################################

    log('Inserting investors-frs edges into database...')

    # Drop, recreate table and fill with df
    table_name = 'ca_temp.Edges_N_Investor_N_FundingRound'
    definition = ['InvestorID CHAR(64)', 'FundingRoundID CHAR(64)', 'KEY InvestorID (InvestorID)',
                  'KEY FundingRoundID (FundingRoundID)']
    df = investors_frs[['InvestorID', 'FundingRoundID']]
    db.drop_create_insert_table(table_name, definition, df)

    log(f'    {sw.delta():.3f}s', color='green')

    ############################################################

    log('Computing investor pairs...')

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
    investors_investors = pd.concat([first_half, second_half]).reset_index(drop=True)

    log(f'    {sw.delta():.3f}s', color='green')

    ############################################################

    log('Inserting investor-investor edges into database...')

    # Drop, recreate table and fill with df
    table_name = 'ca_temp.Edges_N_Investor_N_Investor'
    definition = ['SourceInvestorID CHAR(64)', 'TargetInvestorID CHAR(64)', 'KEY SourceInvestorID (SourceInvestorID)', 'KEY TargetInvestorID (TargetInvestorID)']
    df = investors_investors[['SourceInvestorID', 'TargetInvestorID']].drop_duplicates()
    db.drop_create_insert_table(table_name, definition, df)

    log(f'    {sw.delta():.3f}s', color='green')

    ############################################################

    log('Retrieving investees...')

    # Fetch investees from database
    table_name = 'graph.Edges_N_Organisation_N_FundingRound'
    fields = ['FundingRoundID', 'OrganisationID']
    conditions = {'Action': 'Raised from', 'FundingRoundID': fr_ids}
    frs_investees = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=['FundingRoundID', 'InvesteeID'])
    investee_ids = list(frs_investees['InvesteeID'])

    log(f'    {sw.delta():.3f}s', color='green')

    ############################################################

    log('Inserting investees into database...')

    # Drop, recreate table and fill with df
    table_name = 'ca_temp.Nodes_N_Investee'
    definition = ['InvesteeID CHAR(64)', 'PRIMARY KEY InvesteeID (InvesteeID)']
    df = frs_investees[['InvesteeID']].drop_duplicates()
    db.drop_create_insert_table(table_name, definition, df)

    log(f'    {sw.delta():.3f}s', color='green')

    ############################################################

    log('Inserting frs-investees edges into database...')

    # Drop, recreate table and fill with df
    table_name = 'ca_temp.Edges_N_FundingRound_N_Investee'
    definition = ['FundingRoundID CHAR(64)', 'InvesteeID CHAR(64)', 'KEY FundingRoundID (FundingRoundID)', 'KEY InvesteeID (InvesteeID)']
    df = frs_investees[['FundingRoundID', 'InvesteeID']]
    db.drop_create_insert_table(table_name, definition, df)

    log(f'    {sw.delta():.3f}s', color='green')

    ############################################################

    log('Inserting investors-investees edges into database...')

    # Drop, recreate table and fill with df
    table_name = 'ca_temp.Edges_N_Investor_N_Investee'
    definition = ['InvestorID CHAR(64)', 'InvesteeID CHAR(64)', 'KEY InvestorID (InvestorID)', 'KEY InvesteeID (InvesteeID)']
    df = pd.merge(investors_frs, frs_investees, how='inner', on='FundingRoundID')[['InvestorID', 'InvesteeID']].drop_duplicates()
    db.drop_create_insert_table(table_name, definition, df)

    log(f'    {sw.delta():.3f}s', color='green')

    ############################################################

    log('Retrieving concepts...')

    # Fetch concepts from database
    table_name = 'graph.Edges_N_Organisation_N_Concept'
    fields = ['OrganisationID', 'PageID']
    conditions = {'OrganisationID': investee_ids}
    investees_concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=['InvesteeID', 'PageID'])

    log(f'    {sw.delta():.3f}s', color='green')

    ############################################################

    log('Inserting investees-concepts edges into database...')

    # Drop, recreate table and fill with df
    table_name = 'ca_temp.Edges_N_Investee_N_Concept'
    definition = ['InvesteeID CHAR(64)', 'PageID INT UNSIGNED', 'KEY InvesteeID (InvesteeID)', 'KEY PageID (PageID)']
    db.drop_create_insert_table(table_name, definition, investees_concepts)

    log(f'    {sw.delta():.3f}s', color='green')

    ############################################################

    log('Inserting frs-concepts edges into database...')

    # Drop, recreate table and fill with df
    table_name = 'ca_temp.Edges_N_FundingRound_N_Concept'
    definition = ['FundingRoundID CHAR(64)', 'PageID INT UNSIGNED', 'KEY FundingRoundID (FundingRoundID)', 'KEY PageID (PageID)']
    df = pd.merge(frs_investees, investees_concepts, how='inner', on='InvesteeID')[['FundingRoundID', 'PageID']].drop_duplicates()
    db.drop_create_insert_table(table_name, definition, df)

    log(f'    {sw.delta():.3f}s', color='green')

    ############################################################

    # Merge dataframes investors-frs-investees-concepts to have all information together
    df = investors_frs \
        .merge(frs_investees, how='inner', on='FundingRoundID') \
        .merge(investees_concepts, how='inner', on='InvesteeID') \
        .merge(frs, how='inner', on='FundingRoundID')

    ############################################################

    log('Computing investors-concepts history...')

    investors_concepts_history = compute_investors_concepts_history(df)

    log(f'    {sw.delta():.3f}s', color='green')

    ############################################################

    log('Inserting investors-concepts edges into database...')

    # Drop, recreate table and fill with df
    table_name = 'ca_temp.Edges_N_Investor_N_Concept'
    definition = [
        'InvestorID CHAR(64)',
        'PageID INT UNSIGNED',
        'ScoreLinNInv FLOAT',
        'ScoreLinAmount FLOAT',
        'ScoreLinNInvNorm FLOAT',
        'ScoreLinAmountNorm FLOAT',
        'ScoreQuadNInv FLOAT',
        'ScoreQuadAmount FLOAT',
        'ScoreQuadNInvNorm FLOAT',
        'ScoreQuadAmountNorm FLOAT',
        'ScoreConstNInv FLOAT',
        'ScoreConstAmount FLOAT',
        'ScoreConstNInvNorm FLOAT',
        'ScoreConstAmountNorm FLOAT',
        'KEY InvestorID (InvestorID)',
        'KEY PageID (PageID)'
    ]
    df = compute_scores_investors_concepts(df, investors_concepts_history)
    db.drop_create_insert_table(table_name, definition, df)

    log(f'    {sw.delta():.3f}s', color='green')

    ############################################################

    sw.report()


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    main()
