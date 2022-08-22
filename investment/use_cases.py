import pandas as pd
import numpy as np

from interfaces.db import DB


def get_concepts_top_investors(concept_ids, n_investors=3):
    # Instantiate db interface to communicate with database
    db = DB()

    # Retrieve investor-concept edges
    table_name = 'ca_temp.Edges_N_Investor_N_Concept'
    fields = ['InvestorID', 'PageID']
    conditions = {'PageID': concept_ids}
    investors_concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

    # Extract list of investor ids
    investor_ids = list(investors_concepts['InvestorID'].drop_duplicates())

    # Build investor-concept history for each pair
    history = get_investor_concept_history(investor_ids, concept_ids)

    # Keep top `n_investors` for each concept
    top_investors = {}
    for concept_id in history:
        del history[concept_id]['total']
        concept_top_investor_ids = [investor_id for investor_id, _ in sorted(history[concept_id].items(), key=lambda item: item[1]['total'], reverse=True)]
        top_investors[concept_id] = {investor_id: history[concept_id][investor_id]['total'] for investor_id in concept_top_investor_ids[:n_investors]}

    return top_investors


def get_investor_concept_history(investor_ids, concept_ids):
    # Instantiate db interface to communicate with database
    db = DB()

    ##################
    # INVESTOR -> FR #
    ##################

    # Retrieve investor-fr edges
    table_name = 'ca_temp.Edges_N_Investor_N_FundingRound'
    fields = ['InvestorID', 'FundingRoundID']
    conditions = {'InvestorID': investor_ids}
    investors_frs = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

    # Extract list of funding round ids
    investor_fr_ids = list(investors_frs['FundingRoundID'])

    #############################
    # CONCEPT -> INVESTEE -> FR #
    #############################

    # Retrieve investee-concept edges
    table_name = 'ca_temp.Edges_N_Investee_N_Concept'
    fields = ['InvesteeID', 'PageID']
    conditions = {'PageID': concept_ids}
    investees_concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=['InvesteeID', 'PageID'])

    # Extract list of investee ids
    investee_ids = list(investees_concepts['InvesteeID'])

    # Retrieve fr-investee edges
    table_name = 'ca_temp.Edges_N_FundingRound_N_Investee'
    fields = ['FundingRoundID', 'InvesteeID']
    conditions = {'InvesteeID': investee_ids}
    frs_investees = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

    # Extract list of funding round ids
    concept_fr_ids = list(frs_investees['FundingRoundID'])

    ################
    # INTERSECTION #
    ################

    # Intersect funding round ids coming from investors and from concepts
    fr_ids = [fr_id for fr_id in investor_fr_ids if fr_id in concept_fr_ids]

    # Retrieve funding rounds in time window
    table_name = 'graph.Nodes_N_FundingRound'
    fields = ['FundingRoundID', 'FundingRoundDate', 'FundingAmount_USD / CB_InvestorCount']
    conditions = {'FundingRoundID': fr_ids}
    frs = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=['FundingRoundID', 'FundingRoundDate', 'FundingAmountPerInvestor_USD'])

    ###########
    # HISTORY #
    ###########

    # Sort funding rounds by date
    frs = frs.sort_values(by='FundingRoundDate').reset_index(drop=True)

    # Merge dataframes investors-frs-investees-concepts to have all information together
    df = investors_frs \
        .merge(frs_investees, how='inner', on='FundingRoundID') \
        .merge(investees_concepts, how='inner', on='InvesteeID') \
        .merge(frs, how='inner', on='FundingRoundID')

    # Compute investor-concept history for each pair
    history = {}
    for concept_id in concept_ids:
        history[concept_id] = {}

        for investor_id in investor_ids:
            history[concept_id][investor_id] = {}

            # Restrict df to current concept and investor
            concept_investor_df = df[(df['InvestorID'] == investor_id) & (df['PageID'] == concept_id)]

            # if len(concept_investor_df) == 0:
            #     continue

            # Iterate over all dates and gather all investments made by current investor in current concept each day.
            for fr_date in concept_investor_df['FundingRoundDate'].drop_duplicates():
                history[concept_id][investor_id][str(fr_date)] = {}

                # Restrict df to current concept, investor and date
                concept_investor_date_df = concept_investor_df[concept_investor_df['FundingRoundDate'] == fr_date]

                # Iterate over frs and gather all investments made by current investor in current concept on given day.
                date_total = 0
                for fr_id in concept_investor_date_df['FundingRoundID']:
                    # Restrict df to current concept, investor, date and funding round
                    concept_investor_date_fr_df = concept_investor_date_df[concept_investor_date_df['FundingRoundID'] == fr_id]

                    # There should be exactly one row for each combination of parameters
                    assert len(concept_investor_date_fr_df) == 1, f'There should be exactly one row for investor {investor_id} and funding round {fr_id}, {len(concept_investor_date_fr_df)} found.'

                    # Extract amount and build history
                    amount = concept_investor_date_fr_df['FundingAmountPerInvestor_USD'].iloc[0]
                    history[concept_id][investor_id][str(fr_date)][fr_id] = {'amount': amount}
                    date_total += np.nan_to_num(amount)

                # Update total investment amount for the given concept, investor and date
                history[concept_id][investor_id][str(fr_date)]['total'] = date_total

            # Update total investment amount for the given concept and investor
            history[concept_id][investor_id]['total'] = sum(
                [history[concept_id][investor_id][d]['total'] for d in history[concept_id][investor_id]]
            )

        # Update total investment amount for the given concept
        #   WATCH OUT! This total investment amount only considers investments for which
        #   we know the investors, and hence will not include amounts coming from investments
        #   for which we know the amount but not the investors.
        history[concept_id]['total'] = sum(
            [history[concept_id][investor_id]['total'] for investor_id in history[concept_id]]
        )

    return history


def main():
    from utils.time.stopwatch import Stopwatch
    sw = Stopwatch()

    pd.set_option('display.max_rows', 10)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    get_concepts_top_investors([1164, 28249265])

    sw.report()


if __name__ == '__main__':
    main()








