import pandas as pd

from interfaces.db import DB


def get_concepts_top_investors(concept_ids):
    # Instantiate db interface to communicate with database
    db = DB()

    # Retrieve investor-concept edges
    table_name = 'ca_temp.Edges_N_Investor_N_Concept'
    fields = ['InvestorID', 'PageID']
    conditions = {'PageID': concept_ids}
    investors_concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

    # Extract list of investor ids
    investor_ids = list(investors_concepts['InvestorID'].drop_duplicates())

    investors_concepts_history = get_investor_concept_history(investor_ids, concept_ids)


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

    frs = frs.sort_values(by='FundingRoundDate').reset_index(drop=True)
    print(frs)

    aaa = investors_frs \
        .merge(frs_investees, how='inner', on='FundingRoundID') \
        .merge(investees_concepts, how='inner', on='InvesteeID') \
        .merge(frs, how='inner', on='FundingRoundID')
    print(aaa)

    for concept_id in concept_ids:
        for investor_id in investor_ids:
            print('#' * 50)
            print(concept_id, investor_id)

            bbb = aaa[(aaa['InvestorID'] == investor_id) & (aaa['PageID'] == concept_id)]
            print(bbb)


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








