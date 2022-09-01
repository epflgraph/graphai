import pandas as pd

from interfaces.db import DB

from utils.text.io import log
from utils.time.stopwatch import Stopwatch


def main():
    # Initialize stopwatch to keep track of time
    sw = Stopwatch()

    # Instantiate db interface to communicate with database
    db = DB()

    ############################################################
    # FETCH GRAPH FROM DATABASE                                #
    ############################################################

    log('Fetching investor-investor edges from database...')

    table_name = 'ca_temp.Edges_N_Investor_N_Investor'
    fields = ['SourceInvestorID', 'TargetInvestorID', 'ScoreQuadCount']
    df = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)

    # Duplicate as to have all edges and not only in one direction
    reversed_fields = ['TargetInvestorID', 'SourceInvestorID', 'ScoreQuadCount']
    reversed_df = pd.DataFrame(df[reversed_fields].values, columns=fields)
    investors_investors = pd.concat([df, reversed_df])

    # Extract list of investor ids
    investor_ids = list(investors_investors['SourceInvestorID'].drop_duplicates())

    log('investors_investors')
    log(investors_investors)

    log(f'    {sw.delta():.3f}s', color='green')

    ############################################################

    log('Fetching investor-concept edges from database...')

    table_name = 'ca_temp.Edges_N_Investor_N_Concept'
    fields = ['InvestorID', 'PageID', 'ScoreQuadCount']
    investors_concepts = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)

    # Extract list of concept ids
    concept_ids = list(investors_concepts['PageID'].drop_duplicates())

    log('investors_concepts')
    log(investors_concepts)

    log(f'    {sw.delta():.3f}s', color='green')

    ############################################################

    log('Fetching concept-concept edges from database...')

    table_name = 'graph.Edges_N_Concept_N_Concept_T_GraphScore'
    fields = ['SourcePageID', 'TargetPageID']
    conditions = {'SourcePageID': concept_ids, 'TargetPageID': concept_ids}
    concepts_concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

    log('concepts_concepts')
    log(concepts_concepts)

    log(f'    {sw.delta():.3f}s', color='green')

    ############################################################

    sw.report(laps=False)


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    main()
