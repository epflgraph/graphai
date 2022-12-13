import pandas as pd

from interfaces.db import DB
from utils.breadcrumb import Breadcrumb

from investment.concept_configuration import compute_affinities


def compute_investors_units_2():

    # Initialize breadcrumb to log and keep track of time
    bc = Breadcrumb()

    # Instantiate db interface to communicate with database
    db = DB()

    ############################################################

    bc.log('Fetching unit-concept edges from database...')

    table_name = 'graph.Edges_N_Unit_N_Concept_T_Research'
    fields = ['UnitID', 'PageID', 'Score']
    units_concepts = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)

    # Renormalise units-concepts scores to add to 1
    units_concepts = pd.merge(
        units_concepts,
        units_concepts.groupby('UnitID').aggregate(SumScore=('Score', 'sum')).reset_index(),
        how='left',
        on='UnitID'
    )
    units_concepts['Score'] = units_concepts['Score'] / units_concepts['SumScore']
    units_concepts = units_concepts.drop(columns='SumScore')

    ############################################################

    bc.log('Fetching investor-concept Jaccard edges from database...')

    table_name = 'aitor.Edges_N_Investor_N_Concept_T_Jaccard'
    fields = ['InvestorID', 'PageID', 'Jaccard_000']
    investors_concepts_jaccard = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)

    ############################################################

    bc.log('Deriving investors-units edges from investors-concepts and units-concepts...')

    investors_units = pd.merge(investors_concepts_jaccard, units_concepts, how='inner', on='PageID')
    investors_units['Score'] = investors_units['Score'] * investors_units['Jaccard_000']
    investors_units = investors_units.groupby(by=['InvestorID', 'UnitID']).aggregate(Score=('Score', 'sum')).reset_index()
    investors_units = investors_units.sort_values(by=['UnitID', 'Score'], ascending=[True, False])
    investors_units = investors_units.groupby(by='UnitID').head(5)
    investor_ids = list(investors_units['InvestorID'].drop_duplicates())

    ############################################################

    bc.log('Fetching investor-concept yearly edges from database...')

    table_name = 'aitor.Edges_N_Investor_N_Concept_T_Years'
    fields = ['InvestorID', 'PageID', 'Year', 'CountAmount']
    conditions = {'InvestorID': investor_ids}
    investors_concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

    # Normalise investor-concepts scores
    investors_concepts = pd.merge(
        investors_concepts,
        investors_concepts.groupby(by=['InvestorID', 'Year']).aggregate(MaxScore=('CountAmount', 'max')).reset_index(),
        how='left',
        on=['InvestorID', 'Year']
    )
    investors_concepts['Score'] = investors_concepts['CountAmount'] / investors_concepts['MaxScore']
    investors_concepts = investors_concepts.drop(columns=['CountAmount', 'MaxScore'])

    ############################################################

    unit_concept_ids = list(units_concepts['PageID'].drop_duplicates())
    investor_concept_ids = list(investors_concepts['PageID'].drop_duplicates())
    concept_ids = list(set(unit_concept_ids + investor_concept_ids))

    ############################################################

    bc.log('Fetching concept-concept edges from database...')

    table_name = 'graph.Edges_N_Concept_N_Concept_T_GraphScore'
    fields = ['SourcePageID', 'TargetPageID', 'NormalisedScore']
    conditions = {'OR': {'SourcePageID': concept_ids, 'TargetPageID': concept_ids}}
    concepts_concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=['SourcePageID', 'TargetPageID', 'Score'])

    ############################################################

    bc.log('Adding year to investors-units...')

    investors_units = pd.merge(
        investors_concepts[['InvestorID', 'Year']].drop_duplicates(),
        investors_units,
        how='left',
        on='InvestorID'
    )

    ############################################################

    bc.log('Computing affinities investors-units...')

    investors_units = investors_units.drop(columns=['Score'])
    investors_units = compute_affinities(investors_concepts, units_concepts, investors_units, edges=concepts_concepts, mix_x=True, mix_y=True)

    ############################################################

    bc.log('Inserting investors-units edges into database...')

    table_name = 'aitor.Edges_N_Investor_N_Unit_T_2'
    definition = [
        'InvestorID CHAR(64)',
        'Year SMALLINT',
        'UnitID CHAR(32)',
        'Score FLOAT',
        'KEY InvestorID (InvestorID)',
        'KEY Year (Year)',
        'KEY UnitID (UnitID)'
    ]
    db.drop_create_insert_table(table_name, definition, investors_units)

    ############################################################

    bc.report()


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    compute_investors_units_2()
