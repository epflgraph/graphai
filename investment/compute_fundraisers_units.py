import pandas as pd
import numpy as np

from interfaces.db import DB

from utils.breadcrumb import Breadcrumb

from compute_investors_units import compute_affinities


def compute_fundraisers_units():

    # Initialize breadcrumb to log and keep track of time
    bc = Breadcrumb()

    # Instantiate db interface to communicate with database
    db = DB()

    ############################################################

    bc.log('Fetching unit-concept edges from database...')

    table_name = 'graph.Edges_N_Unit_N_Concept_T_Research'
    fields = ['UnitID', 'PageID', 'Score']
    units_concepts = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)

    # Normalise scores so that they add up to one per unit
    units_concepts = pd.merge(
        units_concepts,
        units_concepts.groupby('UnitID').aggregate(SumScore=('Score', 'sum')).reset_index(),
        how='left',
        on='UnitID'
    )
    units_concepts['Score'] = units_concepts['Score'] / units_concepts['SumScore']
    units_concepts = units_concepts.drop(columns='SumScore')

    ############################################################

    bc.log('Fetching fundraiser-concept edges from database...')

    table_name = 'ca_temp.Edges_N_Fundraiser_N_Concept'
    fields = ['FundraiserID', 'PageID']
    fundraisers_concepts = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)

    # Add number of pages per fundraiser
    fundraisers_concepts = pd.merge(
        fundraisers_concepts,
        fundraisers_concepts.groupby(by='FundraiserID').aggregate(FundraiserPageCount=('PageID', 'count')).reset_index(),
        how='left',
        on='FundraiserID'
    )

    # Add number of fundraisers per page
    fundraisers_concepts = pd.merge(
        fundraisers_concepts,
        fundraisers_concepts.groupby(by='PageID').aggregate(PageFundraiserCount=('FundraiserID', 'count')).reset_index(),
        how='left',
        on='PageID'
    )

    # Compute first version of score
    fundraisers_concepts['Score'] = 1 / (fundraisers_concepts['FundraiserPageCount'] * np.log(1 + fundraisers_concepts['PageFundraiserCount']))

    # Normalise scores so that they add up to one per fundraiser
    fundraisers_concepts = pd.merge(
        fundraisers_concepts,
        fundraisers_concepts.groupby(by='FundraiserID').aggregate(FundraiserScoreSum=('Score', 'sum')).reset_index(),
        how='left',
        on='FundraiserID'
    )
    fundraisers_concepts['Score'] = fundraisers_concepts['Score'] / fundraisers_concepts['FundraiserScoreSum']

    # Extract only relevant columns
    fundraisers_concepts = fundraisers_concepts[['FundraiserID', 'PageID', 'Score']]

    ############################################################

    unit_concept_ids = list(units_concepts['PageID'].drop_duplicates())
    fundraisers_concept_ids = list(fundraisers_concepts['PageID'].drop_duplicates())
    concept_ids = list(set(unit_concept_ids + fundraisers_concept_ids))

    ############################################################

    bc.log('Fetching concept-concept edges from database...')

    table_name = 'graph.Edges_N_Concept_N_Concept_T_GraphScore'
    fields = ['SourcePageID', 'TargetPageID', 'NormalisedScore']
    conditions = {'OR': {'SourcePageID': concept_ids, 'TargetPageID': concept_ids}}
    concepts_concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=['SourcePageID', 'TargetPageID', 'Score'])

    ############################################################

    bc.log('Computing affinities fundraisers-units...')

    # We compute affinity fundraiser-unit for pairs sharing at least one concept with non-negligible unit score
    fundraisers_units = pd.merge(
        fundraisers_concepts,
        units_concepts.sort_values(by='Score', ascending=False).head(100000),
        how='inner',
        on='PageID'
    )
    fundraisers_units = fundraisers_units[['FundraiserID', 'UnitID']].drop_duplicates()
    fundraisers_units = compute_affinities(fundraisers_concepts, units_concepts, fundraisers_units, edges=concepts_concepts, mix_x=True)

    ############################################################

    bc.log('Inserting fundraiser-unit edges into database...')

    table_name = 'ca_temp.Edges_N_Fundraiser_N_Unit'
    definition = [
        'FundraiserID CHAR(64)',
        'UnitID CHAR(32)',
        'Score FLOAT',
        'KEY FundraiserID (FundraiserID)',
        'KEY UnitID (UnitID)'
    ]
    db.drop_create_insert_table(table_name, definition, fundraisers_units)

    ############################################################

    bc.report()


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    compute_fundraisers_units()
