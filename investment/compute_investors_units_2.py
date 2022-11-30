import pandas as pd
import numpy as np

from interfaces.db import DB

from utils.breadcrumb import Breadcrumb

from compute_investors_units import compute_affinities


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

    bc.log('Fetching funding round-fundraiser edges from database...')

    table_name = 'ca_temp.Edges_N_FundingRound_N_Fundraiser'
    fields = ['FundingRoundID', 'FundraiserID']
    frs_fundraisers = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)

    # Merge to obtain funding round-concept edges
    frs_concepts = pd.merge(
        frs_fundraisers,
        fundraisers_concepts,
        how='inner',
        on='FundraiserID'
    )[['FundingRoundID', 'PageID', 'Score']]

    ############################################################

    bc.log('Fetching investor-funding round edges from database...')

    table_name = 'ca_temp.Edges_N_Investor_N_FundingRound'
    fields = ['InvestorID', 'FundingRoundID']
    investors_frs = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)

    # Merge to obtain investor-concept edges
    investors_concepts = pd.merge(
        investors_frs,
        frs_concepts,
        how='inner',
        on='FundingRoundID'
    )

    # Aggregate by averaging scores over all funding rounds of an investor
    investors_concepts = investors_concepts.groupby(by=['InvestorID', 'PageID']).aggregate({'Score': 'sum'}).reset_index()
    investors_concepts = pd.merge(
        investors_concepts,
        investors_frs.groupby(by='InvestorID').aggregate(InvestorFRCount=('FundingRoundID', 'count')).reset_index(),
        how='left',
        on='InvestorID'
    )
    investors_concepts['Score'] = investors_concepts['Score'] / investors_concepts['InvestorFRCount']
    investors_concepts = investors_concepts[['InvestorID', 'PageID', 'Score']]

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

    bc.log('Computing affinities investors-units...')

    # We process this by batches to avoid running into memory issues
    investor_ids = list(investors_concepts['InvestorID'].drop_duplicates())
    n = len(investor_ids)
    batch_size = 1000

    investors_units = None
    for batch_investor_ids in [investor_ids[i: i + batch_size] for i in range(0, n, batch_size)]:
        bc.log('New batch')
        batch_investors_concepts = investors_concepts[investors_concepts['InvestorID'].isin(batch_investor_ids)]

        # We compute affinity investor-unit for pairs sharing at least one concept with non-negligible unit score
        batch_investors_units = pd.merge(
            batch_investors_concepts,
            units_concepts,
            how='inner',
            on='PageID'
        )
        batch_investors_units = batch_investors_units[['InvestorID', 'UnitID']].drop_duplicates()
        batch_investors_units = compute_affinities(batch_investors_concepts, units_concepts, batch_investors_units, edges=concepts_concepts, mix_x=True)

        investors_units = pd.concat([investors_units, batch_investors_units]).reset_index(drop=True)

    ############################################################

    bc.log('Inserting investor-unit edges into database...')

    table_name = 'ca_temp.Edges_N_Investor_N_Unit_T_2'
    definition = [
        'InvestorID CHAR(64)',
        'UnitID CHAR(32)',
        'Score FLOAT',
        'KEY InvestorID (InvestorID)',
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
