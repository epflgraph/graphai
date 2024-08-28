import pandas as pd
import numpy as np

from db_cache_manager.db import DB

from graphai.core.common.config import config
from graphai.core.utils.breadcrumb import Breadcrumb
from graphai.pipelines.investment.concept_configuration import compute_affinities, normalise


def compute_fundraisers_units(params):

    # Initialize breadcrumb to log and keep track of time
    bc = Breadcrumb()

    # Instantiate db interface to communicate with database
    db = DB(config['database'])

    ############################################################

    bc.log('Fetching unit-concept edges from database...')

    table_name = 'graph.Edges_N_Unit_N_Concept_T_Research'
    fields = ['UnitID', 'PageID', 'Score']
    units_concepts = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)

    ############################################################

    bc.log('Normalising unit-concept edge scores...')

    # Normalise scores so that all units have a configuration with norm 1
    units_concepts = normalise(units_concepts)

    ############################################################

    bc.log('Fetching fundraiser-concept manual edges from database...')

    table_name = f'aitor.{params.prefix}_Edges_N_Fundraiser_N_Concept'
    fields = ['FundraiserID', 'PageID']
    fundraisers_concepts = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)

    # Add number of fundraisers per page
    fundraisers_concepts = pd.merge(
        fundraisers_concepts,
        fundraisers_concepts.groupby(by='PageID').aggregate(PageFundraiserCount=('FundraiserID', 'count')).reset_index(),
        how='left',
        on='PageID'
    )

    # Compute first version of score based on concept rarity among fundraisers
    fundraisers_concepts['Score'] = 1 / (1 + np.log(fundraisers_concepts['PageFundraiserCount']))

    # Keep only relevant columns
    fundraisers_concepts = fundraisers_concepts[['FundraiserID', 'PageID', 'Score']]

    ############################################################

    bc.log('Fetching fundraiser-concept NLP edges from database...')

    table_name = f'aitor.{params.prefix}_Edges_N_Fundraiser_N_Concept_T_AutoNLP'
    fields = ['FundraiserID', 'PageID', 'Score']
    fundraisers_concepts_nlp = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)

    ############################################################

    bc.log('Merging fundraiser-concept manual and NLP edges...')

    # Merge both manual and NLP edges in a single DataFrame
    fundraisers_concepts = pd.merge(fundraisers_concepts, fundraisers_concepts_nlp, how='outer', on=['FundraiserID', 'PageID'], suffixes=('Manual', 'NLP'))
    fundraisers_concepts = fundraisers_concepts.fillna(0)

    ############################################################

    bc.log('Combining fundraiser-concept manual and NLP edge scores...')

    # We do not want to combine both scores symetrically, since manual tagging is a priori more reliable.
    # However, we don't want to ignore the NLP pages neither.
    # We define the combined score as follows:
    #   Let base = max(ScoreManual, 0.9 * ScoreManual, 0.1 * ScoreNLP).
    #   Let S = base^(1 - ScoreNLP/2).
    #   If ScoreManual >= 0, then we define Score = S.
    #   Else,
    #       If S >= 0.1, then Score = S.
    #       Else, Score = 0.
    # We do this because we want the following properties:
    #   - Score >= ScoreManual always.
    #   - If ScoreManual = 0, then Score > 0 only for high values of ScoreNLP (greater than ~0.4816777...).
    #   - If ScoreNLP = 0, then Score = ScoreManual.
    #   - If ScoreNLP = 1, then we boost Score = sqrt(ScoreManual) (>= ScoreManual).
    #   - If ScoreNLP = 0.5, then Score is the geometric mean of ScoreManual and sqrt(ScoreManual).
    #   - In general, Score runs (multiplicatively) from ScoreManual to sqrt(ScoreManual) as ScoreNLP runs from 0 to 1.

    # Extract relevant columns
    score_manual = fundraisers_concepts['ScoreManual']
    score_nlp = fundraisers_concepts['ScoreNLP']
    score_weighted = 0.9 * score_manual + 0.1 * score_nlp

    # Compute combined scores
    base = pd.concat([score_manual, score_weighted], axis=1).max(axis=1)
    score = np.power(base, 1 - score_nlp / 2)
    fundraisers_concepts['Score'] = score.mask((score < 0.1) & (score_manual == 0), 0)

    # Keep only positive scores
    fundraisers_concepts = fundraisers_concepts[fundraisers_concepts['Score'] > 0].reset_index(drop=True)

    # Keep only relevant columns
    fundraisers_concepts = fundraisers_concepts[['FundraiserID', 'PageID', 'Score']]

    ############################################################

    bc.log('Normalising fundraiser-concept edge scores...')

    # Normalise scores so that all fundraisers have a configuration with norm 1
    fundraisers_concepts = normalise(fundraisers_concepts)

    ############################################################

    unit_concept_ids = list(units_concepts['PageID'].drop_duplicates())
    fundraiser_concept_ids = list(fundraisers_concepts['PageID'].drop_duplicates())
    concept_ids = list(set(unit_concept_ids + fundraiser_concept_ids))

    ############################################################

    bc.log('Fetching concept-concept edges from database...')

    table_name = 'graph.Edges_N_Concept_N_Concept_T_GraphScore'
    fields = ['SourcePageID', 'TargetPageID', 'NormalisedScore']
    conditions = {'OR': {'SourcePageID': concept_ids, 'TargetPageID': concept_ids}}
    concepts_concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=['SourcePageID', 'TargetPageID', 'Score'])

    ############################################################

    bc.log('Computing affinities fundraisers-units...')

    # We process this by batches to avoid running into memory issues
    fundraiser_ids = list(fundraisers_concepts['FundraiserID'].drop_duplicates())
    n = len(fundraiser_ids)
    batch_size = 1000
    n_batches = np.ceil(n / batch_size).astype(int)

    fundraisers_units = None
    for i in range(0, n_batches):
        bc.log(f'New batch ({i + 1}/{n_batches})')
        batch_fundraiser_ids = fundraiser_ids[i * batch_size: (i + 1) * batch_size]
        batch_fundraisers_concepts = fundraisers_concepts[fundraisers_concepts['FundraiserID'].isin(batch_fundraiser_ids)]

        # We compute affinity fundraiser-unit for pairs sharing at least one concept with non-negligible unit score
        batch_fundraisers_units = pd.merge(
            batch_fundraisers_concepts,
            units_concepts,
            how='inner',
            on='PageID'
        )
        batch_fundraisers_units = batch_fundraisers_units[['FundraiserID', 'UnitID']].drop_duplicates()
        batch_fundraisers_units = compute_affinities(batch_fundraisers_concepts, units_concepts, batch_fundraisers_units, edges=concepts_concepts, mix_x=True)

        fundraisers_units = pd.concat([fundraisers_units, batch_fundraisers_units]).reset_index(drop=True)

    ############################################################

    # Filter out zero scores
    fundraisers_units = fundraisers_units[fundraisers_units['Score'] > 0].reset_index(drop=True)

    ############################################################

    # Keep only the top 500 fundraisers per unit, in case there are too many
    fundraisers_units = fundraisers_units.sort_values(by='Score', ascending=False).groupby(by='UnitID').head(500).reset_index(drop=True)

    ############################################################

    bc.log('Normalising fundraiser-unit edge scores...')

    # Square scores to sum them
    fundraisers_units['SquaredScore'] = np.square(fundraisers_units['Score'])

    # Add squared norms to fundraisers_units DataFrame
    fundraisers_units = pd.merge(
        fundraisers_units,
        fundraisers_units.groupby(by='UnitID').aggregate(SquaredNorm=('SquaredScore', 'sum')).reset_index(),
        how='inner',
        on='UnitID'
    )

    # Compute the norm by taking the sqrt of the squared norm
    fundraisers_units['Norm'] = np.sqrt(fundraisers_units['SquaredNorm'])

    # Divide each score by the norm of the fundraiser configuration
    fundraisers_units['Score'] = fundraisers_units['Score'] / fundraisers_units['Norm']

    # Keep only relevant columns
    fundraisers_units = fundraisers_units[['FundraiserID', 'UnitID', 'Score']]

    ############################################################

    bc.log('Inserting fundraiser-unit edges into database...')

    table_name = f'aitor.{params.prefix}_Edges_N_Fundraiser_N_Unit'
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
    import graphai.pipelines.investment.parameters as params

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    compute_fundraisers_units(params)
