import requests

import pandas as pd

from graphai.core.interfaces.db import DB
from graphai.core.interfaces.config_loader import load_db_config

from graphai.core.utils.breadcrumb import Breadcrumb

from graphai.pipelines.investment.concept_configuration import normalise


def detect_fundraisers_concepts(params):

    # Initialize breadcrumb to log and keep track of time
    bc = Breadcrumb()

    # Instantiate db interface to communicate with database
    db = DB(load_db_config())

    # Define url of endpoint
    WIKIFY_URL = 'http://localhost:28800/text/wikify'

    ############################################################

    bc.log('Fetching already detected fundraisers...')

    # Fetch already detected fundraisers
    try:
        table_name = f'aitor.{params.prefix}_Edges_N_Fundraiser_N_Concept_T_AutoNLP'
        fields = ['FundraiserID', 'PageID', 'Score']
        detected_fundraisers_concepts = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)
        detected_fundraiser_ids = list(detected_fundraisers_concepts['FundraiserID'].drop_duplicates())
    except Exception:
        detected_fundraisers_concepts = None
        detected_fundraiser_ids = []

    print(f'Fetched {len(detected_fundraiser_ids)} already detected fundraisers')

    ############################################################

    bc.log('Fetching fundraiser descriptions...')

    # Fetch manual fundraiser-concept edges
    table_name = f'aitor.{params.prefix}_Edges_N_Fundraiser_N_Concept'
    fields = ['FundraiserID', 'PageID']
    fundraisers_concepts = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)

    print(f'Fetched {len(list(fundraisers_concepts["FundraiserID"].drop_duplicates()))} fundraisers')

    # Exclude already detected fundraisers
    fundraisers_concepts = fundraisers_concepts[~fundraisers_concepts['FundraiserID'].isin(detected_fundraiser_ids)].reset_index(drop=True)
    fundraiser_ids = list(fundraisers_concepts['FundraiserID'].drop_duplicates())

    print(f'After excluding already detected fundraisers, there are {len(fundraiser_ids)} fundraisers left')

    # Fetch fundraiser descriptions
    table_name = 'graph.Nodes_N_Organisation'
    fields = ['OrganisationID', 'ShortDescription', 'Description']
    columns = ['FundraiserID', 'ShortDescription', 'Description']
    fundraisers = db.find_or_split(table_name, fields, columns, 'OrganisationID', fundraiser_ids)

    print(f'There are {len(fundraisers)} fundraisers with description left')

    # Prepare descriptions
    fundraisers = fundraisers.fillna('')
    fundraisers['FullDescription'] = fundraisers['ShortDescription'] + ' ' + fundraisers['Description']
    fundraisers = fundraisers[['FundraiserID', 'FullDescription']]

    ############################################################

    bc.log(f'Detecting concepts through wikify ({len(fundraisers)} fundraisers, eta ~{(len(fundraisers) * 82 / 100) / 3600:.2f} h)...')

    for row in fundraisers.to_dict(orient='records'):
        data = {'raw_text': row['FullDescription']}
        results_list = requests.post(url=WIKIFY_URL, json=data).json()

        if len(results_list) == 0:
            continue

        results = pd.DataFrame(results_list)
        results['FundraiserID'] = row['FundraiserID']

        results = results.rename(columns={'MixedScore': 'Score'})
        detected_fundraisers_concepts = pd.concat([detected_fundraisers_concepts, results])

    detected_fundraisers_concepts = detected_fundraisers_concepts[['FundraiserID', 'PageID', 'Score']]

    ############################################################

    bc.log('Normalising fundraiser-concept detected edge scores...')

    # Normalise scores so that all fundraisers have a configuration with norm 1
    detected_fundraisers_concepts = normalise(detected_fundraisers_concepts)

    ############################################################

    bc.log('Inserting fundraiser-concept detected edges into database...')

    table_name = f'aitor.{params.prefix}_Edges_N_Fundraiser_N_Concept_T_AutoNLP'
    definition = [
        'FundraiserID CHAR(64)',
        'PageID INT UNSIGNED',
        'Score FLOAT',
        'KEY FundraiserID (FundraiserID)',
        'KEY PageID (PageID)'
    ]
    db.drop_create_insert_table(table_name, definition, detected_fundraisers_concepts)

    ############################################################

    bc.report()


if __name__ == '__main__':
    import graphai.pipelines.investment.parameters as params

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    detect_fundraisers_concepts(params)
