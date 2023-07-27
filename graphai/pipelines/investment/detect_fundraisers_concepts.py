import requests

import pandas as pd

from graphai.core.interfaces.db import DB

from graphai.core.utils.breadcrumb import Breadcrumb

from graphai.pipelines.investment.concept_configuration import normalise


def detect_fundraisers_concepts(params):

    # Initialize breadcrumb to log and keep track of time
    bc = Breadcrumb()

    # Instantiate db interface to communicate with database
    db = DB()

    # Define url of endpoint
    WIKIFY_URL = 'http://localhost:28800/text/wikify'

    ############################################################

    bc.log('Fetching fundraiser descriptions...')

    # Fetch manual fundraiser-concept edges
    table_name = 'aitor.Edges_N_Fundraiser_N_Concept'
    fields = ['FundraiserID', 'PageID']
    fundraisers_concepts = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)
    fundraiser_ids = list(fundraisers_concepts['FundraiserID'].drop_duplicates())

    # Fetch fundraiser descriptions
    table_name = 'graph.Nodes_N_Organisation'
    fields = ['OrganisationID', 'ShortDescription', 'Description']
    conditions = {'OrganisationID': fundraiser_ids}
    fundraisers = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=['FundraiserID', 'ShortDescription', 'Description'])
    fundraisers = fundraisers.fillna('')
    fundraisers['FullDescription'] = fundraisers['ShortDescription'] + ' ' + fundraisers['Description']
    fundraisers = fundraisers[['FundraiserID', 'FullDescription']]

    ############################################################

    bc.log(f'Detecting concepts through wikify ({len(fundraisers)} fundraisers, eta ~{(len(fundraisers) * 82 / 100) / 3600:.2f} h)...')

    fundraisers_concepts_detected = None

    for row in fundraisers.to_dict(orient='records'):
        data = {'raw_text': row['FullDescription']}
        results_list = requests.post(url=WIKIFY_URL, json=data).json()

        if len(results_list) == 0:
            continue

        results = pd.DataFrame(results_list)
        results['FundraiserID'] = row['FundraiserID']

        fundraisers_concepts_detected = pd.concat([fundraisers_concepts_detected, results])

    fundraisers_concepts_detected = fundraisers_concepts_detected.rename(columns={'MixedScore': 'Score'})
    fundraisers_concepts_detected = fundraisers_concepts_detected[['FundraiserID', 'PageID', 'Score']]

    ############################################################

    bc.log('Normalising fundraiser-concept detected edge scores...')

    # Normalise scores so that all fundraisers have a configuration with norm 1
    fundraisers_concepts_detected = normalise(fundraisers_concepts_detected)

    ############################################################

    bc.log('Inserting fundraiser-concept detected edges into database...')

    table_name = 'aitor.Edges_N_Fundraiser_N_Concept_T_AutoNLP'
    definition = [
        'FundraiserID CHAR(64)',
        'PageID INT UNSIGNED',
        'Score FLOAT',
        'KEY FundraiserID (FundraiserID)',
        'KEY PageID (PageID)'
    ]
    db.drop_create_insert_table(table_name, definition, fundraisers_concepts_detected)

    ############################################################

    bc.report()


if __name__ == '__main__':
    import graphai.pipelines.investment.parameters as params

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    detect_fundraisers_concepts(params)
