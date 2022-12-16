import pandas as pd
import networkx as nx

from interfaces.db import DB
from utils.breadcrumb import Breadcrumb

from investment.concept_configuration import normalise


def create_startups_graph():

    # Initialize breadcrumb to log and keep track of time
    bc = Breadcrumb()

    # Instantiate db interface to communicate with database
    db = DB()

    ############################################################

    seed_startup_ids = ['es-aqua-tech', 'es-bloom-biorenewables', 'es-bluewatt-engineering', 'es-comppair',
                        'es-ecointesys', 'es-enairys-powertech-sa', 'es-energie-solaire', 'es-enviroscopy',
                        'es-epiqr-renovation', 'es-g24-innovations', 'es-g2e-glass2energy', 'es-gaiasens-technologies',
                        'es-grz-technologies', 'es-insolight', 'es-kaemco', 'es-power-vision-engineering',
                        'es-solaronix', 'es-swiss-inso', 'es-twentygreen', 'es-urbio']

    ############################################################

    bc.log('Fetching startup nodes from database...')
    table_name = 'graph_piper.Nodes_N_EPFLStartup'
    fields = ['EPFLStartupID', 'StartupName']
    conditions = {'Status': 'Private'}
    startups = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

    ############################################################

    bc.log('Fetching startup-founder edges from database...')
    table_name = 'graph_piper.Edges_N_EPFLStartup_N_Person_T_Founder'
    fields = ['EPFLStartupID', 'SCIPER']
    startups_founders = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)
    founder_ids = list(startups_founders['SCIPER'].drop_duplicates())

    ############################################################

    bc.log('Fetching startup-professor edges from database...')
    table_name = 'graph_piper.Edges_N_EPFLStartup_N_Person_T_Professor'
    fields = ['EPFLStartupID', 'SCIPER']
    startups_professors = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)
    professor_ids = list(startups_professors['SCIPER'].drop_duplicates())

    ############################################################

    bc.log('Fetching person-concept edges from database...')
    table_name = 'graph_piper.Edges_N_Person_N_Concept_T_Research'
    fields = ['SCIPER', 'PageID', 'Score']
    conditions = {'SCIPER': founder_ids + professor_ids}
    people_concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

    ############################################################

    bc.log('Merging to obtain startup-concept edges...')
    startups_founders_concepts = pd.merge(startups_founders, people_concepts, how='inner', on='SCIPER')
    startups_professors_concepts = pd.merge(startups_professors, people_concepts, how='inner', on='SCIPER')

    # We give priority to the founders' concepts and use the professors' concepts as fallback
    startup_with_founder_ids = list(startups_founders_concepts['EPFLStartupID'].drop_duplicates())
    startups_professors_concepts = startups_professors_concepts[~startups_professors_concepts['EPFLStartupID'].isin(startup_with_founder_ids)]

    # Combine both in one DataFrame
    startups_concepts = pd.concat([startups_founders_concepts, startups_professors_concepts], ignore_index=True)
    startups_concepts = startups_concepts.groupby(by=['EPFLStartupID', 'PageID']).aggregate({'Score': 'sum'}).reset_index()

    ############################################################

    bc.log('Extracting concepts related to seed startups...')

    # We use only founders' concepts as they are more likely to be close to the startup activity than the professor ones
    concept_ids = list(startups_founders_concepts.loc[startups_founders_concepts['EPFLStartupID'].isin(seed_startup_ids), 'PageID'].drop_duplicates())

    ############################################################

    bc.log('Fetching concept nodes from database...')

    table_name = 'graph_piper.Nodes_N_Concept'
    fields = ['PageID', 'PageTitle']
    conditions = {'PageID': concept_ids}
    concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

    ############################################################

    bc.log('Fetching concept-concept edges from database...')

    table_name = 'graph_piper.Edges_N_Concept_N_Concept_T_GraphScore'
    fields = ['SourcePageID', 'TargetPageID', 'NormalisedScore']
    conditions = {'SourcePageID': concept_ids, 'TargetPageID': concept_ids}
    concepts_concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

    ############################################################

    bc.log('Restricting data to concepts subset...')

    startups_concepts = startups_concepts[startups_concepts['PageID'].isin(concept_ids)].reset_index(drop=True)
    startup_ids = list(startups_concepts['EPFLStartupID'].drop_duplicates())
    startups = startups[startups['EPFLStartupID'].isin(startup_ids)].reset_index(drop=True)

    ############################################################

    # Nodes
    print(startups)
    print(concepts)

    # Edges
    print(startups_concepts)
    print(concepts_concepts)

    # TODO: concat nodes and edges and add type column

    ############################################################

    bc.report()


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    create_startups_graph()
