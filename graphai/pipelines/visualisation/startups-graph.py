import pandas as pd
import networkx as nx

from db_cache_manager.db import DB

from graphai.core.interfaces.config import config
from graphai.core.utils.breadcrumb import Breadcrumb
from graphai.pipelines.investment.concept_configuration import normalise


def print_summary(startups, concepts, startups_concepts, concepts_concepts):
    n_s = len(startups)
    n_c = len(concepts)
    n_sc = len(startups_concepts)
    n_cc = len(concepts_concepts)

    print(f'Number of startups: {n_s}')
    print(f'Number of concepts: {n_c}')
    print(f'Number of startup-concept edges: {n_sc}')
    print(f'Number of concept-concept edges: {n_cc}')


def create_startups_graph():

    # Initialize breadcrumb to log and keep track of time
    bc = Breadcrumb()

    # Instantiate db interface to communicate with database
    db = DB(config['database'])

    ############################################################

    seed_startup_ids = ['es-aqua-tech', 'es-bloom-biorenewables', 'es-bluewatt-engineering', 'es-comppair',
                        'es-ecointesys', 'es-enairys-powertech-sa', 'es-energie-solaire', 'es-enviroscopy',
                        'es-epiqr-renovation', 'es-g24-innovations', 'es-g2e-glass2energy', 'es-gaiasens-technologies',
                        'es-grz-technologies', 'es-insolight', 'es-kaemco', 'es-power-vision-engineering',
                        'es-solaronix', 'es-swiss-inso', 'es-twentygreen', 'es-urbio']

    gexf_filename = 'startups.gexf'

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

    ############################################################

    bc.log('Filter edges with non-existing nodes and isolated nodes...')

    startups_concepts = pd.merge(startups_concepts, startups[['EPFLStartupID']], how='inner', on='EPFLStartupID')
    startups = pd.merge(startups, startups_concepts[['EPFLStartupID']].drop_duplicates(), how='inner', on='EPFLStartupID')

    ############################################################

    bc.log('Recomputing startup-concept edge score...')

    # Divide score by the frequency of each concept
    startups_concepts = pd.merge(
        startups_concepts,
        startups_concepts.groupby(by='PageID').aggregate(PageStartupCount=('EPFLStartupID', 'count')).reset_index(),
        how='left',
        on='PageID'
    )
    startups_concepts['Score'] = startups_concepts['Score'] / startups_concepts['PageStartupCount']
    startups_concepts = startups_concepts[['EPFLStartupID', 'PageID', 'Score']]

    ############################################################

    bc.log('Filtering startup-concept edges based on score...')

    # Keep at most 10 concepts per startup
    startups_concepts = startups_concepts.groupby(by='EPFLStartupID').head(10).reset_index(drop=True)

    # Keep only edges with high score
    proportion = 0.1    # keep 10% of all edges
    startups_concepts = startups_concepts.sort_values(by='Score', ascending=False)
    startups_concepts = startups_concepts.head(int(proportion * len(startups_concepts)))
    startups_concepts = startups_concepts.reset_index(drop=True)

    ############################################################

    bc.log('Normalising startup-concept edge scores...')

    # Normalise scores
    startups_concepts = normalise(startups_concepts)

    ############################################################

    bc.log('Restricting data to filtered subset...')

    startups = pd.merge(startups, startups_concepts[['EPFLStartupID']].drop_duplicates(), how='inner', on='EPFLStartupID')

    ############################################################

    bc.log('Printing summary of resulting startups and concepts nodes and edges...')

    print_summary(startups, concepts, startups_concepts, concepts_concepts)

    ############################################################

    bc.log('Preparing nodes and edges DataFrames...')

    # Nodes
    startups['Type'] = 'Startup'
    concepts['Type'] = 'Concept'

    startups = startups.rename(columns={'EPFLStartupID': 'ID', 'StartupName': 'Name'})
    concepts = concepts.rename(columns={'PageID': 'ID', 'PageTitle': 'Name'})

    nodes = pd.concat([startups, concepts]).reset_index(drop=True)

    # Edges
    startups_concepts['Type'] = 'Startup-Concept'
    concepts_concepts['Type'] = 'Concept-Concept'

    startups_concepts = startups_concepts.rename(columns={'EPFLStartupID': 'SourceID', 'PageID': 'TargetID'})
    concepts_concepts = concepts_concepts.rename(columns={'SourcePageID': 'SourceID', 'TargetPageID': 'TargetID', 'NormalisedScore': 'Score'})

    edges = pd.concat([startups_concepts, concepts_concepts]).reset_index(drop=True)

    ############################################################

    bc.log('Removing nodes without edges and edges with non-existent nodes...')

    edges = pd.merge(edges, nodes[['ID']].rename(columns={'ID': 'SourceID'}), how='inner', on='SourceID')
    edges = pd.merge(edges, nodes[['ID']].rename(columns={'ID': 'TargetID'}), how='inner', on='TargetID')

    nodes = pd.merge(
        nodes,
        pd.concat([
            edges[['SourceID']].rename(columns={'SourceID': 'ID'}),
            edges[['TargetID']].rename(columns={'TargetID': 'ID'})
        ]).drop_duplicates(),
        how='inner',
        on='ID'
    )

    ############################################################

    bc.log('Preparing node and edge lists...')

    node_list = [(row['ID'], {'name': row['Name'], 'type': row['Type']}) for row in nodes.to_dict(orient='records')]
    edge_list = [(row['SourceID'], row['TargetID'], {'weight': row['Score']}) for row in edges.to_dict(orient='records')]

    ############################################################

    bc.log('Building graph...')

    G = nx.Graph()

    G.add_nodes_from(node_list)
    G.add_edges_from(edge_list)

    ############################################################

    bc.log(f'Exporting in gexf format at {gexf_filename}...')

    nx.write_gexf(G, gexf_filename)

    ############################################################

    bc.report()


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    create_startups_graph()
