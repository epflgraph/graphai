import pandas as pd
import networkx as nx

from interfaces.db import DB
from utils.breadcrumb import Breadcrumb

from investment.concept_configuration import normalise


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
    db = DB()

    ############################################################

    seed_startup_ids = ['es-aqua-tech', 'es-bloom-biorenewables', 'es-bluewatt-engineering', 'es-comppair',
                        'es-ecointesys', 'es-enairys-powertech-sa', 'es-energie-solaire', 'es-enviroscopy',
                        'es-epiqr-renovation', 'es-g24-innovations', 'es-g2e-glass2energy', 'es-gaiasens-technologies',
                        'es-grz-technologies', 'es-insolight', 'es-kaemco', 'es-power-vision-engineering',
                        'es-solaronix', 'es-swiss-inso', 'es-twentygreen', 'es-urbio']

    # ontology_seed_concept_ids = [100089, 10091, 1021673, 1407416, 1443002, 1451418, 14967273, 1531457, 155961, 1633173,
    #                              166164, 1777495, 17996959, 18413531, 1860870, 1866009, 188418, 19468941, 208999, 210537,
    #                              23099899, 2364800, 24458151, 25271012, 29501, 301500, 31666505, 3407706, 36804997,
    #                              375743, 45434, 45441, 45446, 45795, 479209, 49542583, 5030939, 66618, 994228, 1038280,
    #                              11153863, 12545698, 12637359, 14207810, 15030, 15143713, 15532928, 16775, 1693139,
    #                              1784516, 21072589, 2119174, 2119179, 21782795, 2263904, 2322153, 2423894, 26515241,
    #                              27447253, 27447324, 30242372, 31898, 3201, 32178, 325232, 326324, 33775893, 351661,
    #                              367276, 3679268, 37104, 41994482, 42673705, 4538124, 4607152, 46255716, 5042951,
    #                              5362623, 5869611, 60779659, 6905345, 7694290, 7726829, 84107, 8941842, 9528025]

    # ontology_seed_concept_ids = [3679268, 5042951, 4538124, 12637359, 23099899, 9528025, 18413531, 29501]

    # gpt_seed_concept_ids = [18413531, 29501, 11561332, 16865582, 216143, 1344439, 1002744, 12675756, 1055890, 4036973,
    #                         5042951, 398356, 301500, 2263904, 31666505, 43543229, 21652112, 316966, 66133338, 33814239,
    #                         4473245, 33982881, 10006949, 8436001, 18136001, 3512226, 28730644, 42431318, 51497431,
    #                         460253, 25784, 13578019, 62468664, 59860313, 237192, 70323350, 52634071, 2364800, 17117300,
    #                         62385249, 24142446, 5036087]

    # gpt_seed_concept_ids = [1055890, 25784]

    seed_concept_ids = [3679268, 5042951, 4538124, 12637359, 23099899, 9528025, 18413531, 29501, 1055890, 25784,
                        33094374]

    # excluded_concept_ids = [66618, 31898, 26515241, 52634071, 30242372, 7726829, 27447253, 21782795, 16775, 326324,
    #                         32178, 3201, 15030, 60779659, 994228, 1021673, 12545698, 1407416, 15143713, 7694290,
    #                         11153863, 27447324, 19468941, 3512226, 24458151, 1777495, 166164, 14967273, 188418,
    #                         49542583, 155961, 325232, 398356, 14207810, 1860870, 2119174, 46255716, 4607152, 2119179,
    #                         84107, 1693139, 479209, 4473245, 237192, 45795, 301500, 45446, 2322153, 2263904, 21072589,
    #                         5362623, 17996959, 8941842, 2364800, 25271012, 45434, 375743, 1866009, 17554500, 33775893,
    #                         6905345, 31666505, 3407706, 15532928, 37104, 351661, 36804997, 28730644, 10091, 2423894,
    #                         42673705, 1784516, 9028960, 26833, 1002744, 216143, 5030939, 1344439, 1443002, 1451418,
    #                         3694602, 1038280, 212253, 277237, 208999, 210537, 2518458, 1225002, 3378256, 2769817,
    #                         72576, 24714, 25065, 395167, 487226, 41994482, 4036973, 1134, 19022, 8267, 154665, 2925093,
    #                         241812, 10474719, 25997913, 548173, 1908142, 609614, 367276, 31146, 1638605, 308906, 48180,
    #                         1886820, 9559, 13764124, 1105937, 375416, 460253, 8286675, 146103, 28748]

    excluded_concept_ids = [72576, 3694602, 24714, 1105937, 241812, 1866009, 2769817, 13764124, 212253, 395167,
                            2925093, 154665, 308906, 1225002, 31146, 1908142, 48180, 146103, 487226, 2518458, 17554500,
                            8267, 28748, 1638605, 19022, 548173, 3378256, 26833, 609614, 8286675, 9559, 25997913,
                            10474719, 9028960, 1886820, 25065, 1134, 277237, 375416]

    gexf_filename = 'startups.gexf'

    ############################################################

    bc.log('Fetching startups data from database...')

    # Startup nodes
    table_name = 'graph_piper.Nodes_N_EPFLStartup'
    fields = ['EPFLStartupID', 'StartupName']
    conditions = {'Status': 'Private'}
    startups = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

    # Startup-founder edges
    table_name = 'graph_piper.Edges_N_EPFLStartup_N_Person_T_Founder'
    fields = ['EPFLStartupID', 'SCIPER']
    startups_founders = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)
    founder_ids = list(startups_founders['SCIPER'].drop_duplicates())

    # Startup-professor edges
    table_name = 'graph_piper.Edges_N_EPFLStartup_N_Person_T_Professor'
    fields = ['EPFLStartupID', 'SCIPER']
    startups_professors = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)
    professor_ids = list(startups_professors['SCIPER'].drop_duplicates())

    # Person-concept edges
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

    bc.log('Extracting list of all concepts...')

    # We use only founders' concepts as they are more likely to be close to the startup activity than the professor ones
    startup_seed_concept_ids = list(startups_founders_concepts.loc[startups_founders_concepts['EPFLStartupID'].isin(seed_startup_ids), 'PageID'].drop_duplicates())

    concept_ids = list(set(seed_concept_ids) | set(startup_seed_concept_ids) - set(excluded_concept_ids))

    ############################################################

    bc.log('Fetching concept data from database...')

    table_name = 'graph_piper.Nodes_N_Concept'
    fields = ['PageID', 'PageTitle']
    conditions = {'PageID': concept_ids}
    concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

    # Update concept ids with fetched nodes
    concept_ids = list(concepts['PageID'])

    table_name = 'graph_piper.Edges_N_Concept_N_Concept_T_GraphScore'
    fields = ['SourcePageID', 'TargetPageID', 'NormalisedScore']
    conditions = {'SourcePageID': concept_ids, 'TargetPageID': concept_ids}
    concepts_concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

    ############################################################

    bc.log('Restricting data to fetched nodes...')

    startups_concepts = pd.merge(startups_concepts, concepts[['PageID']], how='inner', on='PageID')
    startups_concepts = pd.merge(startups_concepts, startups[['EPFLStartupID']], how='inner', on='EPFLStartupID')

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

    # bc.log('Filtering concept-concept edges based on score...')
    #
    # # Keep only edges with high score
    # proportion = 0.1    # keep 10% of all edges
    # concepts_concepts = concepts_concepts.sort_values(by='NormalisedScore', ascending=False)
    # concepts_concepts = concepts_concepts.head(int(proportion * len(concepts_concepts)))
    # concepts_concepts = concepts_concepts.reset_index(drop=True)

    ############################################################

    bc.log('Removing isolated nodes...')

    startups = pd.merge(startups, startups_concepts[['EPFLStartupID']].drop_duplicates(), how='inner', on='EPFLStartupID')
    concepts = pd.merge(
        concepts,
        pd.concat([
            startups_concepts[['PageID']],
            concepts_concepts[['SourcePageID']].rename(columns={'SourcePageID': 'PageID'}),
            concepts_concepts[['TargetPageID']].rename(columns={'TargetPageID': 'PageID'})
        ]).drop_duplicates(),
        how='inner',
        on='PageID'
    )

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
