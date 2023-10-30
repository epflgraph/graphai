import pandas as pd
import networkx as nx

from graphai.core.interfaces.db_cache_manager import DB
from graphai.core.interfaces.config_loader import load_db_config

from graphai.core.utils.breadcrumb import Breadcrumb


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
    db = DB(load_db_config())

    ############################################################

    startups_concepts = pd.DataFrame(
        [['es-aeler-technologies', 77547],
         ['es-aeler-technologies', 22849124],
         ['es-aica', 20903754],
         ['es-aica', 775],
         ['es-aica', 1164],
         ['es-aicrowd', 1164],
         ['es-aicrowd', 233488],
         ['es-aicrowd', 35458904],
         ['es-anemomind', 36674345],
         ['es-anemomind', 33496160],
         ['es-anemomind', 27672],
         ['es-antia-therapeutics', 105219],
         ['es-antia-therapeutics', 75654],
         ['es-aqua-tech', 23001],
         ['es-aqua-tech', 33306],
         ['es-aurora-s-grid', 367397],
         ['es-aurora-s-grid', 24130],
         ['es-bionomous', 36624],
         ['es-bionomous', 1164],
         ['es-bionomous', 156428],
         ['es-bloom-biorenewables', 7906908],
         ['es-bloom-biorenewables', 23001],
         ['es-bloom-biorenewables', 6911],
         ['es-bluebotics', 21854],
         ['es-bluebotics', 32410],
         ['es-bluebotics', 20903754],
         ['es-bluewatt-engineering', 33306],
         ['es-bluewatt-engineering', 5309],
         ['es-calerga', 5309],
         ['es-calerga', 5311],
         ['es-calerga', 43444],
         ['es-camptocamp-sa', 5309],
         ['es-camptocamp-sa', 36674345],
         ['es-camptocamp-sa', 1404353],
         ['es-cell-caps-sa', 844290],
         ['es-cell-caps-sa', 40017873],
         ['es-cfs-engineering', 43444],
         ['es-cfs-engineering', 19559],
         ['es-cfs-engineering', 8386],
         ['es-combo-solutions', 239038],
         ['es-combo-solutions', 1164],
         ['es-combo-solutions', 35458904],
         ['es-comphya-treated', 14783],
         ['es-comphya-treated', 1363291],
         ['es-comppair', 67973511],
         ['es-comppair', 51892],
         ['es-comppair', 157616],
         ['es-comppair', 58890],
         ['es-comppair', 5824713],
         ['es-cyberbotics', 5309],
         ['es-cyberbotics', 43444],
         ['es-cyberbotics', 20903754],
         ['es-depoly', 26145195],
         ['es-depoly', 70157],
         ['es-depoly', 23001],
         ['es-earlysight', 56153],
         ['es-earlysight', 48334],
         ['es-earlysight', 1363291],
         ['es-ecointesys', 10091],
         ['es-ecointesys', 29278],
         ['es-enairys-powertech-sa', 381774],
         ['es-enairys-powertech-sa', 24130],
         ['es-enairys-powertech-sa', 195891],
         ['es-eneftech-corporation', 24130],
         ['es-eneftech-corporation', 25784],
         ['es-epiqr-renovation', 21296224],
         ['es-epiqr-renovation', 3183830],
         ['es-epiqr-renovation', 9649],
         ['es-esmart-technologies', 184978],
         ['es-esmart-technologies', 9649],
         ['es-esmart-technologies', 238420],
         ['es-estia', 26616666],
         ['es-estia', 29501],
         ['es-estia', 1804464],
         ['es-estia', 21296224],
         ['es-excellgene', 23634],
         ['es-excellgene', 7955],
         ['es-excellgene', 619632],
         ['es-fastcom-technology', 41684],
         ['es-fastcom-technology', 18985040],
         ['es-fiveco', 36674345],
         ['es-fiveco', 20903754],
         ['es-flyability', 299871],
         ['es-flyability', 252712],
         ['es-foldaway-haptics', 20903754],
         ['es-foldaway-haptics', 300409],
         ['es-force-dimension-sarl', 300409],
         ['es-force-dimension-sarl', 20903754],
         ['es-g24-innovations', 2352910],
         ['es-g24-innovations', 27743],
         ['es-g2e-glass2energy', 2352910],
         ['es-g2e-glass2energy', 27743],
         ['es-gaiasens-technologies', 37039],
         ['es-gaiasens-technologies', 27643777],
         ['es-gaiasens-technologies', 28191],
         ['es-gaiasens-technologies', 10934212],
         ['es-gamaya', 627],
         ['es-gamaya', 228148],
         ['es-gamaya', 1164],
         ['es-gctronic', 20903754],
         ['es-gctronic', 19559],
         ['es-gctronic', 9663],
         ['es-geoeg', 113728],
         ['es-geoeg', 935041],
         ['es-gridsteer', 367397],
         ['es-gridsteer', 20344155],
         ['es-gridsteer', 24130],
         ['es-grosso-link', 1515653],
         ['es-grosso-link', 19377],
         ['es-grz-technologies', 13255],
         ['es-grz-technologies', 24130],
         ['es-grz-technologies', 6402491],
         ['es-hydromea', 33306],
         ['es-hydromea', 20903754],
         ['es-hydromea', 14552094],
         ['es-imina-technologies', 20903754],
         ['es-imina-technologies', 21488],
         ['es-imina-technologies', 19567],
         ['es-imperix', 9663],
         ['es-imperix', 275473],
         ['es-imperix', 39413611],
         ['es-insolight', 27743],
         ['es-insolight', 627],
         ['es-insolight', 20344155],
         ['es-ipogee', 36674345],
         ['es-ipogee', 12398],
         ['es-ipogee', 660850],
         ['es-ithetis', 32654],
         ['es-ithetis', 9311172],
         ['es-kaemco', 43444],
         ['es-kaemco', 935041],
         ['es-kaemco', 5309],
         ['es-kaemco', 12398],
         ['es-karmic-microandnano-sarl', 156428],
         ['es-karmic-microandnano-sarl', 21488],
         ['es-karmic-microandnano-sarl', 32268692],
         ['es-l-e-s-s', 188386],
         ['es-l-e-s-s', 21488],
         ['es-l-e-s-s', 3372377],
         ['es-lantern-solutions-sarl', 36674345],
         ['es-lantern-solutions-sarl', 9252],
         ['es-ligentec', 6424117],
         ['es-ligentec', 3328072],
         ['es-lumendo', 8005],
         ['es-lumendo', 17939],
         ['es-lumendo', 363430],
         ['es-lymphatica', 1363291],
         ['es-lymphatica', 71425],
         ['es-lyncee-tec', 19567],
         ['es-lyncee-tec', 156428],
         ['es-lyncee-tec', 66338],
         ['es-mecartex', 156428],
         ['es-mecartex', 1260106],
         ['es-mecartex', 19559],
         ['es-medaxis-sa', 1363291],
         ['es-medaxis-sa', 33306],
         ['es-medaxis-sa', 514458],
         ['es-medlight', 1363291],
         ['es-medlight', 291838],
         ['es-medlight', 421149],
         ['es-medusoil', 37738],
         ['es-medusoil', 693334],
         ['es-medusoil', 44603],
         ['es-mobsya', 20903754],
         ['es-mobsya', 9252],
         ['es-mobsya', 59126142],
         ['es-nanolive', 21488],
         ['es-nanolive', 4230],
         ['es-nanolive', 346382],
         ['es-neural-concept', 233488],
         ['es-neural-concept', 9251],
         ['es-neural-concept', 30873116],
         ['es-oculight-dynamics', 75987],
         ['es-oculight-dynamics', 935041],
         ['es-odoma', 18985040],
         ['es-odoma', 1164],
         ['es-odoma', 1462876],
         ['es-omnisens', 3372377],
         ['es-omnisens', 235757],
         ['es-online-control', 2720954],
         ['es-online-control', 275473],
         ['es-ozwe', 5363],
         ['es-ozwe', 32612],
         ['es-pbandb', 16413778],
         ['es-pbandb', 42048],
         ['es-pbandb', 17940],
         ['es-pn-solutions', 9663],
         ['es-pn-solutions', 15150],
         ['es-pn-solutions', 49420],
         ['es-pomelo', 240583],
         ['es-pomelo', 9611],
         ['es-power-vision-engineering', 43444],
         ['es-power-vision-engineering', 381399],
         ['es-power-vision-engineering', 9251],
         ['es-prediggo', 9611],
         ['es-prediggo', 233488],
         ['es-pristem', 1363291],
         ['es-pristem', 34197],
         ['es-program', 38360943],
         ['es-program', 19559],
         ['es-program', 149993],
         ['es-rayform-caustics', 17939],
         ['es-rayform-caustics', 3498698],
         ['es-rayform-caustics', 752],
         ['es-rheon-medical', 74748],
         ['es-rheon-medical', 1363291],
         ['es-rheon-medical', 1099256],
         ['es-rovenso', 20903754],
         ['es-rovenso', 252712],
         ['es-rovenso', 41684],
         ['es-senis', 235757],
         ['es-senis', 27643777],
         ['es-senis', 9532],
         ['es-sensima-inspection', 9532],
         ['es-sensima-inspection', 235757],
         ['es-sensima-inspection', 21188370],
         ['es-sensimed', 74845],
         ['es-sensimed', 56153],
         ['es-sensimed', 1363291],
         ['es-sensimed', 74748],
         ['es-sensorscope', 27643777],
         ['es-sensorscope', 235757],
         ['es-sensorscope', 9251],
         ['es-sensorscope', 33172],
         ['es-shockfish', 5309],
         ['es-shockfish', 9451],
         ['es-shockfish', 24336516],
         ['es-smarthelio', 27743],
         ['es-smarthelio', 1164],
         ['es-smarthelio', 235757],
         ['es-smarthelio', 12057519],
         ['es-solaronix', 27743],
         ['es-solaronix', 2352910],
         ['es-suriasis', 1910996],
         ['es-suriasis', 1176900],
         ['es-suriasis', 8005],
         ['es-swiss-inso', 27743],
         ['es-swiss-inso', 12581],
         ['es-swiss-inso', 935041],
         ['es-swiss-inso', 5921],
         ['es-swisslumix', 184897],
         ['es-swisslumix', 234714],
         ['es-swisspod-technologies', 18580879],
         ['es-swisspod-technologies', 36971117],
         ['es-swisspod-technologies', 32502],
         ['es-swissto12', 42852],
         ['es-swissto12', 1305947],
         ['es-swissto12', 45207],
         ['es-synova', 17556],
         ['es-synova', 2366787],
         ['es-synova', 365765],
         ['es-thinkee', 12057519],
         ['es-thinkee', 5309],
         ['es-thinkee', 33496160],
         ['es-treatech-sarl', 210555],
         ['es-treatech-sarl', 4481763],
         ['es-treatech-sarl', 37738],
         ['es-treatech-sarl', 33306],
         ['es-twentygreen', 731740],
         ['es-twentygreen', 14193930],
         ['es-twentygreen', 627],
         ['es-urbio', 935041],
         ['es-urbio', 9649],
         ['es-urbio', 233488],
         ['es-valais-perovskite-solar', 2352910],
         ['es-valais-perovskite-solar', 19290728],
         ['es-volumen', 20903754],
         ['es-volumen', 233488],
         ['es-volumen', 386407],
         ['es-xsensio-sarl', 23770249],
         ['es-xsensio-sarl', 21488],
         ['es-xsensio-sarl', 3954],
         ['es-xsensio-sarl', 18906],
         ['es-zaphiro-technologies', 20344155],
         ['es-zaphiro-technologies', 5199213],
         ['es-zaphiro-technologies', 5309]],
        columns=['EPFLStartupID', 'PageID']
    )

    startup_ids = ['es-aeler-technologies', 'es-aqua-tech', 'es-aurora-s-grid', 'es-bloom-biorenewables', 'es-bluewatt-engineering', 'es-combo-solutions', 'es-comppair', 'es-depoly', 'es-ecointesys', 'es-enairys-powertech-sa', 'es-eneftech-corporation', 'es-epiqr-renovation', 'es-esmart-technologies', 'es-estia', 'es-flyability', 'es-g24-innovations', 'es-g2e-glass2energy', 'es-gaiasens-technologies', 'es-gamaya', 'es-geoeg', 'es-gridsteer', 'es-grz-technologies', 'es-hydromea', 'es-insolight', 'es-kaemco', 'es-medusoil', 'es-oculight-dynamics', 'es-power-vision-engineering', 'es-rovenso', 'es-sensima-inspection', 'es-sensorscope', 'es-smarthelio', 'es-solaronix', 'es-swiss-inso', 'es-swisspod-technologies', 'es-treatech-sarl', 'es-twentygreen', 'es-urbio', 'es-valais-perovskite-solar', 'es-volumen', 'es-zaphiro-technologies']

    startups_concepts = startups_concepts[startups_concepts['EPFLStartupID'].isin(startup_ids)].reset_index(drop=True)

    concept_ids = list(startups_concepts['PageID'].drop_duplicates())

    gexf_filename = 'startups.gexf'

    ############################################################

    bc.log('Fetching startups data from database...')

    # Startup nodes
    table_name = 'graph_piper.Nodes_N_EPFLStartup'
    fields = ['EPFLStartupID', 'StartupName']
    conditions = {'EPFLStartupID': startup_ids}
    startups = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

    ############################################################

    bc.log('Fetching concept data from database...')

    table_name = 'graph_piper.Nodes_N_Concept_T_Title'
    fields = ['PageID', 'PageTitle']
    conditions = {'PageID': concept_ids}
    concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

    # Update concept ids with fetched nodes
    concept_ids = list(concepts['PageID'])

    table_name = 'graph_piper.Edges_N_Concept_N_Concept_T_GraphScore'
    fields = ['SourcePageID', 'TargetPageID', 'NormalisedScore']
    conditions = {'SourcePageID': concept_ids, 'TargetPageID': concept_ids, 'NormalisedScore': {'>=': 0.4}}
    concepts_concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

    ############################################################

    bc.log('Recomputing startup-concept edge score...')

    # Divide score by the frequency of each concept
    startups_concepts = pd.merge(
        startups_concepts,
        startups_concepts.groupby(by='PageID').aggregate(PageStartupCount=('EPFLStartupID', 'count')).reset_index(),
        how='left',
        on='PageID'
    )
    startups_concepts['Score'] = 1 / startups_concepts['PageStartupCount']
    startups_concepts = startups_concepts[['EPFLStartupID', 'PageID', 'Score']]

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
