import ray

import pandas as pd

from core.interfaces.db import DB
from core.interfaces.es import ES
from core.utils.breadcrumb import Breadcrumb

# Init ray
# ray.init(namespace="populate_elasticsearch", include_dashboard=False, log_to_driver=True)

indices = ['concepts']


@ray.remote
class PopulateActor:
    def __init__(self):
        self.ess = [ES(index) for index in indices]

    def index(self, doc):
        for es in self.ess:
            es.index_doc(doc)


# Instantiate ray actor list
n_actors = 16
actors = [PopulateActor.remote() for i in range(n_actors)]

# Init breadcrumb to log and time statements
bc = Breadcrumb()

# Init DB interface
db = DB()

# Fetch all concepts
bc.log('Fetching concepts...')

table_name = 'graph.Nodes_N_Concept'
fields = ['PageID', 'PageTitle', 'PageContent']
concepts = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)
concept_ids = list(concepts['PageID'])

# Fetch concepts extended content
bc.log('Fetching concepts extended content...')
table_name = 'Campus_Analytics.Page_Title_and_Content_Reduced_No_Macros'
fields = ['PageID', 'Headings', 'OpeningText', 'AuxiliaryText', 'Redirects', 'Popularity']
conditions = {'PageID': concept_ids}
extended_concepts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

# Merge concepts with extended_concepts
concepts = pd.merge(concepts, extended_concepts, how='left', on='PageID')

# Coerce missing values to valid values
concepts['Headings'] = concepts['Headings'].fillna('').str.split(',').apply(list)
concepts['OpeningText'] = concepts['OpeningText'].fillna('')
concepts['AuxiliaryText'] = concepts['AuxiliaryText'].fillna('')
concepts['Redirects'] = concepts['Redirects'].fillna('').str.split(',').apply(list)
concepts['Popularity'] = concepts['Popularity'].fillna(0)

# Fetch categories
bc.log('Fetching categories...')

table_name = 'graph.Edges_N_Concept_N_Category'
fields = ['PageID', 'CategoryID']
conditions = {'PageID': concept_ids}
concepts_categories = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)
category_ids = list(concepts_categories['CategoryID'].drop_duplicates())

table_name = 'graph.Nodes_N_Category'
fields = ['CategoryID', 'CategoryTitle']
conditions = {'CategoryID': category_ids}
categories = pd.DataFrame(db.find(table_name, fields=fields), columns=fields)

# Merging concepts with categories
bc.log('Merging concepts with categories...')
concepts_categories = pd.merge(concepts_categories, categories, how='inner', on='CategoryID')
concepts_categories = concepts_categories.groupby(by='PageID').aggregate(Categories=('CategoryTitle', list)).reset_index()
concepts = pd.merge(concepts, concepts_categories, how='left', on='PageID')

# Coerce NaN values (pages without categories) to empty lists
concepts['Categories'] = concepts['Categories'].fillna('').apply(list)

# Index documents
results = []
actor = 0
for i, concept in concepts.iterrows():
    if i % 5000 == 0:
        bc.log(f'Indexing documents {i}-{min(i + 5000, len(concepts))}...')

    # Prepare document
    doc = {
        'id': concept['PageID'],
        'title': concept['PageTitle'],
        'text': concept['PageContent'],
        'heading': concept['Headings'],
        'opening_text': concept['OpeningText'],
        'auxiliary_text': concept['AuxiliaryText'],
        'file_text': '',                                # TODO change to concept['FileText'] when available
        'redirect': concept['Redirects'],
        'category': concept['Categories'],
        'incoming_links': 1,                            # TODO change to concept['IncomingLinks'] when available
        'popularity_score': concept['Popularity']
    }

    # Index document
    results.append(actors[actor].index.remote(doc))

    # Update actor index
    actor = (actor + 1) % n_actors

# Wait for the results
results = ray.get(results)
