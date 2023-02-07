import ray

import pandas as pd

from interfaces.db import DB
from interfaces.es import ES
from utils.text.markdown import strip
from utils.breadcrumb import Breadcrumb

# Init ray
ray.init(namespace="populate_elasticsearch", include_dashboard=False, log_to_driver=True)

indices = ['concepts']


@ray.remote
class PopulateActor:
    def __init__(self):
        self.ess = [ES(index) for index in indices]

    # def strip_and_index(self, page_id, page, page_categories):
    #     # Strip page content
    #     stripped_page = strip(page['content'])
    #
    #     # Prepare document to index
    #     doc = {
    #         'id': page_id,
    #         'title': page['title'],
    #         'text': stripped_page['text'],
    #         'heading': stripped_page['heading'],
    #         'opening_text': stripped_page['opening_text'],
    #         'auxiliary_text': stripped_page['auxiliary_text'],
    #         'file_text': '',
    #         'redirect': page['redirect'],
    #         'category': page_categories,
    #         'incoming_links': 1,
    #         'popularity_score': page['popularity']
    #     }
    #
    #     # Index on elasticsearch
    #     for es in self.ess:
    #         es.index_doc(doc)

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
bc.log('Indexing documents...')
results = []
actor = 0
for i, concept in concepts.iterrows():
    # Prepare document
    doc = {
        'id': concept['PageID'],
        'title': concept['PageTitle'],
        'text': concept['PageContent'],         # TODO change to concept['Text'] when available
        'heading': [],                          # TODO change to concept['Heading'] when available
        'opening_text': '',                     # TODO change to concept['OpeningText'] when available
        'auxiliary_text': '',                   # TODO change to concept['AuxiliaryText'] when available
        'file_text': '',                        # TODO change to concept['FileText'] when available
        'redirect': [],                         # TODO change to concept['Redirect'] when available
        'category': concept['Categories'],
        'incoming_links': 1,                    # TODO change to concept['IncomingLinks'] when available
        'popularity_score': 1,                  # TODO change to concept['Popularity'] when available
    }


    # Index document
    results.append(actors[actor].index.remote(doc))

    # Update actor index
    actor = (actor + 1) % n_actors

# Wait for the results
results = ray.get(results)
