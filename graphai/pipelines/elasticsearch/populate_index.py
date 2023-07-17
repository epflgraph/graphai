"""
Script that populates an elasticsearch index named "aitor_concepts_<date>" containing concepts in the mediawiki format
for use in concept detection.

All data comes directly from the "Campus_Analytics" schema so this script can be run at any point in the pipeline after the
creation of said schema. Should it use data from the "graph" schema, it could only be run at the very end of the pipeline.
"""

from multiprocessing import Pool

import pandas as pd

from graphai.core.interfaces.db import DB
from graphai.core.interfaces.es import ES
from graphai.core.utils.time.date import now
from graphai.core.utils.breadcrumb import Breadcrumb

################################################################

# Init ES interface for index 'aitor_concepts_YYYY-mm-DD'
es = ES(f'aitor_concepts_{now().date()}')


def index_doc(doc):
    es.index_doc(doc)

################################################################


if __name__ == '__main__':
    pd.set_option('display.max_rows', 400)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    ################################################################

    # Init DB interface
    db = DB()

    # Init breadcrumb to log and time statements
    bc = Breadcrumb()

    # Fetch concepts
    bc.log('Fetching concepts...')
    table_name = 'Campus_Analytics.Pages_Neighbours'
    fields = ['PageID', 'PageTitleDisplay', 'Popularity', 'IncomingLinks']
    columns = ['PageID', 'PageTitle', 'PopularityScore', 'IncomingLinks']
    all_concepts = pd.DataFrame(db.find(table_name, fields=fields), columns=columns).fillna(0)
    all_concept_ids = list(all_concepts['PageID'])

    pool = Pool()

    batch_size = 10000
    concept_batches = [all_concepts[i: i + batch_size] for i in range(0, len(all_concepts), batch_size)]
    for concepts in concept_batches:
        bc.log('Processing concepts batch...')
        bc.indent()

        concept_ids = list(concepts['PageID'])

        # Fetch content
        bc.log('Fetching content...')
        table_name = 'Campus_Analytics.Page_Content_Full'
        fields = ['PageID', 'PageContent', 'OpeningText']
        columns = ['PageID', 'Text', 'OpeningText']
        conditions = {'PageID': concept_ids}
        contents = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=columns).fillna('')

        # Fetch auxiliary texts
        bc.log('Fetching auxiliary texts...')
        table_name = 'Campus_Analytics.Page_Auxiliary_Text'
        fields = ['PageID', 'AuxiliaryText']
        conditions = {'PageID': concept_ids}
        aux_texts = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields).dropna()
        aux_texts = aux_texts.groupby('PageID').aggregate(AuxiliaryTexts=('AuxiliaryText', list)).reset_index()

        # Fetch headings
        bc.log('Fetching headings...')
        table_name = 'Campus_Analytics.Page_Headings'
        fields = ['PageID', 'Heading']
        columns = ['PageID', 'Heading']
        conditions = {'PageID': concept_ids}
        headings = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=columns).dropna()
        headings = headings.groupby('PageID').aggregate(Headings=('Heading', list)).reset_index()

        # Fetch redirects
        bc.log('Fetching redirects...')
        table_name = 'Campus_Analytics.Redirects'
        fields = ['TargetPageID', 'AliasTitle']
        columns = ['PageID', 'Redirect']
        conditions = {'TargetPageID': concept_ids}
        redirects = db.find(table_name, fields=fields, conditions=conditions)
        redirects = [(page_id, redirect.decode('utf-8')) for page_id, redirect in redirects]
        redirects = pd.DataFrame(redirects, columns=columns)
        redirects['Redirect'] = redirects['Redirect'].str.replace('_', ' ')
        redirects = redirects.groupby('PageID').aggregate(Redirects=('Redirect', list)).reset_index()

        # Fetch WikiCategories
        bc.log('Fetching wikicategories...')
        table_name = 'Campus_Analytics.PageID_to_CategoriesID_Mapping'
        fields = ['PageID', 'CategoryID']
        columns = ['PageID', 'WikiCategoryID']
        conditions = {'PageID': concept_ids}
        concepts_categories = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=columns)
        concept_category_ids = list(concepts_categories['WikiCategoryID'].drop_duplicates())

        table_name = 'Campus_Analytics.Categories'
        fields = ['CategoryID', 'CategoryTitle']
        columns = ['WikiCategoryID', 'WikiCategoryTitle']
        conditions = {'CategoryID': concept_category_ids}
        categories = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=columns)
        categories['WikiCategoryTitle'] = categories['WikiCategoryTitle'].str.replace('_', ' ')

        categories = pd.merge(concepts_categories, categories, how='inner', on='WikiCategoryID')
        categories = categories.groupby('PageID').aggregate(WikiCategories=('WikiCategoryTitle', list)).reset_index()

        # Merge concepts with the rest of the data
        bc.log('Merging all dataframes...')
        concepts = pd.merge(concepts, contents, how='left', on='PageID')
        concepts = pd.merge(concepts, aux_texts, how='left', on='PageID').fillna('').apply(list)
        concepts = pd.merge(concepts, headings, how='left', on='PageID').fillna('').apply(list)
        concepts = pd.merge(concepts, redirects, how='left', on='PageID').fillna('').apply(list)
        concepts = pd.merge(concepts, categories, how='left', on='PageID').fillna('').apply(list)

        # Prepare documents
        bc.log('Preparing documents...')
        docs = [
            {
                'id': concept['PageID'],
                'title': concept['PageTitle'],
                'text': concept['Text'],
                'opening_text': concept['OpeningText'],
                'auxiliary_text': concept['AuxiliaryTexts'],
                'file_text': '',  # TODO change to concept['FileText'] if/when available
                'heading': concept['Headings'],
                'redirect': concept['Redirects'],
                'category': concept['WikiCategories'],
                'popularity_score': concept['PopularityScore'],
                'incoming_links': concept['IncomingLinks']
            }
            for i, concept in concepts.iterrows()
        ]

        bc.log('Indexing documents...')
        pool.map(index_doc, docs)

        bc.outdent()

    pool.close()
    pool.join()
