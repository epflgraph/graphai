import pandas as pd

from db_cache_manager.db import DB

from graphai.core.common.config import config


def get_openalex_nearest(category_id=None, topic_id=None):
    # Return if no conditions
    if category_id is None and topic_id is None:
        return []

    # Try to fix topic id if it starts with a `T`
    if topic_id is not None and topic_id.startswith('T'):
        topic_id = topic_id[1:]

    # Set up conditions according to parameters
    conditions = {}

    if category_id is not None:
        conditions['category_id'] = category_id

    if topic_id is not None:
        conditions['topic_id'] = topic_id

    # Define table and fields to be fetched
    table_name = 'data_augmentation.Openalex_Categories_Topics'
    fields = ['category_id', 'category_name', 'topic_id', 'topic_name', 'embedding_score', 'wikipedia_score', 'score']

    # Instantiate database connector
    db = DB(config['database'])

    # Fetch rows that match the conditions
    matches = pd.DataFrame(db.find(table_name, fields=fields, conditions=conditions), columns=fields)

    # Sort by descending score
    matches = matches.sort_values(by='score', ascending=False)

    return matches.to_dict(orient='records')
