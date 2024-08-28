from elasticsearch_interface.es import ES

from graphai.core.common.config import config

from graphai.core.utils.time.date import now
from graphai.core.utils.text.io import read_json

number_of_shards = 1

index = f'aitor_concepts_{now().date()}'
es = ES(config['elasticsearch'], index)

settings = read_json('config/settings.json')
settings['number_of_shards'] = number_of_shards
mapping = read_json('config/mapping.json')

es.delete_index()
es.create_index(settings=settings, mapping=mapping)
