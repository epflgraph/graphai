from interfaces.es import ES

from utils.text.io import read_json

index = 'wikipages_1_shards'
es = ES(index)

settings = read_json('config/settings.json')
settings['number_of_shards'] = 1
mapping = read_json('config/mapping.json')

es.delete_index()
es.create_index(settings=settings, mapping=mapping)
