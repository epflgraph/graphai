from interfaces.es import ES

from concept_detection.text.io import read_json

index = 'wikipages'
es = ES(index)

settings = read_json('config/settings.json')
mapping = read_json('config/mapping.json')

es.delete_index()
es.create_index(settings=settings, mapping=mapping)
