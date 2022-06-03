from concept_detection.interfaces.es import ES

from concept_detection.text.io import read_json

es = ES()

settings = read_json('config/settings.json')
mapping = read_json('config/mapping.json')

# es.delete_index()
es.create_index(settings=settings, mapping=mapping)
