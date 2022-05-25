from concept_detection.text.io import pprint
from concept_detection.interfaces.es import ES

es = ES()

r = es.search('dude')
pprint(r)
