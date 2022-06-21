from concept_detection.text.io import pprint
from interfaces.es import ES

es = ES('wikipages')

r = es.search('dude')
pprint(r)
