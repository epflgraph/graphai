from interfaces.es import ES
from utils.text.io import pprint

es = ES('wikipages')

r = es.search('dude')
pprint(r)
