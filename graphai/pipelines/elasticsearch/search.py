from graphai.core.interfaces.es import ES
from graphai.core.utils.text.io import pprint

es = ES('wikipages')

r = es.search('dude')
pprint(r)
