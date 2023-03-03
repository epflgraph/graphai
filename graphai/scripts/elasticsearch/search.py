from graphai.core.interfaces import ES
from graphai.core.utils.text import pprint

es = ES('wikipages')

r = es.search('dude')
pprint(r)
