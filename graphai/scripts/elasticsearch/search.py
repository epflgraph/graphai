from core.interfaces.es import ES
from core.utils.text.io import pprint

es = ES('wikipages')

r = es.search('dude')
pprint(r)
